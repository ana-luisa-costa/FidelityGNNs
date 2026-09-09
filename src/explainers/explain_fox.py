
from __future__ import annotations

import argparse
import itertools
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdflib import Graph
from tqdm import tqdm

from src.explainers._shared import (
    ExplainerContext,
    _enriched_label,
    _entity_category,
    _entity_short_label,
    _pair_side_caption,
    _plot_global_category,
    _plot_top_entities,
    _scalar_for_model_task,
    _scalar_for_task,
    add_common_explainer_args,
    load_explainer_context,
    setup_matplotlib_style,
    write_summary_json,
)
from src.gnn4ppm.train import get_event_attrs

setup_matplotlib_style()

class _GaussMF(nn.Module):
    def __init__(self, mu: float, sigma: float):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float))
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((x - self.mu) ** 2) / (2 * self.sigma.clamp(min=1e-6) ** 2))

class _FuzzifyVar(nn.Module):
    def __init__(self, centers: List[float], sigma: float):
        super().__init__()
        self.mfs = nn.ModuleList([_GaussMF(c, sigma) for c in centers])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([mf(x) for mf in self.mfs], dim=1)

class _FuzzifyLayer(nn.Module):
    def __init__(self, num_vars: int, num_mfs: int):
        super().__init__()
        sigma = 1.0 / max(num_mfs, 2)
        centers = [i / max(num_mfs - 1, 1) for i in range(num_mfs)]
        self.vars = nn.ModuleList([_FuzzifyVar(centers, sigma) for _ in range(num_vars)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([v(x[:, i]) for i, v in enumerate(self.vars)], dim=1)

class _AntecedentLayer(nn.Module):
    def __init__(self, num_vars: int, num_mfs: int):
        super().__init__()
        indices = list(itertools.product(range(num_mfs), repeat=num_vars))
        self.register_buffer("mf_idx", torch.tensor(indices, dtype=torch.long))

    def forward(self, fuzzified: torch.Tensor) -> torch.Tensor:
        idx = self.mf_idx
        gathered = torch.stack(
            [fuzzified[:, v, idx[:, v]] for v in range(fuzzified.size(1))],
            dim=2,
        )
        return gathered.prod(dim=2)

class _ConsequentLayer(nn.Module):
    def __init__(self, num_rules: int):
        super().__init__()
        self.coeff = nn.Parameter(torch.zeros(num_rules))

    def forward(self, raw_weights: torch.Tensor) -> torch.Tensor:
        norm = raw_weights / raw_weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return (norm * self.coeff).sum(dim=1)

class AnfisNet(nn.Module):
    def __init__(self, num_vars: int, num_mfs: int = 3):
        super().__init__()
        self.num_vars = num_vars
        self.num_mfs = num_mfs
        self.fuzzify = _FuzzifyLayer(num_vars, num_mfs)
        self.antecedent = _AntecedentLayer(num_vars, num_mfs)
        self.consequent = _ConsequentLayer(num_mfs ** num_vars)
        self._mf_idx: List[Tuple[int, ...]] = list(
            itertools.product(range(num_mfs), repeat=num_vars)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fuzz = self.fuzzify(x)
        raw_w = self.antecedent(fuzz)
        return self.consequent(raw_w)

    def rule_strengths(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            fuzz = self.fuzzify(x)
            raw_w = self.antecedent(fuzz)
            return raw_w / raw_w.sum(dim=1, keepdim=True).clamp(min=1e-12)

def _train_anfis(
    model: AnfisNet,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    device: torch.device,
) -> None:
    xt = torch.tensor(X, dtype=torch.float, device=device)
    yt = torch.tensor(y, dtype=torch.float, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        F.mse_loss(model(xt), yt).backward()
        opt.step()
    model.eval()

def _gradient_top_ids(
    encoder: nn.Module,
    head: nn.Module,
    x: torch.Tensor,
    ei: torch.Tensor,
    et: torch.Tensor,
    src_id: int,
    task: str,
    y_act: Optional[torch.Tensor],
    y_time: Optional[torch.Tensor],
    pair_idx: int,
    device: torch.device,
    m: int,
    exclude: set,
    use_model_target: bool = False,
) -> List[int]:
    xr = x.detach().clone().requires_grad_(True)
    z = encoder(xr, ei, et)
    out = head(z[src_id: src_id + 1])
    scalar = (
        _scalar_for_model_task(out, task)
        if use_model_target
        else _scalar_for_task(out, task, y_act, y_time, pair_idx, device)
    )
    encoder.zero_grad()
    head.zero_grad()
    scalar.backward()
    g = xr.grad
    if g is None:
        return []
    norms = (g * xr).abs().sum(dim=1).detach().cpu().numpy()
    order = np.argsort(-norms)
    return [int(i) for i in order if int(i) not in exclude][:m]

def _fox_explain(
    encoder: nn.Module,
    head: nn.Module,
    x: torch.Tensor,
    ei: torch.Tensor,
    et: torch.Tensor,
    src_id: int,
    feat_ids: List[int],
    task: str,
    y_act: Optional[torch.Tensor],
    y_time: Optional[torch.Tensor],
    pair_idx: int,
    device: torch.device,
    n_perturb: int,
    num_mfs: int,
    epochs: int,
    lr: float,
    rng: np.random.Generator,
    use_model_target: bool = False,
) -> Tuple[AnfisNet, np.ndarray]:
    m = len(feat_ids)
    scales = rng.uniform(0.0, 1.0, size=(n_perturb, m)).astype(np.float32)
    targets = np.empty(n_perturb, dtype=np.float32)
    x_fix = x.detach()
    encoder.eval()
    head.eval()
    for i, row in enumerate(scales):
        x_pert = x_fix.clone()
        for j, nid in enumerate(feat_ids):
            x_pert[nid] = x_fix[nid] * float(row[j])
        with torch.no_grad():
            z = encoder(x_pert, ei, et)
            out = head(z[src_id: src_id + 1])
            scalar = (
                _scalar_for_model_task(out, task)
                if use_model_target
                else _scalar_for_task(out, task, y_act, y_time, pair_idx, device)
            )
            targets[i] = float(scalar.item())
    anfis = AnfisNet(m, num_mfs).to(device)
    _train_anfis(anfis, scales, targets, epochs, lr, device)
    return anfis, _entity_importance(anfis)

def _entity_importance(model: AnfisNet) -> np.ndarray:
    dev = next(model.parameters()).device
    x_star = torch.ones(1, model.num_vars, device=dev)
    strengths = model.rule_strengths(x_star).squeeze(0).cpu().numpy()
    coeff = model.consequent.coeff.detach().cpu().numpy()
    imp = np.zeros(model.num_vars, dtype=np.float64)
    for k, rule_mf_idx in enumerate(model._mf_idx):
        w = abs(coeff[k]) * float(strengths[k])
        for j, mf_i in enumerate(rule_mf_idx):
            fuzz_val = model.fuzzify.vars[j].mfs[mf_i](
                torch.tensor([1.0], device=dev)
            ).item()
            imp[j] += w * fuzz_val
    total = imp.sum()
    return imp / total if total > 1e-12 else imp

def _extract_rules(
    model: AnfisNet,
    feat_ids: List[int],
    id2ent: Dict[int, str],
    g: Graph,
    top_k: int = 5,
) -> List[dict]:
    mf_labels = ["LOW", "MEDIUM", "HIGH"]
    dev = next(model.parameters()).device
    x_star = torch.ones(1, model.num_vars, device=dev)
    strengths = model.rule_strengths(x_star).squeeze(0).cpu().numpy()
    coeff = model.consequent.coeff.detach().cpu().numpy()
    order = np.argsort(-strengths)
    rules = []
    for rank, k in enumerate(order[:top_k]):
        antecedents = []
        for j, mf_i in enumerate(model._mf_idx[k]):
            nid = feat_ids[j]
            lbl = _entity_short_label(g, id2ent.get(nid, "?"))
            term = mf_labels[mf_i] if mf_i < len(mf_labels) else f"MF{mf_i}"
            antecedents.append({"entity_id": nid, "label": lbl, "term": term})
        rules.append({
            "rank": rank + 1,
            "rule_idx": int(k),
            "strength": float(strengths[k]),
            "consequent": float(coeff[k]),
            "antecedents": antecedents,
        })
    return rules

def _rules_to_text(rules: List[dict]) -> str:
    lines = []
    for r in rules:
        lines.append(f"Rule {r['rule_idx'] + 1} (strength {r['strength']:.3f}):")
        for i, a in enumerate(r["antecedents"]):
            prefix = "  IF " if i == 0 else "  AND "
            lines.append(f"{prefix}#{a['entity_id']} [{a['label']}] is {a['term']}")
        lines.append(f"  THEN contribution = {r['consequent']:.4f}")
    return "\n".join(lines)

def _plot_fox_pair(
    importance: np.ndarray,
    feat_ids: List[int],
    id2ent: Dict[int, str],
    g: Graph,
    rules: List[dict],
    title: str,
    out_path: str,
    pair_side_text: Optional[str] = None,
) -> None:
    labels = [_enriched_label(g, id2ent, nid) for nid in feat_ids]
    order = np.argsort(-importance)
    s = importance[order]
    lab = [labels[i] for i in order]
    rules_txt = _rules_to_text(rules)
    h = max(4.5, len(s) * 0.38 + len(rules) * 0.5)
    right_text = (rules_txt + "\n\n" + pair_side_text) if pair_side_text else rules_txt
    fig = plt.figure(figsize=(13.5, h))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.8, 1.5], wspace=0.13)
    ax = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    ax_r.axis("off")
    ax_r.text(0.0, 1.0, right_text, transform=ax_r.transAxes,
              va="top", ha="left", fontsize=7.0, linespacing=1.25,
              fontfamily="monospace")
    y_pos = np.arange(len(s))
    ax.barh(y_pos, s, color="darkorange", alpha=0.82)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(lab, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("FOX entity importance (normalised ANFIS attribution)")
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def main() -> None:
    ap = argparse.ArgumentParser(description="FOX neuro-fuzzy surrogate explainer for the R-GCN event-pair model.")
    add_common_explainer_args(ap)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--fox-m", type=int, default=5, dest="fox_m",
                    help="Top entities used as ANFIS input variables (default 5; rules = num_mfs^m)")
    ap.add_argument("--fox-num-mfs", type=int, default=3, dest="fox_num_mfs",
                    help="Membership functions per variable (default 3 -> LOW/MEDIUM/HIGH)")
    ap.add_argument("--fox-perturb", type=int, default=200, dest="fox_perturb",
                    help="Perturbation samples for ANFIS surrogate training (default 200)")
    ap.add_argument("--fox-epochs", type=int, default=80, dest="fox_epochs",
                    help="ANFIS Adam training epochs (default 80)")
    ap.add_argument("--fox-lr", type=float, default=0.01, dest="fox_lr",
                    help="ANFIS Adam learning rate (default 0.01)")
    ap.add_argument("--fox-top-rules", type=int, default=5, dest="fox_top_rules",
                    help="Top firing rules saved per pair (default 5)")
    ap.add_argument("--fox-exclude-dst", action="store_true", dest="fox_exclude_dst",
                    help="Exclude destination event node from candidate entities")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx: ExplainerContext = load_explainer_context(args, device)
    encoder, head = ctx.encoder, ctx.head
    x, g, id2ent = ctx.x, ctx.g, ctx.id2ent
    ei_test, et_test = ctx.ei_test, ctx.et_test
    test_pairs = ctx.test_pairs
    y_act_test, y_time_test = ctx.y_act_test, ctx.y_time_test
    act_vocab, id2act = ctx.act_vocab, ctx.id2act
    tasks, n_test, pair_indices = ctx.tasks, ctx.n_test, ctx.pair_indices
    rng = ctx.rng

    os.makedirs(args.out, exist_ok=True)
    fox_dir = os.path.join(args.out, "fox")
    pair_dir = os.path.join(args.out, "per_pair")
    global_dir = os.path.join(args.out, "global")
    for d in (fox_dir, pair_dir, global_dir):
        os.makedirs(d, exist_ok=True)

    global_imp: Dict[str, Dict[int, List[float]]] = {t: defaultdict(list) for t in tasks}
    summary_rows: List[dict] = []

    for pi in tqdm(pair_indices, desc="pairs"):
        src_id, dst_id = test_pairs[pi]
        exclude = {src_id}
        if args.fox_exclude_dst:
            exclude.add(dst_id)

        for task in tasks:
            try:
                feat_ids = _gradient_top_ids(
                    encoder, head, x, ei_test, et_test,
                    src_id, task, y_act_test, y_time_test, pi, device,
                    args.fox_m, exclude,
                )
            except Exception as exc:
                print(f"[warn] gradient selection failed pair={pi} task={task}: {exc}")
                continue

            if len(feat_ids) < 2:
                print(f"[warn] fox skip pair={pi} task={task}: only {len(feat_ids)} candidate(s)")
                continue

            try:
                anfis_model, importance = _fox_explain(
                    encoder, head, x, ei_test, et_test,
                    src_id, feat_ids, task,
                    y_act_test, y_time_test, pi, device,
                    n_perturb=args.fox_perturb,
                    num_mfs=args.fox_num_mfs,
                    epochs=args.fox_epochs,
                    lr=args.fox_lr,
                    rng=rng,
                )
            except Exception as exc:
                print(f"[warn] FOX ANFIS failed pair={pi} task={task}: {exc}")
                continue

            rules = _extract_rules(anfis_model, feat_ids, id2ent, g, top_k=args.fox_top_rules)

            act_name, _, _, _ = get_event_attrs(g, id2ent.get(dst_id, ""))
            act_label = act_name or ""
            if task == "activity" and act_vocab and act_name and act_name in act_vocab:
                act_label = f"{act_name} (id={act_vocab[act_name]})"

            order = np.argsort(-importance)
            rows: List[dict] = []
            for rank, j in enumerate(order[: args.top_k]):
                nid = feat_ids[j]
                uri = id2ent.get(nid, "?")
                row: dict = {
                    "rank": rank + 1,
                    "entity_id": nid,
                    "uri": uri,
                    "label": _entity_short_label(g, uri),
                    "fox_importance": float(importance[j]),
                    "dst_activity": act_label,
                }
                if task == "activity" and y_act_test is not None:
                    ya = int(y_act_test[pi].item())
                    row["y_act_id"] = ya
                    row["y_act_name"] = id2act.get(ya, "")
                rows.append(row)

            csv_name = f"pair{pi:05d}_{task}_fox.csv"
            pd.DataFrame(rows).to_csv(os.path.join(pair_dir, csv_name), index=False)

            rules_path = os.path.join(fox_dir, f"pair{pi:05d}_{task}_rules.txt")
            with open(rules_path, "w", encoding="utf-8") as f:
                f.write(_rules_to_text(rules))

            side_txt = _pair_side_caption(g, id2ent, src_id, dst_id, pi)
            _plot_fox_pair(
                importance=importance,
                feat_ids=feat_ids,
                id2ent=id2ent,
                g=g,
                rules=rules,
                title=f"FOX entity importance (pair {pi}, {task})",
                out_path=os.path.join(fox_dir, f"pair{pi:05d}_{task}.png"),
                pair_side_text=side_txt,
            )

            for j, nid in enumerate(feat_ids):
                global_imp[task][nid].append(float(importance[j]))

            summary_rows.append({
                "pair_idx": pi,
                "task": task,
                "method": "fox",
                "src_id": src_id,
                "dst_id": dst_id,
                "csv": csv_name,
                "fox_m": len(feat_ids),
                "top_rule_strength": float(rules[0]["strength"]) if rules else None,
            })

    for task in tasks:
        if not global_imp[task]:
            continue

        entity_rows: List[dict] = []
        for nid, imp_list in global_imp[task].items():
            uri = id2ent.get(nid, "?")
            arr = np.array(imp_list, dtype=np.float64)
            entity_rows.append({
                "entity_id": nid,
                "uri": uri,
                "label": _entity_short_label(g, uri),
                "category": _entity_category(uri),
                "mean_importance": float(np.mean(arr)),
                "std_importance": float(np.std(arr)),
                "n_pairs": len(imp_list),
            })

        entity_df = pd.DataFrame(entity_rows).sort_values("mean_importance", ascending=False)
        entity_df.to_csv(os.path.join(global_dir, f"fox_top_entities_{task}.csv"), index=False)

        cat_agg = (
            entity_df.groupby("category")["mean_importance"]
            .agg(mean_importance="mean", std_importance="std", count="count")
            .reset_index()
            .sort_values("mean_importance", ascending=False)
        )
        cat_agg.to_csv(os.path.join(global_dir, f"fox_global_{task}.csv"), index=False)

        _plot_global_category(cat_agg, task, os.path.join(fox_dir, f"global_{task}.png"),
                              value_col="mean_importance", xlabel="Mean FOX entity importance",
                              title_template="Global entity-category FOX importance — {task_lbl}")
        print(f"[OK] Global category plot -> {os.path.join(fox_dir, f'global_{task}.png')}")

        top_n = entity_df.head(20)
        top_labels = [_enriched_label(g, id2ent, int(nid)) for nid in top_n["entity_id"].values]
        _plot_top_entities(
            scores=top_n["mean_importance"].values,
            labels=top_labels,
            title=f"Global top entities by mean FOX importance — {task}",
            out_path=os.path.join(fox_dir, f"global_top_entities_{task}.png"),
            top_k=20,
            xlabel="Mean FOX importance (normalised ANFIS attribution over sampled pairs)",
        )

    write_summary_json(os.path.join(args.out, "summary_fox.json"), args, n_test, pair_indices, tasks,
                       extra_fields={
                           "fox_m": args.fox_m, "fox_num_mfs": args.fox_num_mfs,
                           "fox_perturb": args.fox_perturb, "fox_epochs": args.fox_epochs,
                           "fox_lr": args.fox_lr, "fox_exclude_dst": bool(args.fox_exclude_dst),
                       }, rows=summary_rows)

    print(f"\nDone. Results -> {args.out}")


if __name__ == "__main__":
    main()
