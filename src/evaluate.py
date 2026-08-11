
import argparse
import json
from typing import Dict, Optional

import torch

from src.utils.io_helpers import load_pt, load_vocabs, load_id2ent


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def cls_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    mask   = y_true >= 0
    y_true = y_true[mask].to(torch.long)
    y_pred = y_pred[mask].to(torch.long)

    if y_true.numel() == 0:
        return {"n": 0, "n_unk": int((~mask).sum().item()),
                "acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    acc = (y_true == y_pred).float().mean().item()
    ps, rs, f1s = [], [], []
    for c in torch.unique(y_true).tolist():
        c  = int(c)
        tp = ((y_pred == c) & (y_true == c)).sum().item()
        fp = ((y_pred == c) & (y_true != c)).sum().item()
        fn = ((y_pred != c) & (y_true == c)).sum().item()
        p  = _safe_div(tp, tp + fp)
        r  = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * p * r, p + r) if (p + r) else 0.0
        ps.append(p); rs.append(r); f1s.append(f1)

    return {
        "n":         int(y_true.numel()),
        "n_unk":     int((~mask).sum().item()),
        "acc":       round(float(acc), 4),
        "precision": round(float(sum(ps) / len(ps)), 4),
        "recall":    round(float(sum(rs) / len(rs)), 4),
        "f1":        round(float(sum(f1s) / len(f1s)), 4),
    }


def mae_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    mask = torch.isfinite(y_true) & torch.isfinite(y_pred)
    if mask.sum().item() == 0:
        return {"n": 0, "mae": 0.0}
    v = (y_true[mask] - y_pred[mask]).abs().mean().item()
    return {"n": int(mask.sum().item()), "mae": round(float(v), 4)}


# main

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate R-GCN predictions on the validation set."
    )
    ap.add_argument("--best",          required=True,
                    help="best_val.pt saved by train_2.py (validation data only).")
    ap.add_argument("--output",        required=True,
                    help="Path to write the evaluation report (.txt).")
    ap.add_argument("--vocabs",        default=None,
                    help="vocabs.json saved by train_2.py — enables class-name display.")
    ap.add_argument("--entity2id",     default=None,
                    help="entity2id.json — enables URI display in examples.")
    ap.add_argument("--show_examples", type=int, default=0,
                    help="Print N example val pairs (non-UNK activity only).")
    args = ap.parse_args()

    d = load_pt(args.best)

    y_act    = d["y_act"]         
    y_time   = d["y_time"]        
    y_otherC = d["y_otherC"]      
    y_otherN = d["y_otherN"]      

    pred           = d["pred"]
    act_logits     = pred["act_logits"]
    time_pred_log  = pred.get("time_pred_log", pred.get("time_pred_norm"))
    otherC_logits  = pred["otherC_logits"]
    otherN_pred    = pred["otherN_pred"]

    act_pred = act_logits.argmax(dim=-1)

    N_val = y_act.size(0)
    print(f"Evaluating {N_val} validation pairs …")

    act_vocab = None   
    id2act    = None   
    if args.vocabs:
        vocabs   = load_vocabs(args.vocabs)
        act_vocab = vocabs.get("act_vocab", {})
        id2act    = {v: k for k, v in act_vocab.items()}

    id2ent = load_id2ent(args.entity2id) if args.entity2id else None

    lines = [f"Validation set — {N_val} pairs\n"]

    act_m = cls_metrics(y_act, act_pred)
    lines.append("== Activity (next-event classification) ==")
    lines.append(str(act_m))

    lines.append("\n== Timestamp (MAE, normalised log-space) ==")
    lines.append(str(mae_metrics(y_time, time_pred_log)))

    if otherC_logits:
        lines.append("\n== Other categorical attributes ==")
        for key, logits in otherC_logits.items():
            y    = y_otherC.get(key)
            if y is None:
                continue
            yhat = logits.argmax(dim=-1)
            lines.append(f"  {key}: {cls_metrics(y, yhat)}")

    if otherN_pred:
        lines.append("\n== Other numerical attributes ==")
        for key, yhat in otherN_pred.items():
            y = y_otherN.get(key)
            if y is None:
                continue
            lines.append(f"  {key}: {mae_metrics(y, yhat)}")

    if args.show_examples > 0 and "pairs_val" in d:
        pairs_val = d["pairs_val"]   # (N_val, 2) node IDs
        known_mask = (y_act >= 0).nonzero(as_tuple=True)[0]
        show_n = min(args.show_examples, len(known_mask))
        lines.append(f"\n== Examples (first {show_n} non-UNK validation pairs) ==")
        for idx in known_mask[:show_n].tolist():
            src_id = int(pairs_val[idx, 0].item())
            dst_id = int(pairs_val[idx, 1].item())
            src    = id2ent[src_id] if id2ent else str(src_id)
            dst    = id2ent[dst_id] if id2ent else str(dst_id)
            true_act_id = int(y_act[idx].item())
            pred_act_id = int(act_pred[idx].item())
            row = {
                "src":       src,
                "dst":       dst,
                "y_act":     id2act.get(true_act_id, true_act_id) if id2act else true_act_id,
                "p_act":     id2act.get(pred_act_id, pred_act_id) if id2act else pred_act_id,
                "y_time":    round(float(y_time[idx].item()), 4)
                             if torch.isfinite(y_time[idx]) else None,
                "p_time":    round(float(time_pred_log[idx].item()), 4)
                             if torch.isfinite(time_pred_log[idx]) else None,
            }
            lines.append(str(row))

    report = "\n".join(lines)
    print(report)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nReport saved → {args.output}")


if __name__ == "__main__":
    main()
