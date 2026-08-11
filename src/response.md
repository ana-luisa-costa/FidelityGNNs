Here's the full picture of what the output shows and how it's calculated:

---

## What the `edge_mask` column means

The value (e.g. `0.373`, `0.319`, ...) is a **learned soft importance score** for each edge, in the range `(0, 1)` after sigmoid. Higher = that edge contributed more to the model's prediction for this specific pair and task.

---

## How it's calculated — step by step in the code

**Step 1 — Initialize one trainable scalar per edge** (`_initialize_masks`, PyG internals):
```python
std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
self.edge_mask = Parameter(torch.randn(E, device=device) * std)
```
Every edge in the subgraph starts with a small random real number. This is what will be optimized.

**Step 2 — Register the mask on the model** (`_train`, PyG internals):
```python
set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
```
This hooks the mask into `RGCNConv`'s `propagate()`. During every forward pass, each message is multiplied by `sigmoid(edge_mask[i])` — so a mask near `0` suppresses the edge's message, near `1` keeps it.

**Step 3 — Optimize the mask via gradient descent** (the training loop in `_train`):
```python
for i in range(self.epochs):  # 100 epochs in your run
    h = x * self.node_mask.sigmoid()          # masked node features
    y_hat = model(h, edge_index, **kwargs)    # forward through your RGCN
    loss = prediction_loss(y_hat, target)
          + edge_size_penalty * mean(sigmoid(edge_mask))   # push masks toward 0 (sparsity)
          + edge_entropy_penalty * entropy(sigmoid(edge_mask))  # push masks toward 0 or 1 (sharpness)
    loss.backward()
    optimizer.step()
```
The loss has **three terms**:
- **Prediction fidelity**: the masked model should still predict the same as the unmasked model
- **Size regularization**: penalizes having too many edges with high mask → forces sparsity
- **Entropy regularization**: pushes each mask toward 0 or 1 (binary-like) → forces sharpness

After 100 steps, edges that were necessary to reproduce the prediction end up with high mask values; edges that didn't matter get pushed toward 0.

**Step 4 — Read out and rank in your script**:

```328:330:src/explain_gnnexplainer.py
            edge_mask_np = edge_mask.detach().cpu().numpy()
            ei_sub_np = ei_sub.cpu().numpy()
            et_sub_np = et_sub.cpu().numpy()
            order = np.argsort(-edge_mask_np)
```
The mask values are sorted descending → rank 1 = the single most important edge for that prediction.

---

## Reading your actual output

Looking at the first CSV row:
```
rank=1, edge_mask=0.373, src=Accepted, dst=Accepted|t=..., relation=21
full_pred=0.168, topk_pred=0.610
```

- The model predicted activity class 1 (`Accepted`) with logit `0.168` using the full subgraph
- When you keep only the **top-20 edges** by mask, the prediction jumps to `0.610` — meaning the top-k edges are actually **more decisive** than the full noisy graph
- The most important structural feature was the edge from node `Accepted` (global 3324) to another `Accepted` event via relation type 21

One thing to note: the mask values in your run are all very close to each other (0.30–0.37). This suggests the optimizer hasn't fully converged yet with only 100 epochs — the masks haven't spread out into clearly high vs. clearly low values. Running with `--gnn-epochs 300` or `--gnn-epochs 500` would give sharper, more differentiated importance scores.