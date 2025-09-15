# Time-Series → One-Sentence Descriptions

A compact pipeline that turns **time series** into **one-sentence natural descriptions**, powered by a **soft-prefix projector** on top of **LLaMA** with **LoRA/QLoRA** fine-tuning, plus **multi-reference NLL (MR-NLL)** alignment to human-written captions.

---

## 🔎 Overall Flow

```
📈 Time Series
      │
      ▼
🧹 Preprocess
  (z-score → patches → segments)
      │
      ▼
🧩 TSProjector
  (map features → soft prefix tokens)
      │
      ▼
🤖 LLaMA + LoRA/QLoRA
  (prefix + prompt + captions)
      │
      ├── Training: align to human captions (MR-NLL, CE loss)
      │
      └── Inference: generate
          ├─ Overall sentence
          ├─ Begin / Middle / End phrases
          └─ Fused summary
```

---

## 📌 Pipeline at a Glance

* **Input**
  A univariate time series per sample (with optional human captions).

* **Preprocess**

  ```
  z-score → patch into fixed-size chunks → segment (begin / middle / end one-hot)
  ```

* **Learn**

  * Map patches + segments → soft prefix tokens (same dim as LLaMA embeddings)
  * Concatenate with a short prompt
  * **Only supervise the target sentence (CE)**
  * With multiple human captions, train on the easiest (lowest CE) one (MR-NLL)

* **Infer**

  * Generate one overall sentence and three short phrases (begin / middle / end)
  * Optionally fuse into a fluent summary line
  * Evaluate with **BLEU-1** and **ROUGE-L**

---

## 🚀 Inference & Evaluation

1. Configure dataset test paths and output CSV in **`infer_eval_all.py`**
2. Run:

   ```bash
   python infer_eval_all.py
   ```

### 📂 Outputs per dataset

* A CSV file *(e.g., pilot13/16final\_preds.csv)* with columns:

  ```
  id, pred, begin, middle, end, fused_llama, best_pred, best_from, ref1, ref2, ref3
  ```

* **Console metrics**

  * *Single* (just `pred`): BLEU-1 / ROUGE-L
  * *Best-of-4* (choose best among `pred`, `begin`, `middle`, `end`): BLEU-1 / ROUGE-L

> 💡 **Note**: `fused_llama` is a readable summary synthesized from the four candidates + simple series facts.
> It is **not used for scoring** — only for human readability.

---
