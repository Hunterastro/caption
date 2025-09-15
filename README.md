A compact pipeline that turns time series into one-sentence natural descriptions, powered by a soft-prefix projector on top of LLaMA with LoRA/QLoRA fine-tuning, plus multi-reference NLL (MR‑NLL) alignment to human-written captions.

Input: a univariate time series per sample (with optional human captions).

Preprocess: z-score → patch into fixed-size chunks → segment (begin/middle/end one‑hot).

Learn: map patches+segments → soft prefix tokens (same dim as LLaMA embeddings), concatenate with a short prompt, and only supervise the target sentence (CE). With multiple human captions, train on the easiest (lowest CE) one (MR‑NLL).

Infer: generate one overall sentence and three short phrases (begin/middle/end), optionally fuse into a fluent summary line; score with BLEU‑1 and ROUGE‑L.

Inference & Evaluation

Configure dataset test paths and output CSV in infer_eval_all.py, then:

python infer_eval_all.py

Outputs per dataset:

A CSV with columns(pilot13/16final_preds.csv):

id, pred, begin, middle, end, fused_llama, best_pred, best_from, ref1, ref2, ref3

Console metrics:

Single (just pred): BLEU‑1 / ROUGE‑L

Best‑of‑4 (choose best among pred, begin, middle, end): BLEU‑1 / ROUGE‑L

fused_llama is a readable summary synthesized from the four candidates + simple series facts; it’s not used for scoring—just for human readability


