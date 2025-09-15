# train_single.py —— 软前缀 + QLoRA + 多参考最大似然(MR-NLL)
# 版本说明：已去除对“结构标签/模板化描述/数据增强”的任何依赖。
#   • 训练目标仅来自人工标注（captions_norm；若无则回退到扁平化的 annotations），不再使用 templated_caption。
#   • 数据集中若某条样本没有任何非空的人类参考，将被过滤掉（不进入训练/验证），避免“空串监督”。
#   • 评估同样采用 MR-CE（对多参考取最小交叉熵）。

import os, json, math, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ================= 路径与超参（直接改这里） =================
TRAIN_JSONS = [
    "/home/hunter/timeseries/caption/LLM_V1/Pre_process/ts_caption_pipeline/precomputed/pilot13finaltrain_proc.json",
    "/home/hunter/timeseries/caption/LLM_V1/Pre_process/ts_caption_pipeline/precomputed/pilot16btrain_proc.json",
]
VAL_JSONS = [
    "/home/hunter/timeseries/caption/LLM_V1/Pre_process/ts_caption_pipeline/precomputed/pilot13finalval_proc.json",
    "/home/hunter/timeseries/caption/LLM_V1/Pre_process/ts_caption_pipeline/precomputed/pilot16bval_proc.json",
]

CKPT_DIR   = "/home/hunter/timeseries/caption/LLM_V1/ckpt_ts_caption"
BASE_MODEL = "meta-llama/Llama-2-13b-hf"  # 需 HF 权限
USE_QLORA  = True           # 开启 4bit 量化 + LoRA 的 QLoRA 训练
LORA_R     = 16             # LoRA 秩
PATCH_SIZE = 3              # 与预处理保持一致
SEED       = 42

# 训练策略
EPOCHS      = 3
LR          = 5e-5
BATCH_SIZE  = 2
GRAD_ACCUM  = 1             # 显存紧张可设 4/8
MAX_NORM    = 1.0
WARMUP_FRAC = 0.05

# —— 人工风格对齐（多参考最大似然）——
STYLE_ALIGN = True          # 开启 MR-NLL
MR_MAX_REFS = 3             # 每样本最多取几条人工参考（>1 即可；越大越慢）
APPEND_EOS  = True          # 目标末尾追加 eos，有助稳定收敛
# ==========================================================

# ---------------- 工具函数（数值/文本） ----------------
def zscore(x):
    """对单条序列做 z-score 标准化。
    目的：消除量纲差异，突出形状/趋势，提升训练稳定性。
    说明：逐样本统计；当方差极小（近常量）时仅去均值以避免数值不稳定。"""
    x = np.asarray(x, dtype=np.float32)
    mu, sd = x.mean(), x.std()
    if sd < 1e-6: return x - mu
    return (x - mu) / (sd + 1e-6)


def make_patches(x, patch=3):
    """按固定 patch 大小把标准化后的序列分块。
    目的：把任意长度序列映射为 (K, patch) 的 token-like 片段序列；末尾使用边界值重复最小填充。"""
    T = len(x)
    K = math.ceil(T/patch)
    pads = K*patch - T
    if pads > 0:
        x = np.pad(x, (0, pads), mode="edge")
    return x.reshape(K, patch)


def seg_onehot(T):
    """生成 (T,3) 的相对位置 one-hot（前/中/后三段）。
    目的：把粗粒度时间结构显式注入模型，便于学习“begin/middle/end”类语义。"""
    thirds = max(1, T // 3)
    seg = np.zeros((T, 3), dtype=np.float32)
    seg[:thirds, 0] = 1
    seg[thirds:2*thirds, 1] = 1
    seg[2*thirds:, 2] = 1
    return seg


def _flatten_annotations(ann):
    """把注释统一成一维字符串列表，去空白项。空则返回 []（**本版本不再返回 [''] 占位**）。
    目的：为“多参考监督/评测”提供统一的文本列表接口，避免训练阶段出现空串监督。"""
    out = []
    for a in (ann or []):
        if isinstance(a, list):
            out.extend([str(x) for x in a])
        else:
            out.append(str(a))
    out = [s.strip() for s in out if isinstance(s, str) and s and s.strip()]
    return out  # 若无有效注释，返回 []

# ---------------- 数据集（多文件合并；无模板回退） ----------------
class AllInOneDataset(Dataset):
    """从多个 *_proc.json 汇总构造训练/验证集（**仅使用人工参考**）。

    字段来源优先级：
      • 数值：优先使用 json 中的 patches/segments；缺失则从 series 在线计算。
      • 文本：优先 captions_norm；若无则用扁平化/清洗过的 annotations。
      • 若最终无任何非空文本参考，**丢弃该样本**（避免空目标）。
    """
    def __init__(self, json_paths, patch=PATCH_SIZE, seed=SEED):
        super().__init__()
        self.examples = []
        dropped = 0
        rng = random.Random(seed)
        for p in json_paths:
            with open(p, "r") as f:
                data = json.load(f)
            items = list(data.values())
            for i, ex in enumerate(items):
                # ------- 数值特征 -------
                if ("patches" in ex) and ("segments" in ex):
                    patches = np.array(ex["patches"], dtype=np.float32)
                    segments = np.array(ex["segments"], dtype=np.float32)
                    series = list(ex.get("series") or ex.get("Series") or [])
                else:
                    series = list(ex.get("series") or ex.get("Series") or [])
                    if len(series) == 0:
                        dropped += 1
                        continue
                    x = zscore(series)
                    patches = make_patches(x, patch).astype("float32")
                    segments = seg_onehot(len(series)).astype("float32")

                # ------- 文本参考（仅人类） -------
                refs = ex.get("captions_norm")
                if not refs:
                    refs = _flatten_annotations(ex.get("annotations", []))
                # 统一小写/去首尾空格；剔除空串
                refs = [str(c).strip().lower() for c in (refs or []) if str(c).strip()]
                if len(refs) == 0:
                    dropped += 1  # 没有人类参考，直接丢弃该样本
                    continue

                rid = ex.get("id", f"{os.path.basename(p)}_{i}")
                self.examples.append({
                    "id": rid,
                    "series": series,
                    "patches": patches,
                    "segments": segments,
                    "captions_norm": refs,
                })
        rng.shuffle(self.examples)
        self.patch = patch
        print(f"[dataset] loaded={len(self.examples)} | dropped(no-human-ref or no-series)={dropped}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn_passthrough(batch):
    """按样本列表直接返回，便于逐样本做 MR-NLL。"""
    return batch

# ---------------- 模型：软前缀 + LoRA LLaMA ----------------
class TSProjector(nn.Module):
    """把 (B,K,patch) 的片段序列与 (B,T,3) 的三段 one-hot 融合，映射到与 LLM 词嵌入同维度，作为**软前缀 token**。"""
    def __init__(self, patch, d_model, seg_dim=3, max_tokens=512):
        super().__init__()
        self.linear = nn.Linear(patch, d_model)
        self.seg_embed = nn.Linear(seg_dim, d_model)
        self.pos = nn.Parameter(torch.randn(max_tokens, d_model) * 0.01)  # 可学习的位置编码（用于软前缀 token 序列）

    def forward(self, patches, segments):
        # patches: (B,K,P)；segments: (B,T,3) → 对齐到 K 后融合，得到 (B,K,d)
        B, K, P = patches.shape
        device = patches.device
        T = segments.shape[1]
        # 将 T 个时间步的段位 one-hot 平均聚合到 K 个 patch 上（简单、稳定）
        step = max(1, T // K)
        seg_chunks = []
        for b in range(B):
            ch = []
            for k in range(K):
                lo, hi = k*step, min((k+1)*step, T)
                s = segments[b, lo:hi].mean(dim=0) if hi > lo else segments[b, -1]
                ch.append(s)
            seg_chunks.append(torch.stack(ch, dim=0))
        seg_t = torch.stack(seg_chunks, dim=0).to(device)

        x = self.linear(patches) + self.seg_embed(seg_t)
        x = x + self.pos[:K].unsqueeze(0).to(device)  # 加上前 K 个可学习位置
        return x


class TSLLMWrapper(nn.Module):
    """封装：Tokenizer + LLaMA(base, 冻结) + LoRA(全层目标模块) + TSProjector(软前缀)。"""
    def __init__(self, base_model, lora_r=16, use_qlora=True):
        super().__init__()
        # 1) 分词器（与基座模型匹配）
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)

        # 2) 加载基座模型（可选 QLoRA 量化以节省显存），并做 k-bit 训练准备
        bnb = BitsAndBytesConfig(
            load_in_4bit=True if use_qlora else False,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        self.backbone = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            quantization_config=bnb if use_qlora else None,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        self.backbone = prepare_model_for_kbit_training(self.backbone)

        # 3) 为注意力/MLP 的关键投影层插入 LoRA 适配器（整网插入，但只训练 LoRA 参数）
        peft_cfg = LoraConfig(
            r=lora_r, lora_alpha=16, lora_dropout=0.05, bias="none",
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","down_proj","up_proj"],
            task_type="CAUSAL_LM"
        )
        self.backbone = get_peft_model(self.backbone, peft_cfg)

        # 4) 软前缀投影器：把时间序列表征映射到与词嵌入同维度，并拼接到文本前
        d_model = self.backbone.config.hidden_size
        self.projector = TSProjector(patch=PATCH_SIZE, d_model=d_model)

        # 5) 设备/精度对齐
        embed_w = self.backbone.get_input_embeddings().weight
        dev, dt = embed_w.device, embed_w.dtype
        self.projector.to(device=dev, dtype=dt)

    def build_prompt(self):
        """生成通用提示词（英文，鼓励自然、客观的一句描述）。"""
        return (
            "You are a data analyst. Write one short natural sentence that accurately "
            "describes the trend and key pattern of the time series (beginning / middle / end)."
        )

    def tokenize_with_softprefix(self, patches, segments, target_text):
        """把软前缀与 (prompt + target_text) 的 token 嵌入拼接起来，并构建监督标签。
        监督策略：仅对 target_text 部分打标签（其余为 -100 不参与损失）。"""
        device = next(self.backbone.parameters()).device
        dt = self.backbone.get_input_embeddings().weight.dtype

        # 软前缀 (B,K,d)
        patches  = patches.to(device=device, dtype=dt)
        segments = segments.to(device=device, dtype=dt)
        soft = self.projector(patches, segments)
        K = soft.shape[1]

        # 文本 tokens
        prompt = self.build_prompt()
        tok = self.tokenizer(prompt, return_tensors="pt").to(device)
        P = tok.input_ids.shape[1]

        if APPEND_EOS and (self.tokenizer.eos_token is not None) and (not target_text.endswith(self.tokenizer.eos_token)):
            target_text = target_text + self.tokenizer.eos_token
        tgt = self.tokenizer(target_text, return_tensors="pt", add_special_tokens=False).to(device)
        T = tgt.input_ids.shape[1]

        full_ids  = torch.cat([tok.input_ids, tgt.input_ids], dim=1)     # (B, P+T)
        emb_full  = self.backbone.get_input_embeddings()(full_ids)        # (B, P+T, d)
        inputs_embeds = torch.cat([soft, emb_full], dim=1)                # (B, K+P+T, d)

        attn = torch.ones(inputs_embeds.shape[:-1], dtype=torch.long, device=device)

        labels = torch.full(
            (full_ids.shape[0], K + P + T),
            fill_value=-100,
            dtype=full_ids.dtype,
            device=device,
        )
        labels[:, K + P:] = tgt.input_ids  # 只监督目标文本
        return inputs_embeds, attn, labels

    def forward_ce(self, inputs_embeds, attention_mask, labels):
        """前向：使用 HF 的 CausalLM 接口计算交叉熵损失。"""
        return self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

# ---------------- 多参考最大似然（MR-NLL） ----------------
def get_human_candidates(ex, k=MR_MAX_REFS):
    """返回用于风格对齐的候选目标文本列表（**仅人工参考**）。
    - 首选 ex['captions_norm']（已清洗过的人类标注，多参考）
    - 若不存在则返回 []（本版本不再回退模板/空串）
    - 上限 k 条，随机打乱后截断。
    """
    cands = list(ex.get("captions_norm") or [])
    cands = [s for s in cands if isinstance(s, str) and s.strip()]
    random.shuffle(cands)
    return cands[:max(1, k)]


@torch.no_grad()
def choose_best_ref_by_ce(model, patches, segments, candidates):
    """用 no_grad 逐个候选前向，取 CE 最小的文本；若 candidates 为空，返回 (None, +inf)。"""
    if len(candidates) == 0:
        return None, float("inf")
    device = next(model.backbone.parameters()).device
    best_text, best_loss = None, float("inf")
    for t in candidates:
        in_emb, attn, labels = model.tokenize_with_softprefix(patches, segments, t)
        out = model.forward_ce(in_emb, attn, labels)
        ce = float(out.loss.item())
        if ce < best_loss:
            best_loss = ce
            best_text = t
    return best_text, best_loss

# ---------------- 训练/验证 ----------------
def to_tensor(x, device, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype, device=device)


def train_one_epoch(model, loader, opt, sch, step_g, total_steps):
    """单轮训练：逐样本执行 MR-NLL（或普通 CE）。支持梯度累积与梯度裁剪。"""
    device = next(model.backbone.parameters()).device
    model.train()
    running = {"loss":0.0, "num":0}

    for step, batch in enumerate(loader):
        total_loss = 0.0
        opt.zero_grad(set_to_none=True)

        for b, ex in enumerate(batch):
            patches  = to_tensor(np.expand_dims(ex["patches"], 0), device, torch.float32)
            segments = to_tensor(np.expand_dims(ex["segments"],0), device, torch.float32)

            if STYLE_ALIGN:
                # 1) no_grad 选择当前最“贴合模型”的参考文本（MR-NLL 的 min 近似）
                cands = get_human_candidates(ex, k=MR_MAX_REFS)
                best_text, _ = choose_best_ref_by_ce(model, patches, segments, cands)
                if best_text is None:
                    continue  # 理论上不会发生：数据集已过滤无参考样本
                # 2) 用最佳参考再次前向（带图）来反传
                in_emb, attn, labels = model.tokenize_with_softprefix(patches, segments, best_text)
                out = model.forward_ce(in_emb, attn, labels)
                ce = out.loss
            else:
                # 退化为单一参考：随机取 1 条人工文本
                cands = get_human_candidates(ex, k=1)
                if len(cands) == 0:
                    continue
                in_emb, attn, labels = model.tokenize_with_softprefix(patches, segments, cands[0])
                out = model.forward_ce(in_emb, attn, labels)
                ce = out.loss

            (ce/GRAD_ACCUM).backward()
            total_loss += float(ce.item())
            running["num"] += 1

            # 梯度累积：每 GRAD_ACCUM 个样本/批次进行一次优化器更新
            if ((b+1) % GRAD_ACCUM == 0) or (b == len(batch)-1):
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True); step_g += 1

        if step % 50 == 0:
            bs = max(1, running["num"])  # 避免 0 除
            print(f"[train] step {step:>4} | loss={total_loss/bs:.4f}")
        running["loss"] += total_loss

    return step_g, running


@torch.no_grad()
def evaluate(model, loader):
    """验证：对每个样本做 MR-CE（对多参考取最小 CE），返回平均 CE。"""
    device = next(model.backbone.parameters()).device
    model.eval()
    sums = {"ce":0.0, "n":0}
    for batch in loader:
        for ex in batch:
            patches  = to_tensor(np.expand_dims(ex["patches"], 0), device, torch.float32)
            segments = to_tensor(np.expand_dims(ex["segments"],0), device, torch.float32)
            cands = get_human_candidates(ex, k=MR_MAX_REFS)
            best_text, _ = choose_best_ref_by_ce(model, patches, segments, cands)
            if best_text is None:
                continue
            in_emb, attn, labels = model.tokenize_with_softprefix(patches, segments, best_text)
            out = model.forward_ce(in_emb, attn, labels)
            ce = float(out.loss.item())
            sums["ce"] += ce; sums["n"] += 1
    if sums["n"]==0: return {"ce":None}
    return {"ce":sums["ce"]/sums["n"]}

# ---------------- 主程序 ----------------
def main():
    # 随机种子与性能设置
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(CKPT_DIR, exist_ok=True)

    # 加载数据（本版本已过滤掉无人工参考的样本）
    train_ds = AllInOneDataset(TRAIN_JSONS, patch=PATCH_SIZE, seed=SEED)
    val_ds   = AllInOneDataset(VAL_JSONS,   patch=PATCH_SIZE, seed=SEED) if len(VAL_JSONS)>0 else None
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn_passthrough)
    val_dl   = DataLoader(val_ds,   batch_size=1,           shuffle=False, collate_fn=collate_fn_passthrough) if val_ds else None

    print(f"Total train samples: {len(train_ds)} | Total val samples: {len(val_ds) if val_ds else 0}")

    # 构建模型 + 优化器 + 训练调度
    model = TSLLMWrapper(BASE_MODEL, lora_r=LORA_R, use_qlora=USE_QLORA)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    total_steps = max(1, EPOCHS * len(train_dl))
    sch = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=max(1, int(WARMUP_FRAC * total_steps)),
        num_training_steps=total_steps,
    )

    # 训练循环
    step_g = 0
    for ep in range(EPOCHS):
        print(f"\n===== Epoch {ep+1} / {EPOCHS} =====")
        step_g, train_log = train_one_epoch(model, train_dl, opt, sch, step_g, total_steps)
        if val_dl:
            val_res = evaluate(model, val_dl)
            if val_res["ce"] is not None:
                print(f"[val ] MR-CE={val_res['ce']:.4f}")
            else:
                print("[val ] no valid samples (no human refs)")

    # 保存 LoRA 适配器 + 软前缀投影
    model.backbone.save_pretrained(CKPT_DIR)
    torch.save(model.projector.state_dict(), os.path.join(CKPT_DIR, "ts_projector.pt"))
    print(f"已保存到 {CKPT_DIR}")


if __name__ == "__main__":
    main()
