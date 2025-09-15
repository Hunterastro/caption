# infer_eval_all.py —— 生成 pred + begin/middle/end + fused_llama（不参与评分），评测 Single 与 Best-of-4
# 版本说明：
#   • 与精简后的数据/训练对齐：不依赖 labels6 / templated_caption / 增强。
#   • 参考文本只来自人工 annotations（经展平/清洗）。
#   • 保持与 train_single.py 相同的软前缀投影与 QLoRA 适配加载。
#   • 修正 main() 中重复实例化模型的小问题。

import os, csv, json, math, re, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList, LogitsProcessorList

# 兼容不同 transformers 版本的 NoBadWordsLogitsProcessor（用于约束 LLM 不输出提示工程残留短语）
try:
    from transformers import NoBadWordsLogitsProcessor
except Exception:
    try:
        from transformers.generation.logits_process import NoBadWordsLogitsProcessor
    except Exception:
        NoBadWordsLogitsProcessor = None

from peft import PeftModel, prepare_model_for_kbit_training
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# ========= 路径（按需调整） =========
BASE_MODEL = "meta-llama/Llama-2-13b-hf"
CKPT_DIR   = "/home/hunter/timeseries/caption/LLM_V1/ckpt_ts_caption"

DATASETS = {
    "pilot13": {
        "test_json": "/home/hunter/timeseries/caption/LLM_V1/Pre_process/ts_caption_pipeline/precomputed/pilot13finaltest_proc.json",
        "pred_csv":  "/home/hunter/timeseries/caption/LLM_V1/pilot13final_preds.csv",
    },
    "pilot16b": {
        "test_json": "/home/hunter/timeseries/caption/LLM_V1/Pre_process/ts_caption_pipeline/precomputed/pilot16btest_proc.json",
        "pred_csv":  "/home/hunter/timeseries/caption/LLM_V1/pilot16bfinal_preds.csv",
    },
}

PATCH_SIZE   = 3
USE_QLORA    = True
SEED         = 42

# ========= 生成配置 =========
# 说明：整体句子用确定性（greedy/低温），分段短语用采样（更具多样性），设定最小/最大生成长度、防重复与 n-gram 限制。
MAX_NEW_TOK   = 32
MIN_NEW_TOK   = 6
NO_REPEAT_N   = 3
REPETITION_P  = 1.12

DECODE_OVERALL = {"temperature": 0.4, "top_p": 0.90, "do_sample": False}
DECODE_BEGIN   = {"temperature": 0.7, "top_p": 0.90, "do_sample": True}
DECODE_MIDDLE  = {"temperature": 0.7, "top_p": 0.90, "do_sample": True}
DECODE_END     = {"temperature": 0.7, "top_p": 0.90, "do_sample": True}

# ========= 数值预处理 =========
# 与训练保持一致：逐样本 z-score、按 PATCH_SIZE 分块、三段 one-hot。
def zscore(x):
    x = np.asarray(x, dtype=np.float32)
    mu, sd = x.mean(), x.std()
    if sd < 1e-6: return x - mu
    return (x - mu) / (sd + 1e-6)


def make_patches(x, patch=3):
    T = len(x); K = math.ceil(T/patch); pads = K*patch - T
    if pads > 0: x = np.pad(x, (0, pads), mode="edge")
    return x.reshape(K, patch).astype("float32")


def seg_onehot(T):
    thirds = max(1, T // 3)
    seg = np.zeros((T, 3), dtype=np.float32)
    seg[:thirds, 0] = 1; seg[thirds:2*thirds, 1] = 1; seg[2*thirds:, 2] = 1
    return seg.astype("float32")

# ========= 软前缀投影（把数值特征映射到与词嵌入同维度，作为“可学习前缀 token”） =========
class TSProjector(nn.Module):
    def __init__(self, patch, d_model, seg_dim=3, max_tokens=512):
        super().__init__()
        self.linear = nn.Linear(patch, d_model)      # 把每个 patch 投到 d_model
        self.seg_embed = nn.Linear(seg_dim, d_model)  # 将段位 one-hot 融合为同维度
        self.pos = nn.Parameter(torch.randn(max_tokens, d_model) * 0.01)  # 可学习位置编码
    def forward(self, patches, segments):
        # patches: (B,K,P)；segments: (B,T,3) → 聚合到 K 后相加并加位置编码，输出 (B,K,d)
        B, K, P = patches.shape; device = patches.device
        T = segments.shape[1]; step = max(1, T // K)
        seg_chunks=[]
        for b in range(B):
            ch=[]
            for k in range(K):
                lo, hi = k*step, min((k+1)*step, T)
                s = segments[b, lo:hi].mean(dim=0) if hi>lo else segments[b, -1]
                ch.append(s)
            seg_chunks.append(torch.stack(ch, dim=0))
        seg_t = torch.stack(seg_chunks, dim=0).to(device)
        x = self.linear(patches) + self.seg_embed(seg_t)
        x = x + self.pos[:K].unsqueeze(0).to(device)
        return x

# ========= 文本清洗与评测工具 =========
# 清理生成句子中的提示工程回声、截断到首句、裁掉重复尾巴等，保证评测公平。
_BAD_PAT = re.compile(r"(?i)\b(no extra (words|sentences|text)|only write.*|output\s*format|phrase\s*: )\b")

def clean_sentence(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(_BAD_PAT, "", s)
    for cut in [".", ";", "\n"]:
        if cut in s:
            s = s.split(cut, 1)[0]
    s = s.strip(" ;,|.:-")
    toks = s.split()[:24]
    out = []
    for w in toks:
        out.append(w)
        if len(out) >= 4 and out[-4:-2] == out[-2:]:
            out = out[:-2]
    s = " ".join(out).strip()
    return s or "overall trend is steady"

# 把嵌套/混合列表的人工注释展开为一维并做简单清洗。
def flatten_refs(ann):
    out=[]
    for a in (ann or []):
        if isinstance(a, list): out.extend([str(x) for x in a])
        else: out.append(str(a))
    return [s.strip() for s in out if s and s.strip()]

SMOOTH = SmoothingFunction().method3
ROUGE = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# 句级 BLEU-1 / ROUGE-L（对多参考取最大值）；返回 (bleu, rouge)
def bleu_rouge(hyp, refs):
    refs = refs or [""]
    bleu  = max(sentence_bleu([r.split()], hyp.split(), weights=(1,0,0,0), smoothing_function=SMOOTH) for r in refs)
    rouge = max(ROUGE.score(r, hyp)["rougeL"].fmeasure for r in refs)
    return bleu, rouge

# 辅助：只写前三条参考到 CSV（便于人工比对）
def first_three_refs(ann):
    refs = flatten_refs(ann)
    r1 = refs[0] if len(refs)>0 else ""
    r2 = refs[1] if len(refs)>1 else ""
    r3 = refs[2] if len(refs)>2 else ""
    return r1, r2, r3

# ========= 推理封装（加载基座+LoRA，构建软前缀，逐步生成） =========
class InferWrapper:
    def __init__(self, base_model, ckpt_dir, use_qlora=True):
        # 1) tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
        # 2) 加载基座（可 4bit）并装配 LoRA 适配器
        bnb = BitsAndBytesConfig(
            load_in_4bit=True if use_qlora else False,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            quantization_config=bnb if use_qlora else None,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        base = prepare_model_for_kbit_training(base)
        self.backbone = PeftModel.from_pretrained(base, ckpt_dir, is_trainable=False)
        # 3) 载入软前缀投影器（与训练时保存的一致）
        d_model = self.backbone.config.hidden_size
        self.projector = TSProjector(patch=PATCH_SIZE, d_model=d_model)
        emb = self.backbone.get_input_embeddings().weight
        dev, dt = emb.device, emb.dtype
        self.projector.load_state_dict(torch.load(os.path.join(ckpt_dir, "ts_projector.pt"), map_location="cpu"))
        self.projector.to(device=dev, dtype=dt)
        self.backbone.eval()

    # 文案提示词（四种视角）
    def prompt_overall(self):
        return "Describe the time series in one short sentence focusing on the overall trend and one salient pattern."
    def prompt_begin(self):
        return "Describe the beginning part of the time series in one short phrase (focus on early trend)."
    def prompt_middle(self):
        return "Describe the middle part of the time series in one short phrase (focus on mid-period behavior)."
    def prompt_end(self):
        return "Describe the ending part of the time series in one short phrase (focus on late trend)."

    @torch.no_grad()
    def _soft_from_series(self, patches_np, segments_np):
        # 把 numpy 输入转 tensor，经 TSProjector 得到 (1,K,d) 的软前缀 embeds
        device = next(self.backbone.parameters()).device
        dt = self.backbone.get_input_embeddings().weight.dtype
        patches  = torch.tensor(patches_np[None, ...], dtype=dt, device=device)
        segments = torch.tensor(segments_np[None, ...], dtype=dt, device=device)
        return self.projector(patches, segments)

    @torch.no_grad()
    def _generate_ids(self, soft_prefix_embeds, prompt_text, temperature, top_p, do_sample,
                      max_new=MAX_NEW_TOK, min_new=MIN_NEW_TOK):
        """自实现单步生成循环：把软前缀与 prompt 的嵌入拼接，再逐 token 生成。
        提供：最小/最大生成长度、重复惩罚、n-gram 禁止与 top-p 采样/greedy。"""
        device = next(self.backbone.parameters()).device
        tok = self.tokenizer(prompt_text, return_tensors="pt").to(device)
        emb_prompt = self.backbone.get_input_embeddings()(tok.input_ids)
        inputs_embeds = torch.cat([soft_prefix_embeds, emb_prompt], dim=1)
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

        gen_ids=[]; eos_id = self.tokenizer.eos_token_id

        def block_ngram(logits, hist, n=NO_REPEAT_N):
            if n<=1 or len(hist)<n-1: return
            tail = hist[-(n-1):]
            for i in range(len(hist)-n+1):
                if hist[i:i+n-1] == tail:
                    ban = hist[i+n-1]; logits[0, ban] = -1e9

        for t in range(max_new):
            out = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attn)
            logits = out.logits[:, -1, :]
            # 强制最短生成：在达到最短长度前屏蔽 eos
            if eos_id is not None and t < min_new:
                logits[0, eos_id] = -1e9
            # 重复抑制 & n-gram 禁止
            if gen_ids:
                uniq, cnt = np.unique(gen_ids, return_counts=True)
                for u, c in zip(uniq.tolist(), cnt.tolist()):
                    logits[0, u] = logits[0, u] / (REPETITION_P ** min(c,3))
            block_ngram(logits, gen_ids, NO_REPEAT_N)

            # 采样或贪心
            if do_sample:
                probs = F.softmax(logits / max(1e-6, temperature), dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                keep = cum <= top_p
                if keep.sum()==0: keep[...,0] = True
                filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
                filtered = filtered / filtered.sum(dim=-1, keepdim=True)
                choice_rel = torch.multinomial(filtered, num_samples=1)
                next_id = sorted_idx.gather(-1, choice_rel)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            # 递推一步
            gen_ids.append(next_id.item())
            next_emb = self.backbone.get_input_embeddings()(next_id)
            inputs_embeds = torch.cat([inputs_embeds, next_emb], dim=1)
            attn = torch.cat([attn, torch.ones_like(attn[:, :1])], dim=1)

            # 早停：达到最短长度后遇到句点/分号/eos 则停止
            if t >= min_new:
                if eos_id is not None and next_id.item()==eos_id:
                    break
                if next_id.item() in self._safe_token_ids(".") + self._safe_token_ids(";"):
                    break
        return gen_ids

    def _safe_token_ids(self, w: str):
        # 兼容大小写/带冒号的变体，抓取其 token id
        ids=[]
        for t in [w, w.capitalize(), w.upper(), f"{w}:", f"{w} :"]:
            toks = self.tokenizer.encode(t, add_special_tokens=False)
            if toks: ids.append(toks[-1])
        return list(set(ids))

    @torch.no_grad()
    def generate_one(self, patches_np, segments_np, prompt_text, decode_cfg):
        soft = self._soft_from_series(patches_np, segments_np)
        ids = self._generate_ids(
            soft, prompt_text,
            temperature=decode_cfg["temperature"],
            top_p=decode_cfg["top_p"],
            do_sample=decode_cfg["do_sample"],
        )
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        text = clean_sentence(text)
        return text or "overall trend is steady"

# ========= LLaMA 参与的一句 fused（不参与评分；用于输出更流畅的“总述句”） =========
# 从原始序列中提取“总体趋势 + 代表性事件”，并把四个候选句作为上下文，让 LLaMA 续写一句“Overall, the series …”。

def _ts_facts_from_series(series):
    x = np.asarray(series, dtype=np.float32); T = len(x)
    facts = {}

    if T == 0:
        facts["overall_trend"] = "flat"
        return facts

    # overall trend（把时间轴标准化到 [-1,1]，对 z-score 后的数值做一阶拟合）
    xs = (np.arange(T) - (T - 1) / 2) / max(1, (T - 1) / 2)
    slope = float(np.polyfit(xs, (x - x.mean()) / (x.std() + 1e-6), 1)[0])
    if   slope > 0.15:  facts["overall_trend"] = "increasing"
    elif slope < -0.15: facts["overall_trend"] = "decreasing"
    else:               facts["overall_trend"] = "flat"

    # 中段窗口：确保 hi > lo
    lo = T // 3
    hi = max(2 * T // 3, lo + 1)
    mid = x[lo:hi]

    if mid.size > 0:
        z = (x - x.mean()) / (x.std() + 1e-6)
        mid_peak   = int(np.argmax(mid)) + lo
        mid_trough = int(np.argmin(mid)) + lo
        if z[mid_peak]   >= 1.0:
            facts["salient_event"] = "mid_peak"
        elif z[mid_trough] <= -1.0:
            facts["salient_event"] = "mid_trough"

    # 兜底：末段趋势 / 波动性
    if "salient_event" not in facts and T >= 6:
        tail = x[-(T // 3):]
        xs2 = np.arange(len(tail))
        sl2 = float(np.polyfit(xs2, (tail - tail.mean()) / (tail.std() + 1e-6), 1)[0])
        if   sl2 > 0.2:  facts["salient_event"] = "late_rise"
        elif sl2 < -0.2: facts["salient_event"] = "late_drop"

    if "salient_event" not in facts:
        diff_std = float(np.std(np.diff(x))) if T > 1 else 0.0
        if   diff_std >= 0.5 * (np.std(x) + 1e-6): facts["salient_event"] = "volatility"
        elif diff_std <= 0.1 * (np.std(x) + 1e-6): facts["salient_event"] = "plateau"

    return facts

_TS_LEXICON = ("monotonic increase/decrease; steadily rises/falls; plateaus; "
               "mid-period peak/trough (spike/dip); volatility; oscillation; "
               "early surge; late decline; abrupt drop; gradual uptick; "
               "U-shaped; inverted U; step change; level shift")

def _build_fuse_prompt(facts, cand_all):
    cand_txt = "\n".join([f"{i+1}) {c}" for i,c in enumerate(cand_all)])
    return (
        "You are a time-series analyst.\n"
        "<facts>\n"
        f"overall_trend: {facts.get('overall_trend','flat')}\n"
        f"salient_event: {facts.get('salient_event','')}\n"
        f"style_lexicon: {_TS_LEXICON}\n"
        "</facts>\n\n"
        "<candidates>\n"
        f"{cand_txt}\n"
        "</candidates>\n\n"
        "Write ONE fluent sentence that starts exactly with 'Overall, the series ' "
        "and summarizes the trend and one key event. No extra commentary.\n\n"
        "<final>\n"
        "Overall, the series "
    )

class StopOnStrings(StoppingCriteria):
    def __init__(self, tokenizer, stop_strs):
        self.tokenizer = tokenizer; self.stop_strs = stop_strs
    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return any(s in text for s in self.stop_strs)

_BAD_ECHO2 = re.compile(r"(?i)\b(no extra (words|sentences|text)|only write.*|output\s*format)\b")

def _extract_final_line(full_text):
    seg = full_text.split("<final>", 1)
    txt = seg[1] if len(seg)>1 else full_text
    txt = txt.split("</final>", 1)[0]
    for cut in [".", "。", ";", "；", "\n"]:
        if cut in txt:
            txt = txt.split(cut, 1)[0]
            break
    txt = _BAD_ECHO2.sub("", txt).strip()
    if not txt.startswith("Overall, the series "):
        i = txt.find("Overall, the series ")
        txt = txt[i:] if i>=0 else "Overall, the series " + txt
    txt = txt.strip(" ;,|.:-")
    if txt and not txt.endswith("."): txt += "."
    return txt

@torch.no_grad()
def fuse_with_llama_sentence(backbone, tokenizer, series, cand_all):
    # 构造 facts + candidates 的提示词，让 LLaMA 生成一句更流畅的总述（不计分，仅用于可读性）。
    device = next(backbone.parameters()).device
    facts = _ts_facts_from_series(series)
    prompt = _build_fuse_prompt(facts, cand_all)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    stop = StoppingCriteriaList([StopOnStrings(tokenizer, ["</final>"])])

    # 可选地禁止生成一些提示工程残留词组
    bad_phrases = ["No extra words", "no extra words", "Only write", "only write", "Output format", "output format"]
    bad_words_ids = []
    for w in bad_phrases:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids: bad_words_ids.append(ids)
    logits_proc = LogitsProcessorList()
    if NoBadWordsLogitsProcessor is not None and bad_words_ids:
        logits_proc.append(NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id=tokenizer.eos_token_id))

    out_ids = backbone.generate(
        **inputs,
        max_new_tokens=32,
        min_new_tokens=6,
        do_sample=False,
        temperature=0.3,
        top_p=0.85,
        no_repeat_ngram_size=3,
        repetition_penalty=1.12,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria=stop,
        logits_processor=logits_proc if len(logits_proc)>0 else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return _extract_final_line(text)

# ========= I/O 与特征装载 =========
def load_json(path):
    return json.load(open(path, "r"))


def load_feats_and_series(ex):
    """读取一条样本的数值特征：优先用预处理好的 patches/segments；缺失则由 series 在线计算。"""
    if "patches" in ex and "segments" in ex:
        patches = np.array(ex["patches"], dtype=np.float32)
        segments = np.array(ex["segments"], dtype=np.float32)
    else:
        series = list(ex.get("series") or ex.get("Series"))
        x = zscore(series); patches = make_patches(x, PATCH_SIZE); segments = seg_onehot(len(series))
    series = list(ex.get("series") or ex.get("Series"))
    return patches, segments, series

# ========= 主流程：逐数据域推理 + 评分 =========
def run_domain(model: InferWrapper, name: str, test_json: str, out_csv: str):
    data = load_json(test_json)

    n_single=0; bleu_single=0.0; rouge_single=0.0
    n_best4=0;  bleu_best4=0.0;  rouge_best4=0.0

    rows=[]
    for ex in data.values():
        rid = ex.get("id")
        patches, segments, series = load_feats_and_series(ex)
        refs_all = flatten_refs(ex.get("annotations", []))  # 只用人工参考
        r1, r2, r3 = first_three_refs(ex.get("annotations", []))

        # 四路生成：整体一句 + 三段短语
        pred   = model.generate_one(patches, segments, model.prompt_overall(), DECODE_OVERALL)
        begin  = model.generate_one(patches, segments, model.prompt_begin(),   DECODE_BEGIN)
        middle = model.generate_one(patches, segments, model.prompt_middle(),  DECODE_MIDDLE)
        end    = model.generate_one(patches, segments, model.prompt_end(),     DECODE_END)

        # 评测：Single（pred）
        if refs_all:
            b1, r1s = bleu_rouge(pred, refs_all)
            bleu_single += b1; rouge_single += r1s; n_single += 1

        # 评测：Best-of-4（在四个候选里取分数最高者，用 0.5×BLEU + 0.5×ROUGE 做选择）
        best_idx=0; best_s=-1.0; best_pair=(0.0,0.0)
        for i, c in enumerate([pred, begin, middle, end]):
            b, r = bleu_rouge(c, refs_all)
            s = 0.5*b + 0.5*r
            if s > best_s:
                best_s = s; best_idx=i; best_pair=(b,r)
        bleu_best4 += best_pair[0]; rouge_best4 += best_pair[1]; n_best4 += 1
        best_pred = [pred,begin,middle,end][best_idx]
        src_label = ["pred","begin","middle","end"][best_idx]

        # 生成更流畅的一句“总述”（不参与评分，仅写入 CSV 以供人工阅读）
        fused_llama = fuse_with_llama_sentence(model.backbone, model.tokenizer, series, [pred, begin, middle, end])

        rows.append((rid, pred, begin, middle, end, fused_llama, best_pred, src_label, r1, r2, r3))

    # 写出结果 CSV（便于人工复查）
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","pred","begin","middle","end","fused_llama","best_pred","best_from","ref1","ref2","ref3"]) 
        w.writerows(rows)
    print(f"[{name}] 预测已保存：{out_csv}")

    # 控制台打印汇总指标（按样本平均）
    if n_single>0:
        print(f"[{name}] Single (pred) over {n_single}:     BLEU-1 {bleu_single/n_single:.4f} | ROUGE-L {rouge_single/n_single:.4f}")
    if n_best4>0:
        print(f"[{name}] Best-of-4 over {n_best4}:         BLEU-1 {bleu_best4/n_best4:.4f} | ROUGE-L {rouge_best4/n_best4:.4f}")

    return {"single":{"n":n_single,"bleu":bleu_single,"rouge":rouge_single},
            "best4":{"n":n_best4,"bleu":bleu_best4,"rouge":rouge_best4}}

# ========= 入口 =========
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    model = InferWrapper(BASE_MODEL, CKPT_DIR, use_qlora=USE_QLORA)

    agg = {"single":{"n":0,"bleu":0.0,"rouge":0.0},
           "best4":{"n":0,"bleu":0.0,"rouge":0.0}}

    for name, cfg in DATASETS.items():
        res = run_domain(model, name, cfg["test_json"], cfg["pred_csv"])
        for k in agg:
            agg[k]["n"]    += res[k]["n"]
            agg[k]["bleu"] += res[k]["bleu"]
            agg[k]["rouge"]+= res[k]["rouge"]

    print("\n=== Weighted Overall (by evaluated sample count) ===")
    for k, label in [("single","Single"), ("best4","Best-of-4")]:
        n = agg[k]["n"]
        if n>0:
            print(f"{label:<10}: BLEU-1 {agg[k]['bleu']/n:.4f} | ROUGE-L {agg[k]['rouge']/n:.4f} (N={n})")

if __name__ == "__main__":
    main()
