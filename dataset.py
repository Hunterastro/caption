# dataset_single.py —— 多文件预处理（最小必需版）
# 仅保留：标准化 → 分块(patches) → 段位one-hot → 注释展平/规范化 → 汇总写出 *_proc.json
# 已删除：① 结构标签/模板化描述（labels6 / templated_caption）② 数据增强（augment）
# 这样可直接服务后续训练/评估：若训练代码不再依赖模板回退，则无影响；评测本就只用人工注释。

import os, json, math, statistics, random, re
import numpy as np

# ===================== 全局配置 =====================
# BASE_DIR: 输入原始数据目录；OUT_DIR: 输出目录（会自动创建）
BASE_DIR  = "/home/hunter/timeseries/caption/LLM_V1/Pre_process/ts_caption_pipeline/processed_data"
OUT_DIR   = os.path.join(os.path.dirname(BASE_DIR), "precomputed")
IN_FILES  = [
    "pilot13finaltrain.json",
    "pilot13finalval.json",
    "pilot13finaltest.json",
    "pilot16btrain.json",
    "pilot16bval.json",
    "pilot16btest.json",
]
PATCH_SIZE = 3   # 分块大小：将序列等长划分为每块 patch_size 个点
SEED       = 42
random.seed(SEED); np.random.seed(SEED)
# ==================================================

# ===================== 数值处理 =====================

def zscore(x):
    """对单条序列做 z-score 标准化。返回 float32 ndarray。
    说明：逐条标准化不依赖全局统计，适配不同长度与量级。"""
    x = np.asarray(x, dtype=np.float32)
    mu, sd = x.mean(), x.std()
    if sd < 1e-6:
        return (x - mu)
    return (x - mu) / (sd + 1e-6)


def make_patches(x, patch=3):
    """把标准化后的序列按长度 patch 分块，末尾用边界值重复进行最小填充。
    返回形状 (K, patch)。"""
    T = len(x)
    K = math.ceil(T/patch)
    pads = K*patch - T
    if pads > 0:
        x = np.pad(x, (0, pads), mode="edge")  # 边界重复补齐
    return x.reshape(K, patch)


def seg_onehot(T):
    """基于相对位置生成段位 one-hot（前/中/后三个区段）。返回 (T, 3)。"""
    thirds = max(1, T // 3)
    seg = np.zeros((T, 3), dtype=np.float32)
    seg[:thirds, 0] = 1
    seg[thirds:2*thirds, 1] = 1
    seg[2*thirds:, 2] = 1
    return seg

# ===================== 文本处理 =====================

def _flatten_annotations(ann):
    """把注释统一成一维字符串列表，兼容 ["a","b"] 与 [["a"],["b"]]。空时返回 [""]。"""
    out = []
    for a in ann or []:
        if isinstance(a, list):
            out.extend([str(x) for x in a])
        else:
            out.append(str(a))
    out = [s.strip() for s in out if isinstance(s, str) and s.strip()]
    return out if out else [""]

# 简单同义词/错拼归一（按需扩充即可）
_FIXES = {
    "incresae":"increase",
    "increse":"increase",
    "steadly":"steadily",
    "constantly":"steadily",
    "spike":"peak",
    "spikes":"peak",
    "troughs":"trough",
    "drops sharply":"drop sharply",
}

def normalize_caption(s: str):
    """注释规范化：小写、去尾标点、错拼/同义词替换、多空格收缩。"""
    s = s.lower().strip()
    s = re.sub(r"[.\u3002!,;]+$", "", s)   # 去末尾标点
    for k,v in _FIXES.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", " ", s)
    return s

# ===================== 其他小工具 =====================

def infer_domain_from_name(filename: str):
    """根据文件名推断域，用于简单分域统计（可按需修改/删除）。"""
    name = os.path.basename(filename).lower()
    if "pilot13" in name: return "pilot13"
    if "pilot16" in name: return "pilot16b"
    return "unknown"

# ===================== 核心：单条样本预处理 =====================

def preprocess_record(ex, patch_size=3, domain=None):
    """把一条原始样本处理成训练/评测就绪的最小字典。

    输入字段要求：
      - ex["series"] 或 ex["Series"]: 序列(list/array)
      - ex["annotations"]: 可选，人工注释

    返回字段：
      {
        id,                      # 样本ID（若缺失由上层填充）
        series,                  # 原始序列（不标准化，用于可解释/可视化）
        annotations,             # 原始注释（保留原样）
        captions_flat,           # 展平后的注释
        captions_norm,           # 规范化注释（供训练/评测使用）
        patches,                 # (K, patch_size) 标准化并分块
        segments,                # (T, 3) 段位 one-hot
        domain                   # 领域标签（可选）
      }
    """
    series = ex.get("series") or ex.get("Series")
    if series is None:
        raise ValueError("样本缺少 'series' 字段")
    series = list(series)

    # 数值特征
    x = zscore(series)
    patches = make_patches(x, patch_size)     # (K, patch)
    seg = seg_onehot(len(series))             # (T, 3)

    # 文本特征
    captions_flat = _flatten_annotations(ex.get("annotations", []))
    captions_norm = [normalize_caption(s) for s in captions_flat]

    # 汇总输出（最小字段集）
    return {
        "id": ex.get("id", None),
        "series": series,
        "annotations": ex.get("annotations", []),
        "captions_flat": captions_flat,
        "captions_norm": captions_norm,
        "patches": patches.astype("float32").tolist(),
        "segments": seg.astype("float32").tolist(),
        "domain": domain or "unknown",
    }

# ===================== 处理单个 JSON 文件 =====================

def preprocess_file(in_json_path, out_json_path, patch_size=3, preview_k=1):
    """读取一个输入 JSON（dict[id]->record），逐条预处理并写出 *_proc.json。"""
    with open(in_json_path, "r") as f:
        data = json.load(f)
    items = list(data.values())
    n = len(items)

    if n == 0:
        print(f"[空文件] {in_json_path}")
        os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
        with open(out_json_path, "w") as fo:
            json.dump({}, fo, ensure_ascii=False)
        return

    out_dict = {}
    lengths = []
    domain = infer_domain_from_name(in_json_path)

    for i, ex in enumerate(items):
        rec = preprocess_record(ex, patch_size=patch_size, domain=domain)
        rid = rec["id"] or f"sample_{i}"
        out_dict[rid] = rec
        lengths.append(len(rec["series"]))

    # 统计信息（便于检查数据分布）
    uniq_lengths = sorted(set(lengths))
    mean_len = statistics.mean(lengths)
    print(f"✅ 处理完成: {in_json_path}")
    print(f"   样本数: {n} | 序列长度均值: {mean_len:.2f} | 唯一长度: {uniq_lengths} | domain={domain}")

    # 预览前若干条（只展示关键形状）
    for j, (rid, rec) in enumerate(list(out_dict.items())[:preview_k]):
        K = len(rec["patches"]) if rec.get("patches") else 0
        P = len(rec["patches"][0]) if K>0 else 0
        T = len(rec["segments"]) if rec.get("segments") else 0
        print(f"   预览[{j}] id={rid} | patches=({K}, {P}) | segments=({T}, 3) | refs={len(rec['captions_flat'])}")

    # 写出新JSON
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w") as fo:
        json.dump(out_dict, fo, ensure_ascii=False)
    print(f"   ↳ 已写出: {out_json_path}\n")

# ===================== 主程序：批量处理 =====================
if __name__ == "__main__":
    print("执行：标准化 → patches → 段位 one-hot → 注释展平/规范化 → 输出 *_proc.json（已去除结构标签/模板/增强）")
    for name in IN_FILES:
        in_path  = os.path.join(BASE_DIR, name)
        base, ext = os.path.splitext(name)
        out_path = os.path.join(OUT_DIR, f"{base}_proc.json")
        preprocess_file(
            in_path, out_path,
            patch_size=PATCH_SIZE,
            preview_k=1,
        )
    print("全部完成 ✅")
