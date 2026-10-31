# -*- coding: utf-8 -*-
"""CRBM 流水线配置。

所有路径均指向本地，禁止任何远程下载。
"""

from pathlib import Path

ROOT = Path(r"D:\ncbm_ds4.1f")

# ---------------- 概念词典 ----------------
CONCEPT_CSV = ROOT / "data" / "concepts" / "cbm_concept_set_v2.csv"

# ---------------- 原始数据 ----------------
DATA_RAW = ROOT / "data" / "raw"
DATASET_FILES = {
    "TOXICN": {
        "train": DATA_RAW / "TOXICN" / "train.json",
        "test":  DATA_RAW / "TOXICN" / "test.json",
        "text_key": "content",
        "label_key": "toxic",
    },
    "COLD": {
        "train": DATA_RAW / "COLD" / "train.json",
        "test":  DATA_RAW / "COLD" / "test.json",
        "text_key": "content",
        "label_key": "toxic",
    },
}

# ---------------- 本地模型 ----------------
MODELS_DIR = ROOT / "models"
LLM_HINT = "glm-4-9b"              # 概念标注用，models/ 下按关键字匹配
ENCODER_HINT = "chinese-roberta"   # M1 概念层编码器（可换 macbert 等）

# ---------------- 标注 ----------------
# 每个状态只给提示词里真正要求的那一个字，不加同义词。
# 理由：ternary 里 "否" 既可能表示"这个概念不存在"（→未提及），也可能表示"否定该说法"（→否认），
# 映射到 deny 会把大量"未提及"错记成"否认"，而"否认"正是隐式样本的核心信号。
# 宁可让它落成 -1（缺失）并在 cover 指标里暴露出来，也不要猜。
VERBALIZERS = {
    "ternary": {
        "assert": ["有"],   # 断言
        "deny":   ["反"],   # 否认
        "absent": ["无"],   # 未提及
    },
    "binary": {
        "present": ["是"],
        "absent":  ["否"],
    },
}
ORDINAL_TOKENS = ["0", "1", "2", "3", "4", "5"]

ANNOTATE = {
    "batch_samples": 200,      # 每批样本数，每批产生 batch_samples × 概念数 条 prompt（只控显存）
    "top_logprobs": 20,        # 前 20 个 token 的 logprob，覆盖候选字（与 fscbm 一致）
    "max_model_len": 2048,     # fscbm 用 1024；本任务 prompt 带判定规则，长约 700 token
    "max_num_seqs": 64,        # 与 fscbm 一致
    "max_num_batched_tokens": 16384,   # 与 fscbm 一致
    "gpu_memory_utilization": 0.85,
    "temperature": 0.0,
    "max_text_chars": 512,     # 超长文本截断，避免超出 max_model_len
    "prompt_suffix": "\n",     # glm-4-9b-chat 的 chat template 结尾补一个换行
}

# ---------------- 产物 ----------------
PROCESSED = ROOT / "data" / "processed"
EXPERIMENTS = ROOT / "experiments"
