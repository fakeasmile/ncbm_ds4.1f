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
# 每个状态可给多个同义候选，取并集概率。用中文单字而非英文单字母：
# GLM 的中文分词对单汉字几乎必然是单 token，而 "A"/"D"/"N" 有被切碎的风险。
VERBALIZERS = {
    "ternary": {
        "assert": ["有", "是"],   # 断言
        "deny":   ["反", "否"],   # 否认
        "absent": ["无", "未"],   # 未提及
    },
    "binary": {
        "present": ["是", "有"],
        "absent":  ["否", "无"],
    },
}
ORDINAL_TOKENS = ["0", "1", "2", "3", "4", "5"]

ANNOTATE = {
    "chunk_samples": 200,      # 每块样本数，每块产生 chunk_samples × 概念数 条 prompt
    "top_logprobs": 20,        # 前 20 个 token 的 logprob，覆盖候选字
    "max_model_len": 2048,
    "max_num_seqs": 32,
    "max_num_batched_tokens": 32768,
    "gpu_memory_utilization": 0.85,
    "temperature": 0.0,
    "max_text_chars": 512,     # 超长文本截断，避免超出 max_model_len
    "prompt_suffix": "\n",     # glm-4-9b-chat 的 chat template 结尾补一个换行
    "trust_remote_code": True,
}

# ---------------- 产物 ----------------
PROCESSED = ROOT / "data" / "processed"
EXPERIMENTS = ROOT / "experiments"
