# -*- coding: utf-8 -*-
"""CRBM 单入口流水线：概念标注 + 标签质检。

一个入口串行执行，产物落地即断点。中断后重跑同一命令会跳过已完成的分块。

用法：
    python scripts/pipeline.py --dataset TOXICN
    python scripts/pipeline.py --dataset TOXICN --model_path D:/ncbm_ds4.1f/models/glm-4-9b-chat
    python scripts/pipeline.py --dataset TOXICN --stages annotate --limit 200
    python scripts/pipeline.py --dataset TOXICN --stages audit

产物：
    data/processed/TOXICN/annot/part0000_n9600_c200.npz   分块（labels/conf/idx）
    data/processed/TOXICN/annot/labels_train.npz          合并矩阵
    data/processed/TOXICN/annot/meta_train.json           概念顺序、编码映射、模型指纹
    experiments/audit/TOXICN_train_concept_health.csv     概念健康表

后续阶段（train_concept / train_reasoner / evaluate）待 k 确定后并入本文件。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

# 部分环境下 OMP_NUM_THREADS 为空串或非法值，vLLM/torch 初始化会报错
if "OMP_NUM_THREADS" in os.environ:
    _v = os.environ["OMP_NUM_THREADS"].strip()
    if not _v.isdigit() or int(_v) <= 0:
        os.environ.pop("OMP_NUM_THREADS")

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs import pipeline_config as C

# 标签编码：ternary 0=未提及/1=断言/2=否认，binary 0=不出现/1=出现，ordinal 0-5，-1=缺失
TERNARY_INV = {0: "未提及", 1: "断言", 2: "否认"}
BINARY_INV = {0: "不出现", 1: "出现"}

# 候选字的排列顺序 → 标签编码
TERNARY_LABEL_ORDER = [1, 2, 0]   # assert, deny, absent
BINARY_LABEL_ORDER = [1, 0]       # present, absent


# ---------------------------------------------------------------- 基础工具

def resolve_model(hint: str, explicit: str | None = None) -> Path:
    """确定本地模型目录：显式指定优先，否则在 models/ 下按关键字匹配。"""
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"[error] --model_path 指向的目录不存在：{p}")
        return p

    models_dir = Path(C.MODELS_DIR)
    if not models_dir.exists():
        raise SystemExit(f"[error] 模型目录不存在：{models_dir}")

    cands = [p for p in models_dir.iterdir() if p.is_dir() and hint.lower() in p.name.lower()]
    if not cands:
        existing = [p.name for p in models_dir.iterdir() if p.is_dir()]
        raise SystemExit(
            f"[error] 在 {models_dir} 下找不到名字含 '{hint}' 的目录。\n"
            f"        当前已有：{existing or '（空）'}\n"
            f"        用 --model_path 显式指定模型目录。"
        )
    if len(cands) > 1:
        print(f"[warn] 匹配到多个目录，取第一个：{[p.name for p in cands]}")
    return cands[0]


def load_concepts(path: Path) -> list[dict]:
    """读取词典。文件带 UTF-8 BOM，必须用 utf-8-sig。"""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        cid = (r.get("id") or "").strip()
        if not cid:
            continue
        out.append({
            "id": cid,
            "name": r["name"].strip(),
            "claim": r["claim"].strip(),
            "assert_rule": r["assert_rule"].strip(),
            "deny_rule": (r.get("deny_rule") or "").strip(),
            "absent_rule": r["absent_rule"].strip(),
            "domain": r["value_domain"].strip(),
            "in_bottleneck": (r.get("in_bottleneck") or "1").strip() == "1",
        })
    domains = {}
    for c in out:
        domains[c["domain"]] = domains.get(c["domain"], 0) + 1
    print(f"[info] 载入概念 {len(out)} 条：{domains}")
    return out


def load_records(dataset: str, split: str) -> list[dict]:
    spec = C.DATASET_FILES[dataset]
    path = Path(spec[split])
    if not path.exists():
        raise SystemExit(f"[error] 数据文件不存在：{path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    for i, r in enumerate(records):
        r["_idx"] = i
    print(f"[info] {dataset}/{split}: {len(records)} 条")
    return records


# ---------------------------------------------------------------- 提示构造

SYSTEM_PROMPT = (
    "你是中文社交媒体文本的概念标注员。"
    "你只判断指定的一个概念在文本中处于什么状态，不做整体毒性判断，不解释理由。"
    "严格按用户要求只输出指定的单个字或数字。"
)

TERNARY_USER = """【文本】
{text}

【概念】{name}
【含义】{claim}

【状态判定】
- 有（断言）：{assert_rule}
- 反（否认）：{deny_rule}
- 无（未提及）：{absent_rule}

请只回答一个字：有、反 或 无。"""

BINARY_USER = """【文本】
{text}

【概念】{name}
【含义】{claim}

【出现判定】
- 是（出现）：{assert_rule}
- 否（不出现）：{absent_rule}

请只回答一个字：是 或 否。"""

ORDINAL_USER = """【文本】
{text}

【概念】{name}
【含义】{claim}

【分级标准】
{levels}

请只回答一个数字：0、1、2、3、4 或 5。"""


def _ordinal_levels(rule: str) -> str:
    """把一行式的分级规则拆成逐级一行，避免模型在长句里数错档。"""
    s = rule.replace("：", "：\n").replace("；", "；\n")
    lines = [line.strip() for line in s.splitlines() if line.strip()]
    lines = [line for line in lines if not line.startswith("按级判定")]
    return "\n".join(lines)


def build_user_text(concept: dict, text: str) -> str:
    if concept["domain"] == "ternary":
        return TERNARY_USER.format(
            text=text, name=concept["name"], claim=concept["claim"],
            assert_rule=concept["assert_rule"],
            deny_rule=concept["deny_rule"] or "文本明确否定该说法",
            absent_rule=concept["absent_rule"],
        )
    if concept["domain"] == "ordinal":
        return ORDINAL_USER.format(
            text=text, name=concept["name"], claim=concept["claim"],
            levels=_ordinal_levels(concept["assert_rule"]),
        )
    return BINARY_USER.format(
        text=text, name=concept["name"], claim=concept["claim"],
        assert_rule=concept["assert_rule"], absent_rule=concept["absent_rule"],
    )


def state_tokens(concept: dict) -> list[list[str]]:
    """返回每个状态对应的候选字列表，顺序与 *_LABEL_ORDER 对齐。"""
    if concept["domain"] == "ternary":
        return [C.VERBALIZERS["ternary"]["assert"],
                C.VERBALIZERS["ternary"]["deny"],
                C.VERBALIZERS["ternary"]["absent"]]
    if concept["domain"] == "ordinal":
        return [[t] for t in C.ORDINAL_TOKENS]
    return [C.VERBALIZERS["binary"]["present"], C.VERBALIZERS["binary"]["absent"]]


# ---------------------------------------------------------------- 阶段 1：标注

def load_vllm(model_dir: Path):
    from transformers import AutoTokenizer
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=C.ANNOTATE["trust_remote_code"], padding_side="right"
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    llm = LLM(
        model=str(model_dir),
        trust_remote_code=C.ANNOTATE["trust_remote_code"],
        dtype="auto",
        gpu_memory_utilization=C.ANNOTATE["gpu_memory_utilization"],
        enable_prefix_caching=True,
        max_model_len=C.ANNOTATE["max_model_len"],
        max_num_seqs=C.ANNOTATE["max_num_seqs"],
        max_num_batched_tokens=C.ANNOTATE["max_num_batched_tokens"],
    )
    return tok, llm


def stage_annotate(dataset: str, split: str, limit: int | None,
                   model_path: str | None, overwrite: bool):
    from tqdm import tqdm
    from vllm import SamplingParams

    concepts = load_concepts(Path(C.CONCEPT_CSV))
    records = load_records(dataset, split)
    if limit:
        records = records[:limit]

    n_total = len(records)
    chunk = C.ANNOTATE["chunk_samples"]
    n_chunks = (n_total + chunk - 1) // chunk

    out_dir = Path(C.PROCESSED) / dataset / "annot"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = resolve_model(C.LLM_HINT, model_path)
    print(f"[info] 标注模型：{model_dir}")

    tok, llm = load_vllm(model_dir)

    # 候选 token 开跑前解析一次：切分异常要立刻暴露，而不是几十万次调用之后
    cand_ids: list[list[list[int]]] = []
    for c in concepts:
        per_state = []
        for words in state_tokens(c):
            ids = []
            for w in words:
                enc = tok.encode(w, add_special_tokens=False)
                if not enc:
                    raise SystemExit(f"[error] 概念 {c['id']} 的候选 '{w}' 编码为空")
                if len(enc) > 1:
                    print(f"[warn] 概念 {c['id']} 候选 '{w}' 被切成 {len(enc)} 个 token，取首 token")
                ids.append(enc[0])
            per_state.append(sorted(set(ids)))
        cand_ids.append(per_state)

    sp = SamplingParams(
        max_tokens=1,
        temperature=C.ANNOTATE["temperature"],
        logprobs=C.ANNOTATE["top_logprobs"],
    )
    suffix = C.ANNOTATE["prompt_suffix"]
    max_chars = C.ANNOTATE["max_text_chars"]

    t0 = time.time()
    total_prompts = 0

    pbar = tqdm(total=n_total, desc=f"标注 {dataset}/{split}", unit="sample", ncols=100)
    for ci in range(n_chunks):
        part_path = out_dir / f"part{ci:04d}_n{n_total}_c{chunk}.npz"
        batch = records[ci * chunk: (ci + 1) * chunk]

        if part_path.exists() and not overwrite:
            pbar.update(len(batch))
            continue

        prompts, meta = [], []
        for r in batch:
            text = (r.get(C.DATASET_FILES[dataset]["text_key"]) or "").strip()
            if not text:
                continue
            if len(text) > max_chars:
                text = text[:max_chars]
            for k, c in enumerate(concepts):
                msgs = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_text(c, text)},
                ]
                prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                prompts.append(prompt + suffix)
                meta.append((r["_idx"], k))

        if not prompts:
            pbar.update(len(batch))
            continue

        outs = llm.generate(prompts, sp, use_tqdm=False)
        total_prompts += len(prompts)

        if len(outs) != len(meta):
            raise RuntimeError(f"输出数 {len(outs)} 与请求数 {len(meta)} 不一致")

        n = len(batch)
        base = batch[0]["_idx"]
        labels = np.full((n, len(concepts)), -1, dtype=np.int8)
        conf = np.zeros((n, len(concepts)), dtype=np.float16)

        for j, o in enumerate(outs):
            sidx, k = meta[j]
            cand = o.outputs[0].logprobs
            if not cand:
                continue
            first = cand[0] or {}
            probs = []
            for ids in cand_ids[k]:
                probs.append(sum(math.exp(first[tid].logprob) for tid in ids if tid in first))

            s = sum(probs)
            if s <= 0:
                continue                       # 候选字都不在前 top-k，保持 -1
            probs = [p / s for p in probs]
            best = int(np.argmax(probs))
            domain = concepts[k]["domain"]
            labels[sidx - base, k] = (TERNARY_LABEL_ORDER[best] if domain == "ternary"
                                      else BINARY_LABEL_ORDER[best] if domain == "binary"
                                      else best)
            conf[sidx - base, k] = probs[best]

        np.savez_compressed(
            part_path,
            labels=labels,
            conf=conf,
            idx=np.array([r["_idx"] for r in batch], dtype=np.int32),
        )
        pbar.update(len(batch))
        pbar.set_postfix_str(f"{pbar.n}/{n_total} | {time.time() - t0:.0f}s")

    pbar.close()
    print(f"[done] 共提交 {total_prompts} 条 prompt，用时 {(time.time() - t0) / 60:.1f} min")

    merge_parts(out_dir, split, n_total, chunk, concepts, model_dir)


def merge_parts(out_dir: Path, split: str, n_total: int, chunk: int,
                concepts: list[dict], model_dir: Path):
    parts = sorted(out_dir.glob(f"part*_n{n_total}_c{chunk}.npz"))
    if not parts:
        print("[warn] 没有分块可合并")
        return

    idxs, labels, confs = [], [], []
    for p in parts:
        d = np.load(p)
        idxs.append(d["idx"])
        labels.append(d["labels"])
        confs.append(d["conf"])

    idx = np.concatenate(idxs)
    labels = np.concatenate(labels, axis=0)
    conf = np.concatenate(confs, axis=0)
    order = np.argsort(idx, kind="stable")
    idx, labels, conf = idx[order], labels[order], conf[order]

    np.savez_compressed(out_dir / f"labels_{split}.npz", labels=labels, conf=conf, idx=idx)

    meta = {
        "dataset_split": split,
        "n_samples": int(labels.shape[0]),
        "n_concepts": int(labels.shape[1]),
        "concept_ids": [c["id"] for c in concepts],
        "concept_names": [c["name"] for c in concepts],
        "domains": [c["domain"] for c in concepts],
        "coding": {
            "ternary": "0=未提及 1=断言 2=否认",
            "binary": "0=不出现 1=出现",
            "ordinal": "0-5",
            "missing": "-1 表示候选字均未进入 top-k",
        },
        "llm": str(model_dir),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / f"meta_{split}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] 合并为 labels_{split}.npz，形状 {labels.shape}")


# ---------------------------------------------------------------- 阶段 2：质检

def stage_audit(dataset: str, split: str):
    from sklearn.metrics import mutual_info_score

    annot_dir = Path(C.PROCESSED) / dataset / "annot"
    pack = annot_dir / f"labels_{split}.npz"
    if not pack.exists():
        raise SystemExit(f"[error] 找不到 {pack}，先跑 annotate")

    data = np.load(pack)
    labels, conf = data["labels"], data["conf"]

    concepts = load_concepts(Path(C.CONCEPT_CSV))
    records = load_records(dataset, split)
    y = np.array([int(r.get(C.DATASET_FILES[dataset]["label_key"], 0)) for r in records])
    y = y[:labels.shape[0]]

    n, k = labels.shape
    if len(concepts) != k:
        raise SystemExit(f"[error] 词典概念数 {len(concepts)} 与矩阵列数 {k} 不一致，词典是否改过？")

    py = np.bincount(y) / max(len(y), 1)
    h_label = float(-np.sum(py[py > 0] * np.log(py[py > 0])))

    # 统一成 0/1 指示做相关分析：ternary 取"断言"，ordinal 取"强度>=1"
    present = np.zeros((n, k), dtype=np.int8)
    for j, c in enumerate(concepts):
        if c["domain"] == "ternary":
            present[:, j] = (labels[:, j] == 1).astype(np.int8)
        elif c["domain"] == "ordinal":
            present[:, j] = (labels[:, j] >= 1).astype(np.int8)
        else:
            present[:, j] = (labels[:, j] == 1).astype(np.int8)

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.nan_to_num(np.corrcoef(present.T), nan=0.0)

    rows = []
    for j, c in enumerate(concepts):
        col = labels[:, j]
        miss = int((col == -1).sum())
        valid = col[col >= 0]
        rate = float((valid > 0).mean()) if len(valid) else 0.0

        dist = {}
        if len(valid):
            vals, cnts = np.unique(valid, return_counts=True)
            for v, cnt in zip(vals, cnts):
                key = (TERNARY_INV.get(int(v)) if c["domain"] == "ternary"
                       else BINARY_INV.get(int(v)) if c["domain"] == "binary" else int(v))
                dist[str(key)] = round(float(cnt) / len(valid), 4)

        mask = col >= 0
        mi = float(mutual_info_score(y[mask], (col[mask] > 0).astype(int))) if mask.any() else 0.0
        mi_norm = mi / h_label if h_label > 0 else 0.0

        dup = [concepts[t]["id"] for t in np.where(np.abs(corr[j]) > 0.9)[0] if t != j]
        mean_conf = float(conf[mask, j].mean()) if mask.any() else 0.0

        flags = []
        if miss / n > 0.05:
            flags.append("缺失率高")
        if rate < 0.01:
            flags.append("正例<1%")
        if mean_conf < 0.6:
            flags.append("置信低")
        if mi_norm > 0.9:
            flags.append("近标签代理")
        if dup:
            flags.append("高共线")

        rows.append({
            "id": c["id"],
            "name": c["name"],
            "domain": c["domain"],
            "in_bottleneck": int(c["in_bottleneck"]),
            "missing_rate": round(miss / n, 4),
            "positive_rate": round(rate, 4),
            "mean_conf": round(mean_conf, 4),
            "dist": json.dumps(dist, ensure_ascii=False),
            "mi_with_label": round(mi, 4),
            "mi_normalized": round(mi_norm, 4),
            "high_corr_with": ",".join(dup),
            "flags": "|".join(flags),
        })

    out_dir = Path(C.EXPERIMENTS) / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{dataset}_{split}_concept_health.csv"
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    dims = sum(2 if c["domain"] == "ternary" else 1 for c in concepts if c["in_bottleneck"])
    print("\n======== 质检摘要 ========")
    print(f"样本 {n}，概念 {k}，当前瓶颈维度 {dims}")
    for flag in ("正例<1%", "置信低", "缺失率高", "近标签代理", "高共线"):
        hit = [r["id"] for r in rows if flag in r["flags"]]
        print(f"{flag:>8}({len(hit)}): {hit or '无'}")
    print(f"明细：{out_csv}")


# ---------------------------------------------------------------- 入口

def main():
    ap = argparse.ArgumentParser(description="CRBM 流水线：概念标注 + 标签质检")
    ap.add_argument("--dataset", default="TOXICN", choices=list(C.DATASET_FILES.keys()))
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--stages", default="annotate,audit")
    ap.add_argument("--model_path", default=None,
                    help="LLM 模型目录，默认在 models/ 下按关键字匹配")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    ap.add_argument("--overwrite", action="store_true", help="忽略已有分块重跑")
    a = ap.parse_args()

    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in ("annotate", "audit")]
    if unknown:
        raise SystemExit(f"[error] 未知阶段 {unknown}，可选 annotate / audit")

    for s in stages:
        print(f"\n===== stage: {s} ({a.dataset}/{a.split}) =====")
        if s == "annotate":
            stage_annotate(a.dataset, a.split, a.limit, a.model_path, a.overwrite)
        else:
            stage_audit(a.dataset, a.split)


if __name__ == "__main__":
    main()
