# -*- coding: utf-8 -*-
"""CRBM 单入口流水线：概念标注 + 标签质检。

一个入口串行执行。**不做断点续跑**：中途异常就整条重跑，所以先用 --limit 跑少量样本确认链路。
分批只用于控制显存与 prompt 构造的内存，不落任何中间文件。

用法：
    python scripts/pipeline.py --dataset TOXICN --limit 500      # 冒烟：前 500 条（跨批，>batch_samples）
    python scripts/pipeline.py --dataset TOXICN                   # 全量
    python scripts/pipeline.py --dataset TOXICN --stages audit
    python scripts/pipeline.py --dataset TOXICN --model_path D:/ncbm_ds4.1f/models/glm-4-9b-chat

产物：
    data/processed/TOXICN/annot/concept_labels_train.json
    experiments/audit/TOXICN_train_concept_health.csv

json 结构：
    {"meta": {...}, "concepts": [...], "samples": [{idx, text, label, labels[], conf[]}]}

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
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"[error] 词典不存在：{path}\n        项目根目录：{C.ROOT}")
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
        raise SystemExit(f"[error] 数据文件不存在：{path}\n        项目根目录：{C.ROOT}")
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


def prep_text(record: dict, dataset: str, max_chars: int) -> str:
    """取文本并按打标时的规则截断；打标与落盘必须用同一份文本，否则 json 无法复现。

    必须判类型：COLD/train 里有 1 条 content 是 NaN（float），
    `(nan or "").strip()` 不会走 `or` 分支（NaN 是 truthy），会直接抛 AttributeError。
    """
    t = record.get(C.DATASET_FILES[dataset]["text_key"])
    if not isinstance(t, str):
        return ""
    t = t.strip()
    return t[:max_chars] if len(t) > max_chars else t


def is_present(domain: str, col: np.ndarray) -> np.ndarray:
    """统一"该概念被主张"的判定：ternary 只认断言，ordinal 认强度>=1，binary 认出现。

    正例率、互信息、共线分析必须共用这一个口径，否则同一条概念在不同指标里含义不同。
    """
    if domain == "ternary":
        return col == 1
    if domain == "ordinal":
        return col >= 1
    return col == 1


# ---------------------------------------------------------------- 阶段 1：标注

def load_vllm_model(model_dir: Path):
    """与 d:/fscbm/scripts/generate_scbm_concept.py 的 load_vllm_model 同构。

    max_model_len 取 2048（fscbm 是 1024）：那边 prompt 只有一句话，
    这边的 prompt 带文本 + 概念名 + 三条判定规则，长约 700 token，1024 余量不足。
    """
    from transformers import AutoTokenizer
    from vllm import LLM

    print(f"Loading tokenizer from {model_dir}")
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    llm_kwargs = dict(
        model=str(model_dir),
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=C.ANNOTATE["gpu_memory_utilization"],
        enable_prefix_caching=True,
        max_model_len=C.ANNOTATE["max_model_len"],
        max_num_seqs=C.ANNOTATE["max_num_seqs"],
        max_num_batched_tokens=C.ANNOTATE["max_num_batched_tokens"],
    )
    print(f"Loading vLLM model from {model_dir}")
    llm = LLM(**llm_kwargs)
    return tok, llm


def get_first_token_ids(word_list, tokenizer) -> list[int]:
    """与 fscbm 同名同实现：取每个词的首 token id 并去重。"""
    token_ids = []
    for word in word_list:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[0])
    if not token_ids:
        raise ValueError("get_first_token_ids ERROR: 词表中无有效 token")
    return list(dict.fromkeys(token_ids))


def extract_probs(first_token_logprobs: dict, state_ids: list[list[int]]):
    """首 token logprob → 各状态的归一化概率，与 fscbm 的 extract_binary_score 同构。

    返回 (各状态概率, 候选字的原始概率和)。后者是 prompt 约束力：
    模型如果没按格式只答一个字，候选字概率和会明显小于 1。
    """
    probs_dict = {tid: math.exp(obj.logprob) for tid, obj in first_token_logprobs.items()}
    probs = [sum(probs_dict.get(tid, 0.0) for tid in ids) for ids in state_ids]
    cover = sum(probs)
    total = cover + 1e-8
    return [p / total for p in probs], cover


def _smoke_report(labels: np.ndarray, conf: np.ndarray, covers: list[float],
                  batch: list[dict], texts: list[str], concepts: list[dict]):
    """首批结果诊断。解析类的错误只有真跑才会显形，这里一次核对完。"""
    n, k = labels.shape
    miss = float((labels < 0).mean())
    print("\n======== 首批诊断 ========")
    print(f"  {n} 条 × {k} 个概念，缺失率 {miss:.1%}"
          f"（候选字均未进 top-{C.ANNOTATE['top_logprobs']} 记为缺失）")
    if covers:
        cov = float(np.mean(covers))
        print(f"  候选字概率和均值 {cov:.3f}（1.0 表示模型只答了候选字，"
              f"明显偏低说明提示约束力不足）")

    groups: dict[str, list[int]] = {}
    for j, c in enumerate(concepts):
        groups.setdefault(c["domain"], []).append(j)

    for dom, cols in groups.items():
        sub = labels[:, cols]
        tot = int((sub >= 0).sum())
        names = (["未提及", "断言", "否认"] if dom == "ternary"
                 else ["不出现", "出现"] if dom == "binary"
                 else [str(v) for v in range(6)])
        share = []
        for v, nm in enumerate(names):
            cnt = int((sub == v).sum())
            share.append(f"{nm} {cnt / tot:.0%}" if tot else f"{nm} -")
        print(f"  {dom:<8}{' / '.join(share)}")

    print("  抽样核对：")
    for i in range(min(3, n)):
        for kk in range(min(3, k)):
            c = concepts[kk]
            v = int(labels[i, kk])
            if v < 0:
                nm = "缺失"
            elif c["domain"] == "ternary":
                nm = TERNARY_INV[v]
            elif c["domain"] == "binary":
                nm = BINARY_INV[v]
            else:
                nm = str(v)
            print(f"    #{batch[i]['_idx']:<5} {c['id']} {c['name']} → {nm} "
                  f"({float(conf[i, kk]):.2f})  «{texts[i][:24]}»")


def stage_annotate(dataset: str, split: str, limit: int | None, model_path: str | None):
    from tqdm import tqdm
    from vllm import SamplingParams

    concepts = load_concepts(Path(C.CONCEPT_CSV))
    records = load_records(dataset, split)
    n_all = len(records)

    if limit is not None:
        if limit <= 0:
            raise SystemExit("[error] --limit 必须是正整数")
        if limit < n_all:
            print(f"[warn] --limit {limit}：只跑前 {limit}/{n_all} 条，产物是冒烟结果，"
                  f"不能当正式标签用（正式跑去掉 --limit 会覆盖它）")
            records = records[:limit]

    n_total = len(records)
    if not n_total:
        raise SystemExit("[error] 没有可标注的样本")

    out_dir = Path(C.PROCESSED) / dataset / "annot"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = resolve_model(C.LLM_HINT, model_path)
    print(f"[info] 标注模型：{model_dir}")

    tok, llm = load_vllm_model(model_dir)

    # 每个状态的候选字 token id；顺序与 *_LABEL_ORDER 对齐
    state_ids = [[get_first_token_ids(words, tok) for words in state_tokens(c)] for c in concepts]
    demo = {}
    for c, ids in zip(concepts, state_ids):
        demo.setdefault(c["domain"], ids)
    print(f"[info] 候选 token id：{demo}")

    sp = SamplingParams(
        max_tokens=1,
        temperature=C.ANNOTATE["temperature"],
        logprobs=C.ANNOTATE["top_logprobs"],
    )
    suffix = C.ANNOTATE["prompt_suffix"]
    max_chars = C.ANNOTATE["max_text_chars"]
    batch_samples = C.ANNOTATE["batch_samples"]

    # 结果全部驻留内存：9600 × 55 的 int8 + float16 不到 2 MB，没必要落中间文件
    labels = np.full((n_total, len(concepts)), -1, dtype=np.int8)
    conf = np.zeros((n_total, len(concepts)), dtype=np.float16)

    # 全局 idx → 结果行号。必须在这里建一次：批内的局部下标会把后一批的结果写回前一批的行上
    row_of = {r["_idx"]: i for i, r in enumerate(records)}

    t0 = time.time()
    n_prompt = 0
    t_gen = None

    pbar = tqdm(
        total=n_total, desc=f"标注 {dataset}/{split}", unit="条", ncols=110,
        # 自定义 bar_format 必须显式带 {postfix}，否则 set_postfix_str 的内容不会显示
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    try:
        t_gen = time.time()
        for start in range(0, n_total, batch_samples):
            batch = records[start:start + batch_samples]
            texts = [prep_text(r, dataset, max_chars) for r in batch]

            prompts, meta = [], []
            covers = []
            for i, text in enumerate(texts):
                if not text:
                    continue                       # 空文本留 -1，不浪费一次生成
                for k, c in enumerate(concepts):
                    msgs = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_text(c, text)},
                    ]
                    prompts.append(
                        tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                        + suffix
                    )
                    meta.append((batch[i]["_idx"], k))

            if prompts:
                outs = llm.generate(prompts, sp, use_tqdm=False)
                n_prompt += len(prompts)
                if len(outs) != len(meta):
                    raise RuntimeError(f"输出数 {len(outs)} 与请求数 {len(meta)} 不一致")

                covers = []
                for j, o in enumerate(outs):
                    sidx, k = meta[j]
                    logprobs = o.outputs[0].logprobs
                    if not logprobs:
                        continue                   # 首个 token 即终止，vLLM 不给 logprobs
                    probs, cover = extract_probs(logprobs[0], state_ids[k])
                    covers.append(cover)
                    if cover <= 0:
                        continue                   # 候选字都没进 top-k，留 -1，不猜

                    best = int(np.argmax(probs))
                    domain = concepts[k]["domain"]
                    labels[row_of[sidx], k] = (TERNARY_LABEL_ORDER[best] if domain == "ternary"
                                               else BINARY_LABEL_ORDER[best] if domain == "binary"
                                               else best)
                    conf[row_of[sidx], k] = probs[best]

            pbar.update(len(batch))
            pbar.set_postfix_str(
                f"{n_prompt} prompt | {n_prompt / max(time.time() - t_gen, 1.0):.0f} p/s"
            )

            if start == 0:
                _smoke_report(labels[:len(batch)], conf[:len(batch)], covers,
                              batch, texts, concepts)
    finally:
        pbar.close()

    gen_elapsed = time.time() - t_gen if t_gen else 0.0
    total_elapsed = time.time() - t0
    print(f"[done] 提交 {n_prompt} 条 prompt，生成 {gen_elapsed / 60:.1f} min，"
          f"含加载共 {total_elapsed / 60:.1f} min")

    if limit is not None and n_total < n_all and n_prompt:
        eta = gen_elapsed / n_prompt * n_all * len(concepts)
        print(f"[info] 按此速度，全量 {n_all} 条约需 {eta / 60:.0f} min（不含模型加载）")

    spec = C.DATASET_FILES[dataset]
    extra_keys = C.EXTRA_FIELDS.get(dataset, [])
    samples = [{
        "idx": r["_idx"],
        "text": prep_text(r, dataset, max_chars),
        "label": int(r.get(spec["label_key"], 0)),
        # 原样保留分层所需字段（TOXICN 的 expression/toxic_type/topic）
        **{k: r[k] for k in extra_keys if k in r},
    } for r in records]

    out_path = write_json(out_dir, dataset, split, concepts, model_dir, samples, labels, conf)
    print(f"[done] 写出 {out_path.name}：{len(samples)} 条样本 × {len(concepts)} 概念，"
          f"{out_path.stat().st_size / 1024 / 1024:.1f} MB")


def write_json(out_dir: Path, dataset: str, split: str, concepts: list[dict],
               model_dir: Path, samples: list[dict], labels: np.ndarray,
               conf: np.ndarray) -> Path:
    """单文件产物：meta + concepts + 每条样本的 labels/conf 全在一个 json 里。"""
    meta = {
        "dataset": dataset,
        "split": split,
        "n_samples": len(samples),
        "n_concepts": len(concepts),
        "n_dims": sum(2 if c["domain"] == "ternary" else 1 for c in concepts if c["in_bottleneck"]),
        "coding": {
            "ternary": "0=未提及 1=断言 2=否认",
            "binary": "0=不出现 1=出现",
            "ordinal": "0-5",
            "missing": "-1 表示候选字均未进入 top-k",
        },
        "llm": str(model_dir),
        "max_text_chars": C.ANNOTATE["max_text_chars"],   # 文本按此截断，json 里的 text 与实际送入模型的一致
        "concept_csv_mtime": Path(C.CONCEPT_CSV).stat().st_mtime,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    concepts_out = [{"id": c["id"], "name": c["name"], "domain": c["domain"],
                     "in_bottleneck": int(c["in_bottleneck"])} for c in concepts]

    out_path = out_dir / f"concept_labels_{split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write('{\n"meta": ')
        f.write(json.dumps(meta, ensure_ascii=False))
        f.write(',\n"concepts": ')
        f.write(json.dumps(concepts_out, ensure_ascii=False))
        f.write(',\n"samples": [\n')
        for i, s in enumerate(samples):
            rec = {"idx": int(s["idx"]), "text": s["text"], "label": int(s["label"])}
            for k, v in s.items():          # expression / toxic_type / topic 等分层字段
                if k not in ("idx", "text", "label"):
                    rec[k] = v
            rec["labels"] = labels[i].tolist()
            rec["conf"] = [round(float(v), 3) for v in conf[i]]
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write(",\n" if i < len(samples) - 1 else "\n")
        f.write("]\n}\n")
    return out_path


# ---------------------------------------------------------------- 阶段 2：质检

def stage_audit(dataset: str, split: str):
    from sklearn.metrics import mutual_info_score

    annot_dir = Path(C.PROCESSED) / dataset / "annot"
    pack = annot_dir / f"concept_labels_{split}.json"
    if not pack.exists():
        raise SystemExit(f"[error] 找不到 {pack}，先跑 annotate")

    payload = json.loads(pack.read_text(encoding="utf-8"))
    samples = payload["samples"]
    concepts = payload["concepts"]
    if not samples:
        raise SystemExit(f"[error] {pack.name} 里没有样本，annotate 是否正常结束？")
    if len(samples) < 100:
        print(f"[warn] 只有 {len(samples)} 条样本，质检统计不成立"
              f"（是不是 --limit 冒烟跑出来的 json？）")

    old_mtime = payload.get("meta", {}).get("concept_csv_mtime")
    cur_mtime = Path(C.CONCEPT_CSV).stat().st_mtime
    if old_mtime and abs(cur_mtime - old_mtime) > 1:
        print("[warn] 词典在打标之后被改过：本次质检用的是 json 里的旧概念定义，结论不可直接用于新词典")

    labels = np.array([s["labels"] for s in samples], dtype=np.int8)
    conf = np.array([s["conf"] for s in samples], dtype=np.float32)
    y = np.array([int(s["label"]) for s in samples], dtype=int)

    n, k = labels.shape
    if len(concepts) != k:
        raise SystemExit(f"[error] 概念数 {len(concepts)} 与矩阵列数 {k} 不一致，json 是否被改过？")

    py = np.bincount(y) / max(len(y), 1)
    h_label = float(-np.sum(py[py > 0] * np.log(py[py > 0])))

    # 正例率、互信息、共线三处共用 is_present 口径，否则同一概念在不同指标里含义不同
    present = np.zeros((n, k), dtype=np.int8)
    for j, c in enumerate(concepts):
        present[:, j] = is_present(c["domain"], labels[:, j]).astype(np.int8)

    # 共线只用"无缺失"的样本：把 -1 当成"不出现"去算相关系数，会把相关系统性压低
    complete = (labels >= 0).all(axis=1)
    n_complete = int(complete.sum())
    if n_complete >= 2:
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.nan_to_num(np.corrcoef(present[complete].T), nan=0.0)
        if n_complete < n:
            print(f"[info] 共线分析用 {n_complete}/{n} 条无缺失样本")
    else:
        corr = np.zeros((k, k))
        print("[warn] 无缺失样本不足 2 条，共线分析跳过")

    rows = []
    for j, c in enumerate(concepts):
        col = labels[:, j]
        mask = col >= 0
        miss = int((col == -1).sum())
        valid = col[mask]
        pres = present[:, j].astype(bool)
        rate = float(pres[mask].mean()) if mask.any() else 0.0

        dist = {}
        if len(valid):
            vals, cnts = np.unique(valid, return_counts=True)
            for v, cnt in zip(vals, cnts):
                key = (TERNARY_INV.get(int(v)) if c["domain"] == "ternary"
                       else BINARY_INV.get(int(v)) if c["domain"] == "binary" else int(v))
                dist[str(key)] = round(float(cnt) / len(valid), 4)

        mi = float(mutual_info_score(y[mask], pres[mask].astype(int))) if mask.any() else 0.0
        mi_norm = mi / h_label if h_label > 0 else 0.0

        dup = [concepts[t]["id"] for t in np.where(np.abs(corr[j]) > 0.9)[0] if t != j]
        mean_conf = float(conf[mask, j].mean()) if mask.any() else 0.0

        flags = []
        if miss / n > 0.05:
            flags.append("缺失率高")
        # ternary 的"否认"是有效信号：断言和否认都几乎为 0 才是死维度。
        # 只看断言会把"高否认率"的概念误判为死维度，而否认正是隐式样本的核心信号。
        deny_rate = dist.get("否认", 0.0) if c["domain"] == "ternary" else 0.0
        if rate < 0.01 and deny_rate < 0.01:
            flags.append("死维度")
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
    for flag in ("死维度", "置信低", "缺失率高", "近标签代理", "高共线"):
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
    ap.add_argument("--limit", type=int, default=None,
                    help="只跑前 N 条，用于跑通链路；产物会覆盖正式结果，正式跑别带这个参数")
    a = ap.parse_args()

    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in ("annotate", "audit")]
    if unknown:
        raise SystemExit(f"[error] 未知阶段 {unknown}，可选 annotate / audit")

    print(f"[info] 项目根目录：{C.ROOT}")
    for s in stages:
        print(f"\n===== stage: {s} ({a.dataset}/{a.split}) =====")
        if s == "annotate":
            stage_annotate(a.dataset, a.split, a.limit, a.model_path)
        else:
            stage_audit(a.dataset, a.split)


if __name__ == "__main__":
    main()
