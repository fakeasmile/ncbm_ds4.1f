# -*- coding: utf-8 -*-
"""离线自检：不加载模型，用伪造的 vLLM 跑通 标注 + 质检，逐格核对数值。

用途：每次改完 pipeline.py 跑一次，全部 [ok] 才算可用。
    python scripts/selfcheck.py

覆盖的都是真实踩过的坑：
  1 路径与数据文件可达
  2 词典字段完整、三元概念有 deny_rule、无花括号（format 会炸）
  3 55 个概念 × 真实文本全部能渲染
  4 每个概念各状态的候选 token 互不重叠（重叠会让 argmax 失效）
  5 脏文本（NaN / None / int / 空串）不抛异常
  6 标签映射：ternary/binary/ordinal 逐格等于预期
  7 跨批写回：后一批不能覆盖前一批
  8 空文本样本整行 -1，且不阻塞后续样本
  9 json 结构、列数、n_dims 与词典一致
 10 质检数值与手工复算一致；缺失样本不进分母；死维度判定区分断言与否认
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.pipeline as P
from configs import pipeline_config as C

FAILED = []


def check(name: str, fn):
    try:
        detail = fn()
        print(f"[ok]   {name}" + (f"：{detail}" if detail else ""))
    except Exception as e:                        # noqa: BLE001 自检要把任何异常都算失败
        FAILED.append(name)
        print(f"[FAIL] {name}：{type(e).__name__}: {e}")


def need(cond, msg=""):
    if not cond:
        raise AssertionError(msg or "断言失败")


# ---------------------------------------------------------------- 伪造 vLLM
class _Bar:
    def __init__(self, *a, **k): pass
    def update(self, n=1): pass
    def set_postfix_str(self, *a, **k): pass
    def close(self): pass


class Tok:
    pad_token = "<pad>"

    def __init__(self):
        self._ids = {}

    def encode(self, w, add_special_tokens=False):
        if w not in self._ids:
            self._ids[w] = 1000 + len(self._ids)
        return [self._ids[w]]

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "||".join(m["content"] for m in msgs)


class LP:
    def __init__(self, v): self.logprob = v


class Out:
    def __init__(self, logprobs): self.outputs = [types.SimpleNamespace(logprobs=logprobs)]


def install_fakes(concepts: list[dict], records: list[dict]):
    """把 pipeline 的模型相关函数替换成伪造件，并返回预期表。"""
    tok = Tok()
    state_ids = [[P.get_first_token_ids(w, tok) for w in P.state_tokens(c)] for c in concepts]
    idx_of = {c["name"]: i for i, c in enumerate(concepts)}
    expected = {}
    counter = {"n": 0}

    class LLM:
        def generate(self, prompts, sp, use_tqdm=False):
            outs = []
            for prompt in prompts:
                counter["n"] += 1
                m = re.search(r"样本(\d+)", prompt)
                sidx = int(m.group(1)) if m else -1
                # 按"【概念】"后面那一整行取概念名：子串匹配会让"贬称"命中"群体贬称/隐性贬称"
                nm = re.search(r"【概念】(.+)", prompt)
                need(nm is not None, f"提示里没有概念行：{prompt[:60]}")
                name = nm.group(1).strip()
                need(name in idx_of, f"概念名未登记：{name}")
                k = idx_of[name]
                cd = state_ids[k]
                if sidx >= 0 and (sidx + k) % 19 == 0:        # 人为制造缺失
                    outs.append(Out([{999999: LP(0.0)}]))
                    continue
                st = abs(sidx * 7 + k) % len(cd)
                probs = [0.03] * len(cd)
                probs[st] = 0.9
                d = {t: LP(float(np.log(probs[s] / len(ids))))
                     for s, ids in enumerate(cd) for t in ids}
                outs.append(Out([d]))
                dom = concepts[k]["domain"]
                lab = (P.TERNARY_LABEL_ORDER[st] if dom == "ternary"
                       else P.BINARY_LABEL_ORDER[st] if dom == "binary" else st)
                expected[(sidx, k)] = (lab, round(probs[st] / sum(probs), 3))
            return outs

    P.load_concepts = lambda p: concepts
    P.load_records = lambda d, s: records
    P.resolve_model = lambda h, e=None: C.MODELS_DIR / "_fake"
    P.load_vllm_model = lambda d: (tok, LLM())
    fake = types.ModuleType("vllm")
    fake.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
    sys.modules["vllm"] = fake
    fake_bar = types.ModuleType("tqdm")
    fake_bar.tqdm = _Bar
    sys.modules["tqdm"] = fake_bar
    return expected, state_ids


def run_and_read(records, concepts, batch, limit=None):
    C.ANNOTATE["batch_samples"] = batch
    expected, _ = install_fakes(concepts, records)
    P.stage_annotate("TOXICN", "train", limit, None)
    path = C.PROCESSED / "TOXICN" / "annot" / "concept_labels_train.json"
    j = json.loads(path.read_text(encoding="utf-8"))
    labels = np.array([s["labels"] for s in j["samples"]], dtype=np.int8)
    conf = np.array([s["conf"] for s in j["samples"]], dtype=np.float32)
    return j, labels, conf, expected


TMP = ROOT / "_selfcheck_tmp"
shutil.rmtree(TMP, ignore_errors=True)
(TMP / "p").mkdir(parents=True)
(TMP / "e").mkdir(parents=True)
C.PROCESSED = TMP / "p"
C.EXPERIMENTS = TMP / "e"

ALL = P.load_concepts(C.CONCEPT_CSV)
CON3 = ([c for c in ALL if c["domain"] == "ternary"][:1]
        + [c for c in ALL if c["domain"] == "binary"][:1]
        + [c for c in ALL if c["domain"] == "ordinal"][:1])
RECS = [{"_idx": i, "content": f"样本{i}", "toxic": i % 2} for i in range(40)]


# ---------------------------------------------------------------- 各项检查
def c_paths():
    need(C.CONCEPT_CSV.exists(), f"词典不存在 {C.CONCEPT_CSV}")
    for ds in C.DATASET_FILES.values():
        for sp in ("train", "test"):
            need(ds[sp].exists(), f"数据不存在 {ds[sp]}")
    return f"root={C.ROOT.name}"


def c_dictionary():
    domains = {}
    for c in ALL:
        domains[c["domain"]] = domains.get(c["domain"], 0) + 1
        text = c["name"] + c["claim"] + c["assert_rule"] + c["deny_rule"] + c["absent_rule"]
        need("{" not in text and "}" not in text, f"{c['id']} 含花括号，format 会炸")
        for k in ("name", "claim", "assert_rule", "absent_rule"):
            need(c[k].strip(), f"{c['id']} 的 {k} 为空")
        if c["domain"] == "ternary":
            need(c["deny_rule"].strip(), f"{c['id']} 是三元但缺 deny_rule")
        else:
            need(not c["deny_rule"].strip(), f"{c['id']} 非三元却填了 deny_rule")
    names = [c["name"] for c in ALL]
    need(len(set(names)) == len(names), "概念名重复，提示里无法唯一定位")
    k = sum(2 if c["domain"] == "ternary" else 1 for c in ALL if c["in_bottleneck"])
    return f"{len(ALL)} 条 {domains}，瓶颈 {k} 维"


def c_render():
    texts = [P.prep_text(r, "TOXICN", 512) for r in P.load_records("TOXICN", "train")[:3]]
    n = 0
    longest = 0
    for t in texts:
        for c in ALL:
            s = P.build_user_text(c, t)
            n += 1
            longest = max(longest, len(s))
    return f"{n} 条全部渲染，最长 {longest} 字"


def c_state_ids():
    tok = Tok()
    for c in ALL:
        per = [[tok.encode(w, add_special_tokens=False)[0] for w in words]
               for words in P.state_tokens(c)]
        flat = [i for ids in per for i in ids]
        need(len(flat) == len(set(flat)), f"{c['id']} 各状态候选 token 重叠")
        need(len(per) == (3 if c["domain"] == "ternary" else 6 if c["domain"] == "ordinal" else 2),
             f"{c['id']} 状态数与 domain 不符")
    return "55 个概念的状态候选互不重叠"


def c_dirty_text():
    got = [P.prep_text(r, "COLD", 512) for r in
           [{"content": "正常"}, {"content": float("nan")}, {"content": None},
            {"content": 123}, {"content": "  空格  "}, {"content": ""}]]
    need(got == ["正常", "", "", "", "空格", ""], str(got))
    return "NaN/None/int/空串 均安全"


def c_single_batch():
    j, L, Q, exp = run_and_read(RECS, CON3, batch=200)
    need(len(j["samples"]) == 40, len(j["samples"]))
    bad = [(i, k) for (i, k), (lab, cf) in exp.items()
           if L[i, k] != lab or abs(Q[i, k] - cf) > 0.002]
    need(not bad, f"{len(bad)} 格不符：{bad[:3]}")
    return f"{len(exp)} 格全对"


def c_cross_batch():
    j, L, Q, exp = run_and_read(RECS, CON3, batch=7)
    bad = [(i, k) for (i, k), (lab, cf) in exp.items() if L[i, k] != lab]
    need(not bad, f"{len(bad)} 格不符：{bad[:3]}")
    filled = int((L >= 0).sum())
    need(filled / L.size > 0.8, f"有效格仅 {filled}/{L.size}，后几批可能没写回")
    return f"7 条/批 × 6 批，{filled}/{L.size} 有效格"


def c_full_concepts():
    recs = [{"_idx": i, "content": f"样本{i}", "toxic": i % 2} for i in range(12)]
    j, L, Q, exp = run_and_read(recs, ALL, batch=5)
    bad = [(i, k) for (i, k), (lab, cf) in exp.items() if L[i, k] != lab]
    need(not bad, f"{len(bad)} 格不符：{bad[:3]}")
    need(L.shape == (12, len(ALL)), L.shape)
    return f"55 概念 × 12 条，{L.shape[1]} 列"


def c_empty_text():
    recs = [{"_idx": i, "content": "" if i < 4 else f"样本{i}", "toxic": i % 2} for i in range(12)]
    j, L, Q, exp = run_and_read(recs, CON3, batch=4)
    need((L[:4] == -1).all(), "空文本行必须整行 -1")
    need((L[4:] >= -1).all() and (L[4:] != -1).any(), "空文本之后的样本没跑")
    return "前 4 条整行 -1，后续正常"


def c_json_schema():
    j, L, Q, _ = run_and_read(RECS, ALL, batch=200)
    need(list(j.keys()) == ["meta", "concepts", "samples"], list(j.keys()))
    need(len(j["concepts"]) == len(ALL))
    need(all(set(c) == {"id", "name", "domain", "in_bottleneck"} for c in j["concepts"]))
    s = j["samples"][0]
    need({"idx", "text", "label", "labels", "conf"} <= set(s), set(s))
    need(len(s["labels"]) == len(ALL) and len(s["conf"]) == len(ALL))
    dims = sum(2 if c["domain"] == "ternary" else 1 for c in ALL if c["in_bottleneck"])
    need(j["meta"]["n_dims"] == dims, f"{j['meta']['n_dims']} != {dims}")
    need(j["meta"]["n_samples"] == len(j["samples"]))
    need("max_text_chars" in j["meta"])
    return f"{len(j['samples'])} 条 × {len(ALL)} 列，n_dims={dims}"


def c_audit_numbers():
    j, L, Q, _ = run_and_read(RECS, CON3, batch=200)
    P.stage_audit("TOXICN", "train")
    import csv as _csv
    rows = {r["id"]: r for r in _csv.DictReader(
        open(C.EXPERIMENTS / "audit" / "TOXICN_concept_health.csv".replace("TOXICN_", "TOXICN_train_"),
             encoding="utf-8-sig"))}
    for k, c in enumerate(CON3):
        col = L[:, k]
        mask = col >= 0
        pres = P.is_present(c["domain"], col)
        manual = round(float(pres[mask].mean()) if mask.any() else 0.0, 4)
        need(abs(float(rows[c["id"]]["positive_rate"]) - manual) < 1e-9,
             f"{c['id']} 正例率 {rows[c['id']]['positive_rate']} != {manual}")
        need(int(float(rows[c["id"]]["missing_rate"]) * len(col) + 0.5) == int((col == -1).sum()),
             f"{c['id']} 缺失数不符")
    return "正例率/缺失率与手工复算一致"


def c_dead_dim():
    pack = C.PROCESSED / "TOXICN" / "annot" / "concept_labels_train.json"
    fake = {"meta": {"concept_csv_mtime": None},
            "concepts": [{"id": "C001", "name": "a", "domain": "ternary", "in_bottleneck": 1},
                         {"id": "C002", "name": "b", "domain": "ternary", "in_bottleneck": 1}],
            "samples": [{"idx": i, "text": "", "label": i % 2,
                         "labels": [2 if i % 2 else 0, -1], "conf": [0.9, 0.0]} for i in range(120)]}
    pack.write_text(json.dumps(fake, ensure_ascii=False), encoding="utf-8")
    P.stage_audit("TOXICN", "train")
    import csv as _csv
    rows = {r["id"]: r for r in _csv.DictReader(
        open(C.EXPERIMENTS / "audit" / "TOXICN_train_concept_health.csv", encoding="utf-8-sig"))}
    need("死维度" not in rows["C001"]["flags"], "全否认的概念被判死维度")
    need("死维度" in rows["C002"]["flags"], "全缺失的概念没被判死维度")
    return "全否认不死、全缺失判死"


for name, fn in [
    ("路径与数据可达", c_paths),
    ("词典完整性", c_dictionary),
    ("提示渲染", c_render),
    ("候选 token 唯一", c_state_ids),
    ("脏文本", c_dirty_text),
    ("单批：逐格标签/置信", c_single_batch),
    ("跨批：写回不串行", c_cross_batch),
    ("全 55 概念", c_full_concepts),
    ("空文本样本", c_empty_text),
    ("json 结构", c_json_schema),
    ("质检数值", c_audit_numbers),
    ("死维度判定", c_dead_dim),
]:
    check(name, fn)

shutil.rmtree(TMP, ignore_errors=True)

print()
if FAILED:
    print(f"[FAIL] {len(FAILED)} 项未通过：{FAILED}")
    sys.exit(1)
print("[PASS] 全部检查通过")
