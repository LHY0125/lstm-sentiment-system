# -*- coding: utf-8 -*-
"""数据层：语料下载、解析、分词、词表构建、索引化。

对外只暴露「文本 → 张量」所需的能力，不感知模型与训练。
"""

import csv
import random
from pathlib import Path

import requests

from senti import config

TIMEOUT = 60  # 语料约 3 MB，给足超时避免弱网下误判为源不可用

# 由 Task 2 Step 2 的探测结果固化。若探测发现更优的源，改这里即可。
# 语料统一落成 CSV（两列：text,label），后续解析逻辑与源解耦。
CORPUS_FILENAME = "chnsenticorp.csv"

# 2026-10-11 实测：hf-mirror.com 与 raw.githubusercontent.com 均返回 200，
# 且两处内容逐字节相同（sha256 前缀 1d414b9c65caf3a8，2 904 858 字节，7 766 条）。
# 放两个独立站点是为了单点故障时能自动回退——语料可得性是本方案的核心风险。
DOWNLOAD_URLS: list[str] = [
    "https://hf-mirror.com/datasets/dirtycomputer/ChnSentiCorp_htl_all/resolve/main/ChnSentiCorp_htl_all.csv",
    "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv",
]


def corpus_path() -> Path:
    return config.DATA_DIR / CORPUS_FILENAME


def is_downloaded() -> bool:
    """语料已落盘且非空。"""
    p = corpus_path()
    return p.exists() and p.stat().st_size > 0


def download(force: bool = False) -> Path:
    """下载语料并落盘为 CSV。

    已存在且 force=False 时直接返回，保证幂等。
    """
    path = corpus_path()
    if is_downloaded() and not force:
        print(f"语料已存在，跳过下载：{path}")
        return path

    if not DOWNLOAD_URLS:
        raise RuntimeError(
            "DOWNLOAD_URLS 为空。请先运行 backend/scripts/probe_sources.py 探测可用源，\n"
            "把确认可达的 URL 填入 senti/data.py 的 DOWNLOAD_URLS 列表。"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in DOWNLOAD_URLS:
        try:
            print(f"尝试下载：{url}")
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            path.write_bytes(r.content)
            print(f"已保存：{path}（{len(r.content)} bytes）")
            return path
        except Exception as exc:  # noqa: BLE001 —— 逐个源回退，记录最后一个错误
            last_error = exc
            print(f"  失败：{type(exc).__name__}: {exc}")

    raise RuntimeError(f"所有下载源均失败，最后一个错误：{last_error}")


def _read_rows() -> list[tuple[str, int]]:
    """读取 CSV，返回 (文本, 标签) 列表。

    兼容两种列名：text/label（英文）与 文本/标签（中文）。
    标签同时接受 0/1 与 -1/1 编码，统一映射为 1=积极 / 0=消极。
    """
    path = corpus_path()
    if not path.exists():
        raise FileNotFoundError(f"找不到语料：{path}\n请先调用 download()")

    rows: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        text_key = next((k for k in ("text", "文本", "review", "content") if k in fields), None)
        label_key = next((k for k in ("label", "标签", "sentiment") if k in fields), None)
        if text_key is None or label_key is None:
            raise ValueError(f"CSV 列名无法识别：{fields}（需要 text/label 或 文本/标签）")

        for row in reader:
            raw_label = str(row[label_key]).strip()
            text = str(row[text_key])
            # -1/1 编码 → 0/1；0/1 编码原样保留
            label = 0 if raw_label in ("0", "-1") else 1
            rows.append((text, label))
    return rows


def load_split() -> tuple[list[str], list[int], list[str], list[int]]:
    """载入并划分语料。

    R3 规定「语料自带 train/test 划分时优先沿用」。本语料不适用该分支：
    下载到的是单一未划分文件（仅 label,review 两列，无 split 列），
    故走 R3 的后备分支——合并全部样本后按固定种子 9:1 重切。
    这样做划分完全可复现，也不依赖语料自带的目录结构。

    返回 (train_texts, train_labels, test_texts, test_labels)。
    """
    rows = _read_rows()
    if not rows:
        raise ValueError("语料为空")

    print(f"使用划分方式：单一未划分文件 → 固定种子 {config.RANDOM_SEED} 按 "
          f"{1 - config.TEST_RATIO:.0%}/{config.TEST_RATIO:.0%} 重切")

    # 先按固定种子打乱，避免语料按标签排序导致切分后类别失衡
    rng = random.Random(config.RANDOM_SEED)
    rng.shuffle(rows)

    n_test = max(1, int(len(rows) * config.TEST_RATIO))
    test_rows, train_rows = rows[:n_test], rows[n_test:]

    train_texts = [t for t, _ in train_rows]
    train_labels = [y for _, y in train_rows]
    test_texts = [t for t, _ in test_rows]
    test_labels = [y for _, y in test_rows]
    print(f"语料载入：训练 {len(train_texts)} 条 / 测试 {len(test_texts)} 条")
    return train_texts, train_labels, test_texts, test_labels
