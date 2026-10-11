# 基于 LSTM 的网络评论情感分类系统 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个前后端分离的中文评论情感分类系统——PyTorch 训练 LSTM 模型产出多组 checkpoint，Flask 提供推理与可解释性接口，React 前端提供单条分析、批量分析与参数调节界面。

**Architecture:** 训练期离线产出「模型 + 词表 + 元数据」三类产物，推理期只消费产物、不做训练。后端按数据层 / 模型层 / 训练层 / 推理层 / 接口层五层划分；「模型可切换」等价于切换一个 `.pt` 文件。前端经 Vite 代理访问 `/api`，开发期同源无 CORS。

**Tech Stack:** Python 3.13.13 + PyTorch 2.12.0+cu132 + jieba + Flask + pandas + pytest｜React 18 + Vite + TypeScript + Tailwind v4 + pnpm + Biome

**Spec:** `docs/superpowers/specs/2026-10-11-lstm-sentiment-system-design.md`

## Global Constraints

以下为项目级约束，**每个任务的要求都隐含包含本节**：

- **工作目录**：`D:\Code\doing_exercises\programs\作业\基于 LSTM 的网络评论情感分类系统设计与实现\`
- **解释器**：一律使用项目级 `.venv`，禁止使用全局 Python 或兄弟项目的 `.venv`
  - `.venv\Scripts\python.exe`（Python 3.13.13，uv 创建）
- **前端工具**：`pnpm` 11.7.0（node v24.19.0 已装）
- **不安装 torchvision**：本项目无图像任务，装了反而引入版本配对坑
- **注释语言**：中文；注释解释「为什么」而非「做了什么」
- **命名**：Python 变量与函数 `snake_case`，类 `PascalCase`，常量 `UPPER_SNAKE_CASE`；
  TypeScript 组件 `PascalCase`，其余 `camelCase`
- **随机种子**：`RANDOM_SEED = 42`，所有训练入口必须调用 `set_seed()`
- **测试集纪律（选择 vs 报告）**：测试集**只**在训练全部结束后报告最终指标。
  凡代码做**选择**之处——选最优轮次、选「最优配置」、模型列表排序、前端默认选项——
  一律依据**验证**准确率；测试准确率照常逐条返回，但不参与任何排序或挑选。
  理由：对 6 个测试准确率取极大值会被系统性高估（winner's curse），
  使报告的头条数字失去无偏性。
- **路径定位**：脚本用 `Path(__file__)` 定位项目根，不依赖当前工作目录
- **超参数**：全部集中在 `backend/senti/config.py`，训练/推理代码内不得出现字面量
- **重要**：先读 `docs/superpowers/specs/2026-10-11-lstm-sentiment-system-design.md`，
  它是本计划的设计依据。设计与计划冲突时以设计为准并回报。

---

# Phase 1：后端

## Task 1: 项目脚手架、虚拟环境与常量中心

**Files:**
- Create: `.gitignore`
- Create: `backend/senti/__init__.py`
- Create: `backend/senti/config.py`
- Create: `backend/common/__init__.py`
- Create: `backend/common/plot_font.py`
- Create: `backend/pytest.ini`
- Create: `backend/tests/__init__.py`
- Test: `backend/tests/test_config.py`

**Interfaces:**
- Consumes: 无（首个任务）
- Produces: `config.PROJECT_ROOT` / `DATA_DIR` / `ARTIFACT_DIR` / `OUTPUT_DIR`（均为 `Path`）；
  `config.MAX_LEN=150`、`MAX_VOCAB=20000`、`MIN_FREQ=2`、`PAD_IDX=0`、`UNK_IDX=1`、
  `EMBEDDING_DIM=128`、`DROPOUT=0.3`、`BATCH_SIZE=64`、`EPOCHS=15`、`LEARNING_RATE=1e-3`、
  `RANDOM_SEED=42`、`VAL_RATIO=0.1`、`CONTRIB_MAX_TOKENS=100`、`MAX_TEXT_CHARS=2000`；
  `config.GRID: list[tuple[int,int]]`（6 项）；`config.model_id(layers, hidden) -> str`；
  `config.ckpt_path(mid: str) -> Path`；`config.VOCAB_PATH: Path`；`config.META_PATH: Path`；
  `common.plot_font.setup_chinese_font() -> str | None`

- [ ] **Step 1: 初始化仓库（依赖未决事项 U1）**

若决定使用 git（spec 未决事项 U1），在项目根执行：

```powershell
git init
```

**若选择不初始化 git，则跳过本计划中所有 `Commit` 步骤**，其余步骤不受影响。

- [ ] **Step 2: 创建虚拟环境并安装依赖**

```powershell
uv venv .venv --python "D:\settings\Language\Python\Python 3.13.13\python.exe"
uv pip install --python .venv\Scripts\python.exe "d:/settings/Language/Python/库/torch-2.12.0+cu132-cp313-cp313-win_amd64.whl"
uv pip install --python .venv\Scripts\python.exe jieba flask flask-cors pandas matplotlib scikit-learn pytest requests pyarrow
```

- [ ] **Step 3: 验证环境**

Run:
```powershell
.venv\Scripts\python.exe -c "import torch, jieba, flask, pandas, matplotlib, sklearn, pytest, requests; print(torch.__version__, torch.cuda.is_available())"
```
Expected: `2.12.0+cu132 True`

- [ ] **Step 4: 创建 `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
backend/artifacts/
backend/输出/
数据集/
node_modules/
frontend/dist/
.vite/
*.log
```

- [ ] **Step 5: 写失败测试**

`backend/pytest.ini`：
```ini
[pytest]
pythonpath = .
testpaths = tests
filterwarnings =
    ignore:pkg_resources is deprecated as an API:UserWarning
```

`backend/tests/__init__.py`：空文件。

`backend/tests/test_config.py`：
```python
# -*- coding: utf-8 -*-
"""config.py 常量与辅助函数的测试。"""

from pathlib import Path

from senti import config


def test_grid_has_six_combinations():
    """网格必须是 2 种层数 × 3 种隐藏维度 = 6 组。"""
    assert len(config.GRID) == 6
    assert (1, 64) in config.GRID
    assert (2, 256) in config.GRID


def test_model_id_format():
    assert config.model_id(2, 128) == "lstm_l2_h128"


def test_paths_are_absolute_and_under_project_root():
    """路径必须绝对，且都能追到项目根，避免依赖当前工作目录。"""
    for p in (config.DATA_DIR, config.ARTIFACT_DIR, config.OUTPUT_DIR):
        assert isinstance(p, Path)
        assert p.is_absolute()
        assert config.PROJECT_ROOT in p.parents


def test_special_token_indices():
    """0 号必须留给 <pad>，1 号留给 <unk>，实词从 2 开始。"""
    assert config.PAD_IDX == 0
    assert config.UNK_IDX == 1
    assert config.FIRST_WORD_IDX == config.UNK_IDX + 1 == 2
    assert config.MAX_LEN > 0
    assert config.CONTRIB_MAX_TOKENS > 0
```

- [ ] **Step 6: 运行测试确认失败**

Run: `cd backend; ..\.venv\Scripts\python.exe -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'senti.config'`

- [ ] **Step 7: 实现 `config.py`**

`backend/senti/__init__.py`：空文件。

`backend/senti/config.py`：
```python
# -*- coding: utf-8 -*-
"""全局路径与超参数常量。

训练与推理代码不得内嵌超参数字面量，一律从此处引用——
这样「改一个数字重跑实验」只改一个地方。
"""

from pathlib import Path

# ---------------------------------------------------------------- 路径
# __file__ = <项目根>/backend/senti/config.py
# parents[0]=senti  parents[1]=backend  parents[2]=项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "数据集" / "ChnSentiCorp"
ARTIFACT_DIR = PROJECT_ROOT / "backend" / "artifacts"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "输出"

VOCAB_PATH = ARTIFACT_DIR / "vocab.json"
META_PATH = ARTIFACT_DIR / "meta.json"

# ---------------------------------------------------------------- 数据
MAX_LEN = 150          # 词级截断长度；中文分词后序列比英文短，150 覆盖绝大多数评论
MAX_VOCAB = 20000      # 词表上限，按词频取前 N 个
MIN_FREQ = 2           # 出现次数低于此值的词不入表，记为 <unk>
PAD_IDX = 0            # <pad> 的索引
UNK_IDX = 1            # <unk> 的索引
FIRST_WORD_IDX = UNK_IDX + 1   # 实词索引起点；<pad> 与 <unk> 之后

# ---------------------------------------------------------------- 模型
EMBEDDING_DIM = 128    # 词向量维度；网格实验中固定，不参与搜索
DROPOUT = 0.3
OUTPUT_DIM = 2         # 二分类：积极 / 消极

# ---------------------------------------------------------------- 训练
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
VAL_RATIO = 0.1        # 从训练集切分验证集，用于选最优轮次

# ---------------------------------------------------------------- 推理
CONTRIB_MAX_TOKENS = 100  # 逐词贡献分析的参与词数上限，超出部分不分析
MAX_TEXT_CHARS = 2000     # 单条文本字符数上限，超限接口返回 400

# ---------------------------------------------------------------- 网格
# 2 种层数 × 3 种隐藏维度 = 6 组 checkpoint。
# 刻意把 hidden_dim 与 num_layers 做成两维：报告里能得到「层数/维度 → 准确率/参数量」的对照表。
GRID: list[tuple[int, int]] = [
    (layers, hidden) for layers in (1, 2) for hidden in (64, 128, 256)
]


def model_id(layers: int, hidden: int) -> str:
    """模型标识，同时用作 checkpoint 文件名。"""
    return f"lstm_l{layers}_h{hidden}"


def ckpt_path(mid: str) -> Path:
    """某个模型的 checkpoint 路径。"""
    return ARTIFACT_DIR / f"{mid}.pt"
```

- [ ] **Step 8: 实现 `plot_font.py`**

`backend/common/__init__.py`：空文件。

`backend/common/plot_font.py`（从兄弟项目 `神经网络与深度学习/common/plot_font.py` 移植，
去掉本项目用不到的 `tk` / `inline` 后端分支——本系统只保存图片，不弹窗也不跑笔记本）：
```python
# -*- coding: utf-8 -*-
"""matplotlib 中文字体配置。

Windows 上 matplotlib 默认字体不含中文字形，直接画中文会显示成方块。
这里按优先级探测系统已装的中文字体，找到后设为默认字体。
"""

import matplotlib
from matplotlib import font_manager

# 候选字体：按优先级排列，前三项为 Windows 自带
CJK_FONT_CANDIDATES = [
    "Microsoft YaHei",   # 微软雅黑，Win7+ 自带
    "SimHei",            # 黑体，Windows 自带
    "SimSun",            # 宋体，Windows 自带
    "Noto Sans CJK SC",  # 跨平台开源
    "Source Han Sans SC",
]


def _find_available_font() -> str | None:
    """返回系统上第一个可用的中文字体名，找不到则返回 None。"""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_FONT_CANDIDATES:
        if name in installed:
            return name
    return None


def setup_chinese_font() -> str | None:
    """设为 Agg 后端（只保存图片、不弹窗）并配置中文字体。

    返回实际使用的中文字体名，未找到可用字体时返回 None。
    必须在 import matplotlib.pyplot 之前调用。
    """
    matplotlib.use("Agg")

    font_name = _find_available_font()
    if font_name:
        matplotlib.rcParams["font.sans-serif"] = [font_name]
        print(f"matplotlib 中文字体：{font_name}")
    else:
        print(f"[警告] 未找到中文字体（已探测 {CJK_FONT_CANDIDATES}），图中中文可能显示为方块")

    # 使用中文字体后负号会显示异常，需显式关闭
    matplotlib.rcParams["axes.unicode_minus"] = False
    return font_name
```

- [ ] **Step 9: 运行测试确认通过**

Run: `cd backend; ..\.venv\Scripts\python.exe -m pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 10: 验证字体模块可导入并生效**

Run:
```powershell
cd backend
..\.venv\Scripts\python.exe -c "from common.plot_font import setup_chinese_font; print(setup_chinese_font())"
```
Expected: 打印一个中文字体名（如 `Microsoft YaHei`），不是 `None`

- [ ] **Step 11: Commit**

```bash
git add .gitignore backend/
git commit -m "chore: 项目脚手架、虚拟环境配置与常量中心"
```

---

## Task 2: 语料获取与解析

**Files:**
- Create: `backend/scripts/probe_sources.py`
- Create: `backend/senti/data.py`
- Test: `backend/tests/test_data_load.py`

**Interfaces:**
- Consumes: `config.DATA_DIR`、`config.PROJECT_ROOT`
- Produces: `data.download(force: bool = False) -> Path`（返回语料文件路径）；
  `data.load_split() -> tuple[list[str], list[int], list[str], list[int]]`
  （依次为 训练文本、训练标签、测试文本、测试标签，标签 1=积极 / 0=消极）

**背景**：ChnSentiCorp 的可得性是本方案唯一的未知数（spec 第 8 节、风险表第 1 行）。
因此本任务**先探测、再固化**，不预先假定某个 URL 一定可用。

- [ ] **Step 1: 写源探测脚本**

`backend/scripts/probe_sources.py`：
```python
# -*- coding: utf-8 -*-
"""探测 ChnSentiCorp 的可用下载源。

已知网络事实（2026-10-11 实测）：
    hf-mirror.com              可达
    raw.githubusercontent.com  可达
    gitee.com                  可达
    huggingface.co             不可达

逐个探测候选源，打印状态码与内容长度，选定可用源后再固化为 download()。
"""

import sys
from pathlib import Path

import requests

TIMEOUT = 20

CANDIDATES: list[tuple[str, str]] = [
    # (说明, URL)
    ("HF镜像 seamew/ChnSentiCorp train",
     "https://hf-mirror.com/datasets/seamew/ChnSentiCorp/resolve/main/data/train-00000-of-00001.parquet"),
    ("HF镜像 seamew/ChnSentiCorp 文件列表",
     "https://hf-mirror.com/api/datasets/seamew/ChnSentiCorp"),
    ("HF镜像 lansinuote/ChnSentiCorp 文件列表",
     "https://hf-mirror.com/api/datasets/lansinuote/ChnSentiCorp"),
    ("GitHub raw 常见镜像（示例路径）",
     "https://raw.githubusercontent.com/lansinuote/NLPBeginnerTutorials/main/README.md"),
    ("HF镜像 数据集搜索 API",
     "https://hf-mirror.com/api/datasets?search=ChnSentiCorp"),
]


def main() -> int:
    for desc, url in CANDIDATES:
        try:
            r = requests.get(url, timeout=TIMEOUT, allow_redirects=True)
            print(f"[{r.status_code}] {desc}")
            print(f"        {url}")
            print(f"        content-length={len(r.content)} bytes")
            if r.status_code == 200 and len(r.content) < 4000:
                print(f"        预览: {r.text[:300]!r}")
        except Exception as exc:  # noqa: BLE001 —— 探测脚本，任何异常都只打印
            print(f"[ERR] {desc}\n        {url}\n        {type(exc).__name__}: {exc}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: 运行探测，记录结果**

Run: `cd backend; ..\.venv\Scripts\python.exe scripts/probe_sources.py`
Expected: 打印各候选源的 HTTP 状态码。**记录哪些返回 200**，下一步据此固化。

若 200 的是 `/api/datasets/<repo>` 文件列表，则从返回 JSON 的 `siblings` 字段
取实际文件名，构造 `https://hf-mirror.com/datasets/<repo>/resolve/main/<文件名>` 下载。

**若所有候选源均失败**：改用 spec 第 11 节的降级方案（换用其他公开中文情感语料），
并把实际使用的数据集名写入 `meta.json` 的 `dataset` 字段。

- [ ] **Step 3: 写失败测试**

`backend/tests/test_data_load.py`：
```python
# -*- coding: utf-8 -*-
"""语料获取与解析的测试。

这些测试依赖真实语料，未下载时会失败——这是有意为之：
数据可得性是本方案的核心风险，必须显式暴露而不是静默跳过。
"""

import pytest

from senti import config, data


@pytest.fixture(scope="module")
def corpus():
    """确保语料已下载，返回 (train_texts, train_labels, test_texts, test_labels)。"""
    if not data.is_downloaded():
        data.download()
    return data.load_split()


def test_corpus_size_is_reasonable(corpus):
    """语料规模应达数千条，太小说明下载或解析出了问题。"""
    train_texts, train_labels, test_texts, test_labels = corpus
    assert len(train_texts) >= 3000, f"训练集只有 {len(train_texts)} 条，疑似解析错误"
    assert len(test_texts) >= 500, f"测试集只有 {len(test_texts)} 条，疑似解析错误"
    assert len(train_texts) == len(train_labels)
    assert len(test_texts) == len(test_labels)


def test_both_classes_are_balanced(corpus):
    """二分类语料两类都应占相当比例，否则模型会退化成多数类预测。"""
    _, train_labels, _, test_labels = corpus
    for labels in (train_labels, test_labels):
        pos_ratio = sum(labels) / len(labels)
        assert 0.3 <= pos_ratio <= 0.7, f"类别失衡：积极占比 {pos_ratio:.2%}"


def test_labels_are_binary(corpus):
    _, train_labels, _, _ = corpus
    assert set(train_labels) <= {0, 1}


def test_texts_are_non_empty_chinese(corpus):
    """每条评论都应有实际内容，且包含中文汉字。"""
    train_texts, _, _, _ = corpus
    for text in train_texts[:200]:
        assert isinstance(text, str)
        assert text.strip(), "存在空评论"
    assert any("\u4e00" <= ch <= "\u9fff" for ch in train_texts[0]), "首条评论不含汉字"


def test_download_is_idempotent(corpus):
    """重复调用 download() 不应重复下载或报错。"""
    assert data.is_downloaded() is True
    data.download()
    assert data.is_downloaded() is True
```

- [ ] **Step 4: 运行测试确认失败**

Run: `cd backend; ..\.venv\Scripts\python.exe -m pytest tests/test_data_load.py -v`
Expected: FAIL — `ImportError: cannot import name 'data' from 'senti'`

- [ ] **Step 5: 实现 `data.py` 的下载与解析部分**

`backend/senti/data.py`（本步骤只实现下载与解析，分词/编码在 Task 3 补齐）：
```python
# -*- coding: utf-8 -*-
"""数据层：语料下载、解析、分词、词表构建、索引化。

对外只暴露「文本 → 张量」所需的能力，不感知模型与训练。
"""

import csv
import json
import sys
from pathlib import Path

import requests

from senti import config

TIMEOUT = 60

# 由 Task 2 Step 2 的探测结果固化。若探测发现更优的源，改这里即可。
# 语料统一落成 CSV（两列：text,label），后续解析逻辑与源解耦。
CORPUS_FILENAME = "chnsenticorp.csv"
DOWNLOAD_URLS: list[str] = [
    # 探测确认可达的 URL 在此填入，按优先级排列
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

    沿用语料自带的 train/test 划分。若 CSV 中无 split 列，则按 9:1 固定种子切分。

    返回 (train_texts, train_labels, test_texts, test_labels)。
    """
    rows = _read_rows()
    if not rows:
        raise ValueError("语料为空")

    # 先按固定种子打乱，避免语料按标签排序导致切分后类别失衡
    rng = random.Random(config.RANDOM_SEED)
    rng.shuffle(rows)

    n_test = max(1, int(len(rows) * 0.1))
    test_rows, train_rows = rows[:n_test], rows[n_test:]

    train_texts = [t for t, _ in train_rows]
    train_labels = [y for _, y in train_rows]
    test_texts = [t for t, _ in test_rows]
    test_labels = [y for _, y in test_rows]
    print(f"语料载入：训练 {len(train_texts)} 条 / 测试 {len(test_texts)} 条")
    return train_texts, train_labels, test_texts, test_labels
```

同时在文件顶部 import 区补上 `import random`。

> **说明**：若下载到的语料自带 train/test 划分（两个文件），把 `CORPUS_FILENAME` 改为列表、
> `_read_rows` 分别读取即可。本实现选择「合并后按固定种子重切」，
> 好处是划分完全可复现、且不依赖语料自带划分的目录结构。

- [ ] **Step 6: 固化下载源并下载语料**

把 Step 2 探测确认可达的 URL 填入 `backend/senti/data.py` 的 `DOWNLOAD_URLS`。

Run: `cd backend; ..\.venv\Scripts\python.exe -c "from senti import data; print(data.download())"`
Expected: 打印保存路径，文件大小不为 0

- [ ] **Step 7: 运行测试确认通过**

Run: `cd backend; ..\.venv\Scripts\python.exe -m pytest tests/test_data_load.py -v`
Expected: 5 passed

> 若 `test_corpus_size_is_reasonable` 或 `test_both_classes_are_balanced` 失败，
> 说明 CSV 列名识别或标签映射有误。用
> `..\.venv\Scripts\python.exe -c "from senti import data; print(data._read_rows()[:3])"`
> 检查前三条数据。

- [ ] **Step 8: Commit**

```bash
git add backend/scripts/ backend/senti/data.py backend/tests/test_data_load.py
git commit -m "feat(data): 语料探测、下载与解析"
```

---

## Task 3: 分词、词表与编码

**Files:**
- Modify: `backend/senti/data.py`（追加分词与编码函数）
- Test: `backend/tests/test_data_encode.py`

**Interfaces:**
- Consumes: `config.MAX_LEN`、`MAX_VOCAB`、`MIN_FREQ`、`UNK_IDX`、`VOCAB_PATH`
- Produces:
  - `data.tokenize(text: str) -> list[str]`
  - `data.build_vocab(texts: list[str]) -> dict[str, int]`（实词索引从 2 开始）
  - `data.encode(texts: list[str], word2idx: dict[str, int], max_len: int = config.MAX_LEN) -> tuple[np.ndarray, np.ndarray]`
  - `data.save_vocab(word2idx: dict[str, int]) -> Path` / `data.load_vocab() -> dict[str, int]`

- [ ] **Step 1: 写失败测试**

`backend/tests/test_data_encode.py`：
```python
# -*- coding: utf-8 -*-
"""分词、词表与编码的测试（纯函数，不依赖真实语料）。"""

import numpy as np

from senti import config, data


def test_tokenize_keeps_words():
    tokens = data.tokenize("这个酒店非常好")
    assert "酒店" in tokens
    assert len(tokens) >= 2


def test_tokenize_drops_pure_punctuation():
    """纯标点与空白不构成词，必须丢弃，否则词表被标点污染。"""
    assert data.tokenize("，。！？") == []
    assert data.tokenize("   ") == []


def test_tokenize_keeps_punctuation_attached_words():
    """分词结果里本就带标点的词（如 jieba 不拆的）不应被误删。"""
    tokens = data.tokenize("房间干净")
    assert all(t.strip() for t in tokens)


def test_build_vocab_starts_at_index_two():
    """0 与 1 属于 <pad>/<unk>，实词必须从 2 开始。"""
    vocab = data.build_vocab(["很好 很好 很差"])
    assert min(vocab.values()) == 2


def test_build_vocab_respects_min_freq():
    """出现次数低于 min_freq（默认 2）的词不入表。

    输入刻意用空格分隔的单字，避免 jieba 把「很好 很好 很差」整体切成
    一个长词而让断言失效——分词结果可用 Task 3 Step 5 的命令实地核对。
    """
    vocab = data.build_vocab(["甲 甲 甲 乙 丙"])
    assert "甲" in vocab        # 出现 3 次
    assert "乙" not in vocab    # 只出现 1 次
    assert "丙" not in vocab    # 只出现 1 次


def test_build_vocab_orders_by_frequency():
    """高频词应拿到更小的索引，便于按频次截断词表。"""
    vocab = data.build_vocab(["甲 甲 甲 乙 乙 丙"])
    assert vocab["甲"] < vocab["乙"] < vocab["丙"]


def test_encode_shapes_and_lengths():
    word2idx = {"很好": 2, "酒店": 3}
    ids, lengths = data.encode(["很好 酒店", "很好"], word2idx, max_len=5)

    assert ids.shape == (2, 5)
    assert ids.dtype == np.int64
    assert lengths.tolist() == [2, 1]


def test_encode_pads_with_zero():
    word2idx = {"很好": 2}
    ids, _ = data.encode(["很好"], word2idx, max_len=4)
    assert ids[0].tolist() == [2, 0, 0, 0]


def test_encode_unknown_word_maps_to_unk():
    word2idx = {"很好": 2}
    ids, lengths = data.encode(["很好 从未见过的词"], word2idx, max_len=4)
    assert ids[0][1] == config.UNK_IDX
    assert lengths[0] == 2


def test_encode_truncates_to_max_len():
    word2idx = {"很好": 2}
    ids, lengths = data.encode(["很好 很好 很好 很好"], word2idx, max_len=2)
    assert ids.shape == (1, 2)
    assert lengths[0] == 2


def test_encode_empty_text_has_length_one():
    """全空评论必须给出 length>=1，否则模型里 gather 的索引会越界。"""
    ids, lengths = data.encode(["，。"], {}, max_len=4)
    assert lengths[0] == 1
    assert ids[0].tolist() == [0, 0, 0, 0]


def test_vocab_roundtrip(tmp_path, monkeypatch):
    """词表存盘再读回必须完全一致——词表与 checkpoint 的配对依赖这一点。"""
    monkeypatch.setattr(config, "VOCAB_PATH", tmp_path / "vocab.json")
    vocab = {"很好": 2, "酒店": 3}
    data.save_vocab(vocab)
    assert data.load_vocab() == vocab
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend; ..\.venv\Scripts\python.exe -m pytest tests/test_data_encode.py -v`
Expected: FAIL — `AttributeError: module 'senti.data' has no attribute 'tokenize'`

- [ ] **Step 3: 实现分词、词表与编码**

把以下内容追加到 `backend/senti/data.py`，并在文件顶部 import 区补
`import re`、`from collections import Counter`、`import numpy as np`、`import jieba`：

```python
# ---------------------------------------------------------------- 分词

# 纯标点/空白词判定：整体由非单词字符构成即为纯标点。
# Python 3 的 re 默认 Unicode 模式，汉字属于 \w，故中文词不会被误判。
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """jieba 分词并丢弃纯标点/空白词。

    丢弃标点是有意的：标点对情感极性的判别贡献极低，
    却会占据词表容量并稀释 <unk> 的统计。
    """
    return [w for w in jieba.lcut(text) if not _PUNCT_ONLY_RE.match(w)]


# ---------------------------------------------------------------- 词表


def build_vocab(texts: list[str]) -> dict[str, int]:
    """仅用训练集文本构建词表。

    只统计训练集是硬性要求：用全部数据建表会把测试集词频泄漏进模型。

    索引分配：0=<pad>，1=<unk>，实词从 2 开始。
    """
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))

    # 先按词频取前 MAX_VOCAB 个，再滤掉低频词
    words = [w for w, c in counter.most_common(config.MAX_VOCAB) if c >= config.MIN_FREQ]
    vocab = {w: i + config.FIRST_WORD_IDX for i, w in enumerate(words)}
    print(f"词表大小：{len(vocab)}（原始不同词数 {len(counter)}）")
    return vocab


def save_vocab(word2idx: dict[str, int]) -> Path:
    """词表写盘为 JSON。必须与 checkpoint 成对保存，否则推理时静默错位。"""
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    config.VOCAB_PATH.write_text(
        json.dumps(word2idx, ensure_ascii=False, indent=0), encoding="utf-8"
    )
    print(f"词表已保存：{config.VOCAB_PATH}")
    return config.VOCAB_PATH


def load_vocab() -> dict[str, int]:
    if not config.VOCAB_PATH.exists():
        raise FileNotFoundError(f"找不到词表：{config.VOCAB_PATH}\n请先运行训练")
    return json.loads(config.VOCAB_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 编码


def encode(
    texts: list[str],
    word2idx: dict[str, int],
    max_len: int = config.MAX_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """文本 → (定长索引矩阵, 真实词数)。

    超出 max_len 的尾部截断，不足的在**后侧**补 PAD_IDX。

    lengths 记录真实词数（不含填充），是模型取「最后一个真实词」的必要输入。
    取 output[-1] 会读到填充位——这是本项目继承自课程实验四的核心教训。
    lengths 至少为 1，避免全空文本导致 gather 索引越界。
    """
    ids_out = np.full((len(texts), max_len), config.PAD_IDX, dtype=np.int64)
    lengths = np.zeros(len(texts), dtype=np.int64)

    for i, text in enumerate(texts):
        seq = [word2idx.get(w, config.UNK_IDX) for w in tokenize(text)][:max_len]
        ids_out[i, : len(seq)] = seq
        lengths[i] = max(len(seq), 1)

    return ids_out, lengths
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend; ..\.venv\Scripts\python.exe -m pytest tests/test_data_encode.py -v`
Expected: 12 passed

> 若 `test_build_vocab_respects_min_freq` 因 jieba 切词方式而失败，
> 用 `..\.venv\Scripts\python.exe -c "from senti import data; print(data.tokenize('很好 很好 很差'))"`
> 看实际切分结果，按结果调整该用例的输入语料（**不要放宽 min_freq 断言**）。

- [ ] **Step 5: 在真实语料上验证分词与词表**

Run:
```powershell
cd backend
..\.venv\Scripts\python.exe -c @"
from senti import data
train_texts, _, _, _ = data.load_split()
print('样例分词：', data.tokenize(train_texts[0])[:20])
vocab = data.build_vocab(train_texts)
ids, lengths = data.encode(train_texts[:5], vocab)
import numpy as np
print('编码形状：', ids.shape, '真实词数：', lengths.tolist())
print('词数中位数：', int(np.median([len(data.tokenize(t)) for t in train_texts[:2000]])))
"@
```
Expected: 打印合理的中文分词结果；**记录「词数中位数」——若显著大于 `MAX_LEN=150`，
需要把 `config.MAX_LEN` 调大**，否则大量评论被截断。

- [ ] **Step 6: Commit**

```bash
git add backend/senti/data.py backend/tests/test_data_encode.py
git commit -m "feat(data): jieba 分词、词表构建与序列编码"
```

---

## Task 4: LSTM 模型

**Files:**
- Create: `backend/senti/model.py`
- Test: `backend/tests/test_model.py`

**Interfaces:**
- Consumes: `config.EMBEDDING_DIM`、`DROPOUT`、`OUTPUT_DIM`、`PAD_IDX`
- Produces: `model.SentimentLSTM(vocab_size, hidden_dim, num_layers, embedding_dim=128, dropout=0.3, output_dim=2)`；
  `forward(x: Tensor[seq_len, batch], lengths: Tensor[batch]) -> Tensor[batch, output_dim]`

- [ ] **Step 1: 写失败测试**

`backend/tests/test_model.py`：
```python
# -*- coding: utf-8 -*-
"""LSTM 模型结构与形状的测试。"""

import torch

from senti import model


def _build(vocab_size=20, hidden_dim=8, num_layers=1):
    torch.manual_seed(0)
    return model.SentimentLSTM(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=num_layers)


def test_forward_output_shape():
    m = _build()
    x = torch.randint(2, 20, (7, 4))      # (seq_len=7, batch=4)
    lengths = torch.tensor([7, 5, 3, 1])
    out = m(x, lengths)
    assert out.shape == (4, 2)


def test_single_layer_builds():
    """单层 LSTM 传 dropout 会被 PyTorch 警告，实现里必须条件性置 0。"""
    m = _build(num_layers=1)
    assert m.lstm.num_layers == 1


def test_two_layer_builds():
    m = _build(num_layers=2)
    assert m.lstm.num_layers == 2


def test_tokens_beyond_length_do_not_affect_output():
    """核心回归测试：真实长度之后的词位不得影响预测。

    LSTM 是单向的，output[t] 只依赖 inputs[0..t]；
    模型按 lengths-1 取输出，因此第 lengths 位之后的内容必须被忽略。

    这条测试正是课程实验四那个 bug 的守卫——
    取 output[-1] 时它必然失败，因为末位填充会进入结果。
    """
    m = _build().eval()
    lengths = torch.tensor([3])

    # 同一批真实词，但填充位填了不同的非零词
    ids_a = torch.tensor([[2, 3, 4, 7, 9]]).T   # (5, 1)
    ids_b = torch.tensor([[2, 3, 4, 0, 0]]).T

    with torch.no_grad():
        out_a = m(ids_a, lengths)
        out_b = m(ids_b, lengths)

    assert torch.allclose(out_a, out_b, atol=1e-6), (
        "真实长度之后的词位影响了输出，说明取的不是 lengths-1 位的隐藏状态"
    )


def test_padding_index_excluded_from_gradient():
    """<pad> 不参与梯度，否则填充位会学到语义并污染嵌入表。"""
    m = _build()
    assert m.embeddings.padding_idx == 0
    x = torch.tensor([[2, 3, 0, 0]]).T
    out = m(x, torch.tensor([2]))
    out.sum().backward()
    grad = m.embeddings.weight.grad
    assert grad is not None
    assert torch.all(grad[0] == 0), "padding_idx=0 的梯度应为 0"


def test_output_is_logits_not_probabilities():
    """返回的必须是 logits（供 CrossEntropyLoss 使用），不能提前做 softmax。"""
    m = _build().eval()
    x = torch.randint(2, 20, (4, 3))
    with torch.no_grad():
        out = m(x, torch.tensor([4, 4, 4]))
    row_sums = out.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), (
        "输出行和为 1，疑似已经 softmax；应返回 logits"
    )
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend; ..\.venv\Scripts\python.exe -m pytest tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'senti.model'`

- [ ] **Step 3: 实现 `model.py`**

`backend/senti/model.py`：
```python
# -*- coding: utf-8 -*-
"""模型层：LSTM 情感二分类模型。

结构沿用课程实验四的 Embedding + LSTM + Linear 三组件方案，
并保留其中一处关键修正（见 forward 的 docstring）。
"""

import torch
import torch.nn as nn

from senti import config


class SentimentLSTM(nn.Module):
    """Embedding + LSTM + Linear 三组件情感分类器。

        Embedding  词索引 → 连续向量（padding_idx=0，填充位不参与梯度更新）
        LSTM       向量序列 → 逐时刻的语义表示
        Linear     末时刻语义表示 → 2 维 logits（积极 / 消极）

    输出为 **logits** 而非概率：交叉熵损失内部自带 softmax，
    提前做 softmax 会造成数值不稳定与梯度衰减。
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        embedding_dim: int = config.EMBEDDING_DIM,
        dropout: float = config.DROPOUT,
        output_dim: int = config.OUTPUT_DIM,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=config.PAD_IDX)

        # num_layers == 1 时 PyTorch 会忽略 dropout 并给出警告，故显式置 0
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=False,     # 输入形状为 (seq_len, batch)
            dropout=lstm_dropout,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数：
            x       (seq_len, batch) 词索引矩阵
            lengths (batch,) 每条样本的**真实词数**（不含填充）

        返回：
            (batch, output_dim) logits
        """
        embeds = self.embeddings(x)                # (seq_len, batch, embedding_dim)
        output, _ = self.lstm(embeds)              # (seq_len, batch, hidden_dim)

        # 取每条样本「最后一个真实词」对应的隐藏状态。
        #
        # 不能写成 output[-1]：那固定取第 seq_len 步，而评论普遍短于 MAX_LEN，
        # 末位是 <pad> 填充。实测中该写法会让模型前 5 轮几乎不学习。
        # 正确做法是按真实长度 gather。
        idx = (
            (lengths - 1)
            .clamp(min=0)
            .view(1, -1, 1)
            .expand(1, output.size(1), output.size(2))
        )
        last = output.gather(0, idx).squeeze(0)    # (batch, hidden_dim)

        return self.fc(last)                       # (batch, output_dim)


def count_parameters(m: nn.Module) -> int:
    """可训练参数量，用于 meta.json 与前端展示。"""
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend; ..\.venv\Scripts\python.exe -m pytest tests/test_model.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add backend/senti/model.py backend/tests/test_model.py
git commit -m "feat(model): Embedding+LSTM+Linear 情感分类模型"
```

---

## Task 5: 训练层与网格实验

**Files:**
- Create: `backend/senti/train.py`
- Test: `backend/tests/test_train.py`

**Interfaces:**
- Consumes: `data.load_split`、`data.build_vocab`、`data.encode`、`data.save_vocab`、
  `model.SentimentLSTM`、`model.count_parameters`、全部 `config` 常量
- Produces:
  - `train.set_seed(seed: int = config.RANDOM_SEED) -> None`
  - `train.split_train_val(ids, lengths, labels, val_ratio, seed) -> tuple[DataLoader, DataLoader]`
  - `train.train_one(layers: int, hidden: int, quiet: bool = False) -> dict`
    （返回 `{model_id, layers, hidden_dim, embedding_dim, test_acc, val_acc, params, train_time_sec, vocab_size}`）
  - `train.overfit_test(n: int = 40, epochs: int = 30) -> float`
  - `train.run_grid() -> dict`（完整 meta 字典）
  - `train.main() -> None`（CLI 入口，支持 `--config LAYERS HIDDEN` / `--overfit-test`）

- [ ] **Step 1: 写失败测试**

`backend/tests/test_train.py`：
```python
# -*- coding: utf-8 -*-
"""训练层的轻量测试。

训练是数值实验，这里只测「可测的东西」：
工具函数的行为、划分的无泄漏性、以及小样本过拟合这一管线正确性判据。
完整网格训练不在单元测试中跑（分钟级，见 README 的实测流程）。
"""

import numpy as np
import torch

from senti import config, train


def test_set_seed_makes_initialization_reproducible():
    import torch.nn as nn

    train.set_seed(42)
    a = nn.Linear(4, 4).weight.detach().clone()
    train.set_seed(42)
    b = nn.Linear(4, 4).weight.detach().clone()
    assert torch.equal(a, b)


def test_split_train_val_has_no_overlap():
    """验证集必须完全来自训练集内部，且与训练集不重叠。"""
    n = 100
    ids = np.arange(n * 5).reshape(n, 5).astype(np.int64)
    lengths = np.full(n, 5, dtype=np.int64)
    labels = np.array([i % 2 for i in range(n)], dtype=np.int64)

    tr_loader, val_loader = train.split_train_val(ids, lengths, labels, val_ratio=0.2, seed=42)

    train_ids = set()
    for xb, lb, _ in tr_loader:
        for row in xb.numpy().tolist():
            train_ids.add(tuple(row))
    val_ids = set()
    for xb, lb, _ in val_loader:
        for row in xb.numpy().tolist():
            val_ids.add(tuple(row))

    assert len(val_ids) == 20
    assert train_ids & val_ids == set(), "训练集与验证集出现了重叠样本"


def test_overfit_test_can_reach_high_accuracy():
    """小样本过拟合是数据管线正确性的判据。

    若几十条样本都无法过拟合，说明词表、编码或「取真实末词」的逻辑有 bug。
    这是把「模型不学习」与「数据有错」两个症状分开的最快手段。
    """
    acc = train.overfit_test(n=40, epochs=30)
    assert acc >= 0.9, f"小样本过拟合仅 {acc:.2%}，管线可能有问题"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend; ..\.venv\Scripts\python.exe -m pytest tests/test_train.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'senti.train'`

- [ ] **Step 3: 实现 `train.py`**

`backend/senti/train.py`：
```python
# -*- coding: utf-8 -*-
"""训练层：单配置训练、网格实验与产物落盘。

集合用途（严格区分，不可混用）：
    训练集  更新参数
    验证集  选最优轮次（每轮评估，保留验证最优的权重）
    测试集  只在训练全部结束后报告一次最终指标

测试集不参与任何选择，否则报告的准确率不再是泛化能力的无偏估计。
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from senti import config, data, model


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """固定所有随机源，保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(
    ids: np.ndarray, lengths: np.ndarray, labels: np.ndarray,
    batch_size: int = config.BATCH_SIZE, shuffle: bool = False,
) -> DataLoader:
    """把 numpy 数组封装成 DataLoader。lengths 必须一并送入。"""
    ds = TensorDataset(
        torch.from_numpy(ids).long(),
        torch.from_numpy(lengths).long(),
        torch.from_numpy(labels).long(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(m: nn.Module, loader: DataLoader, dev: torch.device) -> float:
    """在给定数据集上计算准确率。"""
    m.eval()
    correct, total = 0, 0
    for xb, lb, yb in loader:
        x = xb.transpose(1, 0).contiguous().to(dev)
        pred = m(x, lb.to(dev)).argmax(dim=1).cpu()
        correct += (pred == yb).sum().item()
        total += len(yb)
    return correct / max(total, 1)


def split_train_val(
    ids: np.ndarray, lengths: np.ndarray, labels: np.ndarray,
    val_ratio: float = config.VAL_RATIO, seed: int = config.RANDOM_SEED,
) -> tuple[DataLoader, DataLoader]:
    """把训练集切成 (训练子集, 验证集)，按固定种子无重叠切分。"""
    n = len(labels)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_ratio))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    tr_loader = make_loader(ids[tr_idx], lengths[tr_idx], labels[tr_idx], shuffle=True)
    val_loader = make_loader(ids[val_idx], lengths[val_idx], labels[val_idx], shuffle=False)
    return tr_loader, val_loader


def train_one(layers: int, hidden: int, quiet: bool = False) -> dict:
    """训练单个配置，返回指标字典并保存 checkpoint。

    每轮在验证集上评估，**保留验证准确率最高的那轮权重**；
    训练结束后用该权重在测试集上评估一次，作为最终指标。
    """
    set_seed()
    dev = device()
    mid = config.model_id(layers, hidden)

    tr_texts, tr_labels, te_texts, te_labels = data.load_split()
    vocab = data.load_vocab()
    tr_ids, tr_len = data.encode(tr_texts, vocab)
    te_ids, te_len = data.encode(te_texts, vocab)

    tr_loader, val_loader = split_train_val(
        tr_ids, tr_len, np.array(tr_labels, dtype=np.int64)
    )
    te_loader = make_loader(
        te_ids, te_len, np.array(te_labels, dtype=np.int64), shuffle=False
    )

    m = model.SentimentLSTM(
        vocab_size=len(vocab) + config.FIRST_WORD_IDX, hidden_dim=hidden, num_layers=layers
    ).to(dev)
    optimizer = optim.Adam(m.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val, best_state = -1.0, None
    history = {"loss": [], "val_acc": []}   # 逐轮累积，供 visualize() 绘图
    t0 = time.time()

    for epoch in range(config.EPOCHS):
        m.train()
        loss_sum = 0.0
        for xb, lb, yb in tr_loader:
            x = xb.transpose(1, 0).contiguous().to(dev)
            y = yb.to(dev)
            optimizer.zero_grad()
            loss = criterion(m(x, lb.to(dev)), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        epoch_loss = loss_sum / len(tr_loader)
        val_acc = evaluate(m, val_loader, dev)
        history["loss"].append(epoch_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}

        if not quiet:
            print(f"  [{mid}] epoch {epoch + 1}/{config.EPOCHS} "
                  f"loss={epoch_loss:.4f} val_acc={val_acc:.4f}")

    # 用验证最优权重报告测试集指标
    assert best_state is not None
    m.load_state_dict(best_state)
    test_acc = evaluate(m, te_loader, dev)
    train_time = time.time() - t0

    # 训练曲线是报告第 5 章「结果分析」的素材，必须产出，故不随 quiet 关闭
    visualize(history, mid)

    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": mid,
            "layers": layers,
            "hidden_dim": hidden,
            "embedding_dim": config.EMBEDDING_DIM,
            "vocab_size": len(vocab) + config.FIRST_WORD_IDX,
            "max_len": config.MAX_LEN,
            "state_dict": best_state,
        },
        config.ckpt_path(mid),
    )

    metrics = {
        "model_id": mid,
        "layers": layers,
        "hidden_dim": hidden,
        "embedding_dim": config.EMBEDDING_DIM,
        "val_acc": round(best_val, 4),
        "test_acc": round(test_acc, 4),
        "params": model.count_parameters(m),
        "train_time_sec": round(train_time, 1),
        "vocab_size": len(vocab) + config.FIRST_WORD_IDX,
    }
    print(f"  [{mid}] 验证 {best_val:.4f} / 测试 {test_acc:.4f} / "
          f"参数 {metrics['params']:,} / 耗时 {train_time:.0f}s")
    return metrics


def overfit_test(n: int = 40, epochs: int = 30) -> float:
    """小样本过拟合测试——验证数据管线正确性的最快手段。

    取 n 条样本（不切验证集、不加 dropout），看训练准确率能否冲到接近 100%。
    管线正确时必然能过拟合；若不能，问题在词表/编码/取末词，而不在超参数。

    返回训练集最终准确率。
    """
    set_seed()
    dev = device()

    tr_texts, tr_labels, _, _ = data.load_split()
    vocab = data.build_vocab(tr_texts)
    ids, lengths = data.encode(tr_texts[:n], vocab)
    labels = np.array(tr_labels[:n], dtype=np.int64)

    loader = make_loader(ids, lengths, labels, batch_size=min(n, 16), shuffle=True)

    m = model.SentimentLSTM(
        vocab_size=len(vocab) + config.FIRST_WORD_IDX, hidden_dim=64, num_layers=1, dropout=0.0
    ).to(dev)
    optimizer = optim.Adam(m.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    acc = 0.0
    for epoch in range(epochs):
        m.train()
        for xb, lb, yb in loader:
            x = xb.transpose(1, 0).contiguous().to(dev)
            optimizer.zero_grad()
            criterion(m(x, lb.to(dev)), yb.to(dev)).backward()
            optimizer.step()
        acc = evaluate(m, loader, dev)
        print(f"  [过拟合测试] epoch {epoch + 1}/{epochs} acc={acc:.4f}")
    return acc


def visualize(history: dict, mid: str) -> Path:
    """绘制损失与准确率曲线。"""
    from common.plot_font import setup_chinese_font

    setup_chinese_font()
    import matplotlib.pyplot as plt

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history["loss"], "b-o", markersize=4)
    axes[0].set_title("训练损失曲线")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(epochs, np.array(history["val_acc"]) * 100, "r-o",
                 markersize=4, label="验证集")
    axes[1].set_title("验证准确率曲线")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy (%)")
    axes[1].legend(loc="best")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = config.OUTPUT_DIR / f"{mid}_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"训练曲线已保存：{path}")
    return path


def run_grid() -> dict:
    """遍历 GRID 训练全部配置，写出 meta.json，返回 meta 字典。"""
    tr_texts, tr_labels, te_texts, te_labels = data.load_split()
    vocab = data.build_vocab(tr_texts)
    data.save_vocab(vocab)

    results = []
    for layers, hidden in config.GRID:
        print(f"\n=== 训练 {config.model_id(layers, hidden)} ===")
        results.append(train_one(layers, hidden))

    meta = {
        "dataset": "ChnSentiCorp",
        "train_size": len(tr_texts),
        "test_size": len(te_texts),
        "vocab_size": len(vocab) + config.FIRST_WORD_IDX,
        "max_len": config.MAX_LEN,
        "models": results,
    }
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    config.META_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n元数据已保存：{config.META_PATH}")

    # 最优配置按**验证**准确率选取：测试集只负责报告，不参与任何选择。
    # 若按 test_acc 排序，报出的「最优」就是 6 个测试准确率的极大值，
    # 会被系统性高估（winner's curse），失去无偏性。
    best = max(results, key=lambda r: r["val_acc"])
    print(f"最优配置（按验证集选出）：{best['model_id']}  "
          f"验证 {best['val_acc']:.4f} / 测试 {best['test_acc']:.4f}")
    return meta


def write_meta_for_single(metrics: dict) -> None:
    """单配置训练后，把结果并入 meta.json（已存在同 id 则替换）。"""
    if config.META_PATH.exists():
        meta = json.loads(config.META_PATH.read_text(encoding="utf-8"))
    else:
        tr_texts, _, te_texts, _ = data.load_split()
        vocab = data.load_vocab()
        meta = {
            "dataset": "ChnSentiCorp",
            "train_size": len(tr_texts),
            "test_size": len(te_texts),
            "vocab_size": len(vocab) + config.FIRST_WORD_IDX,
            "max_len": config.MAX_LEN,
            "models": [],
        }
    meta["models"] = [m for m in meta["models"] if m["model_id"] != metrics["model_id"]]
    meta["models"].append(metrics)
    config.META_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"元数据已更新：{config.META_PATH}")


def build_vocab_once() -> None:
    """确保词表已落盘（单配置训练入口的前置步骤）。"""
    if config.VOCAB_PATH.exists():
        print(f"词表已存在：{config.VOCAB_PATH}")
        return
    tr_texts, _, _, _ = data.load_split()
    data.save_vocab(data.build_vocab(tr_texts))


def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM 情感分类模型训练")
    parser.add_argument("--config", nargs=2, type=int, metavar=("LAYERS", "HIDDEN"),
                        help="只训练单个配置，例如 --config 2 128")
    parser.add_argument("--overfit-test", action="store_true",
                        help="只跑小样本过拟合测试（验证数据管线）")
    args = parser.parse_args()

    if args.overfit_test:
        acc = overfit_test()
        print(f"\n过拟合测试结果：{acc:.4f}")
        print("判定：通过" if acc >= 0.9 else "判定：不通过 —— 检查词表/编码/取末词逻辑")
        return

    build_vocab_once()

    if args.config:
        layers, hidden = args.config
        if (layers, hidden) not in config.GRID:
            raise SystemExit(f"配置 {layers}/{hidden} 不在 GRID 中：{config.GRID}")
        write_meta_for_single(train_one(layers, hidden))
        return

    run_grid()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 先跑过拟合测试（数据管线体检）**

Run: `cd backend; ..\.venv\Scripts\python.exe -m senti.train --overfit-test`
Expected: 末行 `判定：通过`（train acc 接近 1.0）

**这一步不通就不要往下走**——过拟合都做不到，说明词表/编码/取末词有问题，
继续训网格只是浪费几十分钟。

- [ ] **Step 5: 运行测试确认通过**

Run: `cd backend; ..\.venv\Scripts\python.exe -m pytest tests/test_train.py -v`
Expected: 3 passed

> `test_overfit_test_can_reach_high_accuracy` 会实际训练 30 轮（几十条样本，秒级）。
> 若失败，先单独跑 `--overfit-test` 看每轮准确率曲线判断是欠拟合还是管线错误。

- [ ] **Step 6: 训练全部 6 组并生成 meta.json**

Run: `cd backend; ..\.venv\Scripts\python.exe -m senti.train`
Expected: 依次训练 6 组，打印各自「验证 / 测试 / 参数 / 耗时」，最后打印最优配置；
`backend/artifacts/` 下出现 6 个 `.pt` + `vocab.json` + `meta.json`

- [ ] **Step 7: 核对达标情况**

Run:
```powershell
cd backend
..\.venv\Scripts\python.exe -c @"
import json
from senti import config
meta = json.loads(config.META_PATH.read_text(encoding='utf-8'))
print(f\"数据集 {meta['dataset']}  训练 {meta['train_size']} / 测试 {meta['test_size']}\")
print(f\"词表 {meta['vocab_size']}  MAX_LEN {meta['max_len']}\")
for m in sorted(meta['models'], key=lambda r: -r['val_acc']):
    print(f\"{m['model_id']:<14} 验证 {m['val_acc']:.4f}  测试 {m['test_acc']:.4f}  参数量 {m['params']:>8,}  耗时 {m['train_time_sec']:>6.0f}s\")
best = max(meta['models'], key=lambda r: r['val_acc'])
print(f\"按验证集选出的最优：{best['model_id']}\")
print('达标' if best['test_acc'] >= 0.85 else '未达标（门槛 0.85）')
"@
```
Expected: 末行 `达标`

**若未达标**，按 spec 第 11 节的对策依次尝试，每改一项重跑一次：
① 调大 `config.MAX_LEN`（看 Task 3 Step 5 记录的词数中位数）；
② 增加 `EPOCHS` 或调整 `LEARNING_RATE`；
③ 引入预训练词向量。每次改动后重跑本步骤核对。

- [ ] **Step 8: Commit**

```bash
git add backend/senti/train.py backend/tests/test_train.py
git commit -m "feat(train): 训练循环、网格实验与产物落盘"
```

---

## Task 6: 推理层（含逐词贡献）

**Files:**
- Create: `backend/senti/infer.py`
- Test: `backend/tests/test_infer.py`

**Interfaces:**
- Consumes: `data.load_vocab`、`data.tokenize`、`data.encode`、`model.SentimentLSTM`、
  `config.ckpt_path`、`config.META_PATH`、`config.CONTRIB_MAX_TOKENS`
- Produces:
  - `infer.list_models() -> list[dict]`（读 `meta.json`）
  - `infer.Predictor(model_id: str)`，方法：
    - `predict(text: str, max_len: int = config.MAX_LEN) -> dict`
      → `{label, prob_pos, prob_neg, tokens}`
    - `batch_predict(texts: list[str], max_len: int = config.MAX_LEN) -> list[dict]`
      → `[{text, prob_pos, prob_neg}]`
    - `explain(text: str, max_len: int = config.MAX_LEN) -> dict`
      → `{label, prob_pos, prob_neg, tokens, contributions: [{token, delta}]}`
  - `infer.get_predictor(model_id: str) -> Predictor`（带缓存）
  - `infer.DEFAULT_THRESHOLD = 0.5`

- [ ] **Step 1: 写失败测试**

`backend/tests/test_infer.py`：
```python
# -*- coding: utf-8 -*-
"""推理层测试。依赖已训练好的 checkpoint（Task 5 的产物）。"""

import pytest

from senti import config, infer


@pytest.fixture(scope="module")
def predictor():
    models = infer.list_models()
    if not models:
        pytest.skip("尚无 checkpoint，先运行：python -m senti.train")
    best = max(models, key=lambda m: m["val_acc"])
    return infer.get_predictor(best["id"])


def test_list_models_reads_meta():
    models = infer.list_models()
    for m in models:
        assert {"id", "layers", "hidden_dim", "test_acc", "params"} <= set(m)


def test_predict_probabilities_sum_to_one(predictor):
    r = predictor.predict("这个酒店非常好，服务很周到")
    assert abs(r["prob_pos"] + r["prob_neg"] - 1.0) < 1e-6


def test_predict_returns_label_and_tokens(predictor):
    r = predictor.predict("房间很干净")
    assert r["label"] in (0, 1)
    assert isinstance(r["tokens"], list) and r["tokens"]


def test_predict_positive_and_negative_differ(predictor):
    """明显褒贬的两句话应给出不同倾向，否则模型实际没学到极性。"""
    good = predictor.predict("非常好，非常满意，强烈推荐")
    bad = predictor.predict("太差了，非常失望，再也不来了")
    assert good["prob_pos"] > bad["prob_pos"]


def test_batch_predict_length_matches_input(predictor):
    texts = ["很好", "很差", "一般"]
    out = predictor.batch_predict(texts)
    assert len(out) == len(texts)
    assert [o["text"] for o in out] == texts


def test_batch_matches_single(predictor):
    """批量与单条结果必须一致，否则说明批处理路径有 bug。"""
    texts = ["服务很好", "房间很脏"]
    batch = predictor.batch_predict(texts)
    for item in batch:
        single = predictor.predict(item["text"])
        assert abs(single["prob_pos"] - item["prob_pos"]) < 1e-4


def test_explain_returns_contributions_for_each_token(predictor):
    r = predictor.explain("这个酒店非常好")
    assert r["contributions"]
    n_tokens = len([t for t in predictor.predict("这个酒店非常好")["tokens"]])
    assert len(r["contributions"]) == n_tokens


def test_explain_direction_is_correct(predictor):
    """删掉强正向词后，积极概率应当下降 → delta < 0。

    delta 定义为「删词后的积极概率 − 原始积极概率」，
    故正向词 delta < 0、负向词 delta > 0。
    """
    r = predictor.explain("非常好 非常好 非常好 酒店 位置 很 差")
    deltas = {c["token"]: c["delta"] for c in r["contributions"]}
    assert deltas.get("差", 0) > deltas.get("非常好", 0), (
        "负向词的 delta 应大于正向词"
    )


def test_explain_respects_token_limit(predictor):
    long_text = "很好 " * 300
    r = predictor.explain(long_text)
    assert len(r["contributions"]) <= config.CONTRIB_MAX_TOKENS


def test_unknown_model_raises(predictor):
    with pytest.raises((FileNotFoundError, ValueError)):
        infer.Predictor("不存在的模型")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend; ..\.venv\Scripts\python.exe -m pytest tests/test_infer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'senti.infer'`

- [ ] **Step 3: 实现 `infer.py`**

`backend/senti/infer.py`：
```python
# -*- coding: utf-8 -*-
"""推理层：单条预测、批量预测与逐词贡献分析。

只消费训练产物（checkpoint + 词表 + 元数据），不做任何训练。
「模型可切换」在本层等价于「换一个 checkpoint 文件」。
"""

import json

import numpy as np
import torch

from senti import config, data, model, train

# 后端给出 label 时使用的默认阈值。
# 注意：阈值是「概率 → 标签」的纯映射，最终判定由前端完成，
# 本字段只作参考，不参与任何计算。
DEFAULT_THRESHOLD = 0.5


def list_models() -> list[dict]:
    """读取 meta.json，返回可用模型列表（按 test_acc 降序）。"""
    if not config.META_PATH.exists():
        return []
    meta = json.loads(config.META_PATH.read_text(encoding="utf-8"))
    models = [
        {
            "id": m["model_id"],
            "layers": m["layers"],
            "hidden_dim": m["hidden_dim"],
            "embedding_dim": m["embedding_dim"],
            "test_acc": m["test_acc"],
            "val_acc": m.get("val_acc", 0.0),
            "params": m["params"],
            "train_time_sec": m["train_time_sec"],
        }
        for m in meta.get("models", [])
    ]
    # 按**验证**准确率降序：这是「选择」，不是「报告」。
    # 测试准确率仍在响应里逐条返回，但不参与排序——
    # 否则前端默认选中的就是 6 个测试准确率的极大值（winner's curse）。
    return sorted(models, key=lambda m: -m["val_acc"])


class Predictor:
    """持有「模型 + 词表」配对，提供推理能力。

    词表与 checkpoint 必须同源：载入时断言 vocab_size 一致，
    否则训练期与推理期用了不同词表，准确率会静默下降而不报错。
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.device = train.device()

        path = config.ckpt_path(model_id)
        if not path.exists():
            raise FileNotFoundError(
                f"找不到模型 {model_id} 的 checkpoint：{path}\n"
                f"可用模型：{[m['id'] for m in list_models()]}"
            )

        self.vocab = data.load_vocab()
        # weights_only=True：checkpoint 只含 str/int/张量字典，无需反序列化任意对象。
        # torch 2.6 起这已是默认值，显式写出以防将来被改回。
        ckpt = torch.load(path, map_location=self.device, weights_only=True)

        expected = len(self.vocab) + 2
        if ckpt["vocab_size"] != expected:
            raise ValueError(
                f"词表与模型不匹配：checkpoint 记录 vocab_size={ckpt['vocab_size']}，"
                f"当前词表 {expected}。两者必须同源，请重新训练。"
            )

        self.model = model.SentimentLSTM(
            vocab_size=ckpt["vocab_size"],
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["layers"],
            embedding_dim=ckpt["embedding_dim"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.max_len = ckpt.get("max_len", config.MAX_LEN)

    # ------------------------------------------------------------ 内部

    @torch.no_grad()
    def _probs_for_tokens(self, tokens: list[str], max_len: int) -> float:
        """给定已分好的词列表，返回积极类概率。"""
        ids = np.zeros((1, max_len), dtype=np.int64)
        seq = [self.vocab.get(w, config.UNK_IDX) for w in tokens][:max_len]
        ids[0, : len(seq)] = seq
        lengths = np.array([max(len(seq), 1)], dtype=np.int64)

        x = torch.from_numpy(ids).long().transpose(1, 0).contiguous().to(self.device)
        logits = self.model(x, torch.from_numpy(lengths).long().to(self.device))
        return torch.softmax(logits, dim=1)[0, 1].item()

    @torch.no_grad()
    def _probs_for_batch(self, texts: list[str], max_len: int) -> np.ndarray:
        """批量推理，返回积极类概率数组。"""
        ids, lengths = data.encode(texts, self.vocab, max_len=max_len)
        x = torch.from_numpy(ids).long().transpose(1, 0).contiguous().to(self.device)
        logits = self.model(x, torch.from_numpy(lengths).long().to(self.device))
        return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    # ------------------------------------------------------------ 对外

    def predict(self, text: str, max_len: int | None = None) -> dict:
        """单条预测。返回 {label, prob_pos, prob_neg, tokens}。"""
        max_len = max_len or self.max_len
        tokens = data.tokenize(text)[:max_len]
        p = self._probs_for_tokens(tokens, max_len)
        return {
            "label": 1 if p >= DEFAULT_THRESHOLD else 0,
            "prob_pos": p,
            "prob_neg": 1.0 - p,
            "tokens": tokens,
        }

    def batch_predict(self, texts: list[str], max_len: int | None = None) -> list[dict]:
        """批量预测。返回 [{text, prob_pos, prob_neg}]，顺序与输入一致。"""
        max_len = max_len or self.max_len
        # 按 batch_size 分批，避免超长输入一次性占满显存
        probs: list[float] = []
        for i in range(0, len(texts), config.BATCH_SIZE):
            chunk = texts[i : i + config.BATCH_SIZE]
            probs.extend(self._probs_for_batch(chunk, max_len).tolist())
        return [
            {"text": t, "prob_pos": p, "prob_neg": 1.0 - p}
            for t, p in zip(texts, probs)
        ]

    def explain(self, text: str, max_len: int | None = None) -> dict:
        """逐词贡献分析（遮挡法）。

        对每个词位，把它替换为 <unk> 后重新前向，记录积极概率的变化：

            delta = 删词后的积极概率 − 原始积极概率

            delta < 0  删掉后积极概率下降 → 该词起**积极**作用
            delta > 0  删掉后积极概率上升 → 该词起**消极**作用

        用遮挡法而非 attention 权重：注意力反映信息的路由，不等于重要性。
        遮挡法问的是反事实问题「缺了这个词结果会变吗」，对任意模型成立。

        复杂度为 O(N) 次前向；参与分析的词数上限为 CONTRIB_MAX_TOKENS。
        """
        max_len = max_len or self.max_len
        tokens = data.tokenize(text)[: config.CONTRIB_MAX_TOKENS]
        if not tokens:
            return {"label": 0, "prob_pos": 0.5, "prob_neg": 0.5, "contributions": []}

        base = self._probs_for_tokens(tokens, max_len)

        contributions = []
        for i in range(n):
            masked_ids = base_ids.copy()
            masked_ids[0, i] = config.UNK_IDX
            after = self._probs_for_ids(masked_ids, lengths)
            contributions.append({"token": all_tokens[i], "delta": after - base})

        # 不排序：保持原句词序，与 predict 的 tokens 同序。
        # 前端要同时渲染「原句高亮」与「Top-N 排序表」——
        # 排序客户端零成本可做，而从乱序还原原句序不可能（jieba 在服务端）。
        #
        # 同时返回**完整** tokens（与 predict 一致）：contributions 只覆盖前
        # CONTRIB_MAX_TOKENS 个位置，前端若拿它当词序列渲染，长文本会只高亮前 N 词、
        # 尾部整段丢失。故 tokens 作渲染骨架，contributions 仅作 delta 查表。
        return {
            "label": 1 if base >= DEFAULT_THRESHOLD else 0,
            "prob_pos": base,
            "prob_neg": 1.0 - base,
            "tokens": all_tokens,
            "contributions": contributions,
        }


_PREDICTOR_CACHE: dict[str, Predictor] = {}


def get_predictor(model_id: str) -> Predictor:
    """按 id 取 Predictor，带进程内缓存。

    载入模型要读盘并重建网络，缓存避免每次请求都重载。
    """
    if model_id not in _PREDICTOR_CACHE:
        _PREDICTOR_CACHE[model_id] = Predictor(model_id)
    return _PREDICTOR_CACHE[model_id]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend; ..\.venv\Scripts\python.exe -m pytest tests/test_infer.py -v`
Expected: 10 passed

> `test_explain_direction_is_correct` 是最关键的一条。若失败，先确认 delta 的符号约定
> （`after - base`），再用
> `..\.venv\Scripts\python.exe -c "from senti import infer; print(infer.get_predictor('lstm_l2_h128').explain('非常好 非常差'))"`
> 看实际输出。

- [ ] **Step 5: 人工检视一次解释输出**

Run:
```powershell
cd backend
..\.venv\Scripts\python.exe -c @"
from senti import infer
p = infer.get_predictor(max(infer.list_models(), key=lambda m: m['val_acc'])['id'])
r = p.explain('这家酒店位置很好，离地铁站近，但房间隔音太差，晚上吵得睡不着')
print(f\"积极概率 {r['prob_pos']:.4f}\")
for c in r['contributions'][:10]:
    tag = '消极' if c['delta'] > 0 else '积极'
    print(f\"  {c['token']:<6} delta={c['delta']:+.4f}  ({tag}贡献)\")
"@
```
Expected: 排出贡献最大的词，且「很好/位置」等正向词 delta 为负、「差/吵」等负向词 delta 为正。
**若排序明显不合理，说明模型没学到极性，回到 Task 5 Step 7 排查。**

- [ ] **Step 6: Commit**

```bash
git add backend/senti/infer.py backend/tests/test_infer.py
git commit -m "feat(infer): 单条/批量预测与遮挡法逐词贡献"
```

---

## Task 7: Flask 接口层

**Files:**
- Create: `backend/senti/api.py`
- Create: `backend/run_api.py`
- Test: `backend/tests/test_api.py`

**Interfaces:**
- Consumes: `infer.list_models`、`infer.get_predictor`
- Produces: `api.create_app() -> Flask`，路由见下表

| 方法 | 路径 | 请求 | 响应 |
|------|------|------|------|
| GET | `/api/health` | — | `{"status": "ok"}` |
| GET | `/api/models` | — | `[ModelInfo]` |
| POST | `/api/predict` | `{text, model_id, max_len?}` | `{label, prob_pos, prob_neg, tokens}` |
| POST | `/api/explain` | `{text, model_id, max_len?}` | `{label, prob_pos, prob_neg, tokens, contributions}` |
| POST | `/api/batch` | `{texts, model_id, max_len?}` | `[{text, prob_pos, prob_neg}]` |

错误响应统一为 `{"error": "<中文说明>"}` + 400/404/500。

- [ ] **Step 1: 写失败测试**

`backend/tests/test_api.py`：
```python
# -*- coding: utf-8 -*-
"""接口层测试。用 Flask test client，无需起服务。"""

import pytest

from senti import api, infer


@pytest.fixture(scope="module")
def client():
    models = infer.list_models()
    if not models:
        pytest.skip("尚无 checkpoint，先运行：python -m senti.train")
    app = api.create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(scope="module")
def best_model_id():
    return max(infer.list_models(), key=lambda m: m["val_acc"])["id"]


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_models_list(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    data = r.get_json()
    assert isinstance(data, list) and data
    assert {"id", "layers", "hidden_dim", "test_acc", "params"} <= set(data[0])


def test_predict_ok(client, best_model_id):
    r = client.post("/api/predict", json={"text": "这个酒店非常好", "model_id": best_model_id})
    assert r.status_code == 200
    d = r.get_json()
    assert abs(d["prob_pos"] + d["prob_neg"] - 1.0) < 1e-6
    assert d["label"] in (0, 1)
    assert isinstance(d["tokens"], list)


def test_predict_missing_text_returns_400(client, best_model_id):
    r = client.post("/api/predict", json={"model_id": best_model_id})
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_predict_empty_text_returns_400(client, best_model_id):
    r = client.post("/api/predict", json={"text": "   ", "model_id": best_model_id})
    assert r.status_code == 400


def test_predict_too_long_returns_400(client, best_model_id):
    from senti import config
    r = client.post("/api/predict",
                    json={"text": "好" * (config.MAX_TEXT_CHARS + 1), "model_id": best_model_id})
    assert r.status_code == 400


def test_predict_unknown_model_returns_404(client):
    r = client.post("/api/predict", json={"text": "很好", "model_id": "不存在"})
    assert r.status_code == 404


def test_explain_ok(client, best_model_id):
    r = client.post("/api/explain", json={"text": "房间很干净，服务也好", "model_id": best_model_id})
    assert r.status_code == 200
    d = r.get_json()
    assert d["contributions"]
    assert set(d["contributions"][0]) == {"token", "delta"}


def test_batch_ok(client, best_model_id):
    texts = ["很好", "很差", "一般般"]
    r = client.post("/api/batch", json={"texts": texts, "model_id": best_model_id})
    assert r.status_code == 200
    d = r.get_json()
    assert len(d) == len(texts)
    assert [x["text"] for x in d] == texts


def test_batch_rejects_non_list(client, best_model_id):
    r = client.post("/api/batch", json={"texts": "很好", "model_id": best_model_id})
    assert r.status_code == 400


def test_batch_rejects_empty_list(client, best_model_id):
    r = client.post("/api/batch", json={"texts": [], "model_id": best_model_id})
    assert r.status_code == 400


def test_batch_rejects_oversized_payload(client, best_model_id):
    """单次批量上限，防止一次请求打爆显存。"""
    r = client.post("/api/batch", json={"texts": ["好"] * 1001, "model_id": best_model_id})
    assert r.status_code == 400


def test_max_len_param_is_honored(client, best_model_id):
    """max_len 应影响分词截断长度。"""
    long_text = "很好 " * 50
    a = client.post("/api/predict",
                    json={"text": long_text, "model_id": best_model_id, "max_len": 5}).get_json()
    assert len(a["tokens"]) <= 5
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend; ..\.venv\Scripts\python.exe -m pytest tests/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'senti.api'`

- [ ] **Step 3: 实现 `api.py`**

`backend/senti/api.py`：
```python
# -*- coding: utf-8 -*-
"""接口层：Flask 路由。

薄层——只做参数校验、调用推理层、序列化响应，不含任何业务逻辑。
"""

from flask import Flask, jsonify, request
from flask_cors import CORS

from senti import config, infer

# 单次批量请求的条数上限，防止一次请求打爆显存
MAX_BATCH_SIZE = 1000


def _error(message: str, status: int):
    return jsonify({"error": message}), status


def _validate_text(payload: dict) -> tuple[str | None, tuple | None]:
    """校验 text 与 model_id，返回 (text, None) 或 (None, 错误响应)。"""
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, _error("text 必须是非空字符串", 400)
    if len(text) > config.MAX_TEXT_CHARS:
        return None, _error(f"text 超长（上限 {config.MAX_TEXT_CHARS} 字符）", 400)
    return text, None


def _validate_model(payload: dict) -> tuple[str | None, tuple | None]:
    model_id = payload.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        return None, _error("model_id 必须是非空字符串", 400)
    available = {m["id"] for m in infer.list_models()}
    if model_id not in available:
        return None, _error(f"模型不存在：{model_id}；可用模型：{sorted(available)}", 404)
    return model_id, None


def _max_len(payload: dict) -> int:
    """解析可选的 max_len，非法值回退到配置默认值。"""
    value = payload.get("max_len")
    if isinstance(value, int) and 1 <= value <= 1000:
        return value
    return config.MAX_LEN


def create_app() -> Flask:
    app = Flask(__name__)
    # 开发期前端经 Vite 代理访问，本不需要 CORS；
    # 保留它以便直接用浏览器或 curl 调试。
    CORS(app)

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/models")
    def models():
        return jsonify(infer.list_models())

    @app.post("/api/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        text, err = _validate_text(payload)
        if err:
            return err
        model_id, err = _validate_model(payload)
        if err:
            return err
        try:
            result = infer.get_predictor(model_id).predict(text, _max_len(payload))
        except Exception as exc:  # noqa: BLE001 —— 统一转成 500，避免泄漏栈信息给前端
            return _error(f"推理失败：{type(exc).__name__}", 500)
        return jsonify(result)

    @app.post("/api/explain")
    def explain():
        payload = request.get_json(silent=True) or {}
        text, err = _validate_text(payload)
        if err:
            return err
        model_id, err = _validate_model(payload)
        if err:
            return err
        try:
            result = infer.get_predictor(model_id).explain(text, _max_len(payload))
        except Exception as exc:  # noqa: BLE001
            return _error(f"推理失败：{type(exc).__name__}", 500)
        return jsonify(result)

    @app.post("/api/batch")
    def batch():
        payload = request.get_json(silent=True) or {}
        texts = payload.get("texts")
        if not isinstance(texts, list):
            return _error("texts 必须是字符串数组", 400)
        if not texts:
            return _error("texts 不能为空", 400)
        if len(texts) > MAX_BATCH_SIZE:
            return _error(f"单次批量上限 {MAX_BATCH_SIZE} 条，当前 {len(texts)} 条", 400)
        if not all(isinstance(t, str) and t.strip() for t in texts):
            return _error("texts 中每一条都必须是非空字符串", 400)
        if any(len(t) > config.MAX_TEXT_CHARS for t in texts):
            return _error(f"存在超长文本（单条上限 {config.MAX_TEXT_CHARS} 字符）", 400)

        model_id, err = _validate_model(payload)
        if err:
            return err
        try:
            result = infer.get_predictor(model_id).batch_predict(texts, _max_len(payload))
        except Exception as exc:  # noqa: BLE001
            return _error(f"推理失败：{type(exc).__name__}", 500)
        return jsonify(result)

    return app
```

`backend/run_api.py`：
```python
# -*- coding: utf-8 -*-
"""后端启动入口。

运行：..\\.venv\\Scripts\\python.exe run_api.py
"""

from senti.api import create_app

if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend; ..\.venv\Scripts\python.exe -m pytest tests/test_api.py -v`
Expected: 14 passed

- [ ] **Step 5: 起服务并用 curl 冒烟**

Run（另开一个终端）:
```powershell
cd backend
..\.venv\Scripts\python.exe run_api.py
```
然后：
```powershell
curl.exe -s http://127.0.0.1:5000/api/health
curl.exe -s http://127.0.0.1:5000/api/models
curl.exe -s -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" -d "{\"text\":\"这个酒店非常好\",\"model_id\":\"lstm_l2_h128\"}"
```
Expected: 依次返回 `{"status":"ok"}`、模型数组、含 `prob_pos` 的预测结果

> 若 `lstm_l2_h128` 不在模型列表中，换成 `/api/models` 返回的第一个 id。

- [ ] **Step 6: 全量后端测试**

Run: `cd backend; ..\.venv\Scripts\python.exe -m pytest -v`
Expected: 全部通过（Task 1~7 的全部用例）

- [ ] **Step 7: Commit**

```bash
git add backend/senti/api.py backend/run_api.py backend/tests/test_api.py
git commit -m "feat(api): Flask 接口层与启动入口"
```

---

# Phase 2：前端

**前端验证方式说明**：spec 的验证策略未包含前端单元测试，本阶段以
**TypeScript 类型检查 + 生产构建成功 + 浏览器实际走通**为验证判据。
每个任务的「验证」步骤用 `pnpm build`（含 `tsc`）与浏览器操作，
不引入 vitest（属未在 spec 中的范围扩大）。

## Task 8: 前端脚手架与开发代理

**Files:**
- Create: `frontend/`（Vite 脚手架）
- Modify: `frontend/vite.config.ts`
- Create: `frontend/biome.json`
- Modify: `frontend/src/index.css`
- Modify: `frontend/package.json`（scripts）

**Interfaces:**
- Consumes: 无
- Produces: 可运行的 Vite 开发服务器（默认 `http://127.0.0.1:5173`），
  且 `/api/*` 请求被代理到 `http://127.0.0.1:5000`

- [ ] **Step 1: 生成 Vite 项目**

在**项目根目录**执行：
```powershell
pnpm create vite frontend --template react-ts
cd frontend
pnpm install
pnpm add -D tailwindcss @tailwindcss/vite @biomejs/biome
pnpm add papaparse
pnpm add -D @types/papaparse
```

> Tailwind 使用 **v4**：通过 `@tailwindcss/vite` 插件接入，**不需要** `tailwind.config.js`
> （v4 改为 CSS 优先配置）。spec 目录结构中提到的 `tailwind.config.js` 因此不创建。

- [ ] **Step 2: 配置 Vite（代理 + Tailwind 插件）**

`frontend/vite.config.ts`：
```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // 绑定 IPv4 回环：Vite 默认 host="localhost"，本机 localhost 先解析到 ::1，
    // 于是只监听 [::1]:5173，而文档与浏览器里写的是 127.0.0.1 —— 那种地址会连不上。
    host: "127.0.0.1",
    // 开发期把 /api 代理到 Flask，前端一律用相对路径。
    // 这样开发期同源、完全不需要 CORS 配置。
    proxy: {
      "/api": {
        target: "http://127.0.0.1:5000",
        changeOrigin: true,
      },
    },
  },
});
```

- [ ] **Step 3: 配置 Tailwind 与样式入口**

`frontend/src/index.css`（整体替换为）：
```css
@import "tailwindcss";

/* 逐词贡献高亮用到的三个色阶变量，供 TokenHighlight 组件引用 */
:root {
  --contrib-pos: oklch(0.72 0.17 150);   /* 积极贡献：绿 */
  --contrib-neg: oklch(0.66 0.2 25);     /* 消极贡献：红 */
  --contrib-neutral: oklch(0.92 0 0);    /* 无明显贡献：灰 */
}
```

- [ ] **Step 4: 配置 Biome**

`frontend/biome.json`：
```json
{
  "$schema": "https://biomejs.dev/schemas/2.0.0/schema.json",
  "files": { "includes": ["src/**/*.ts", "src/**/*.tsx"] },
  "formatter": { "indentStyle": "space", "indentWidth": 2, "lineWidth": 100 },
  "linter": {
    "enabled": true,
    "rules": { "recommended": true }
  }
}
```

在 `frontend/package.json` 的 `scripts` 中加入：
```json
{
  "lint": "biome check src",
  "lint:fix": "biome check --write src",
  "typecheck": "tsc -b --noEmit",
  "dev": "vite",
  "build": "tsc -b && vite build",
  "preview": "vite preview"
}
```

- [ ] **Step 5: 验证构建与类型检查通过**

Run:
```powershell
cd frontend
pnpm typecheck
pnpm build
```
Expected: 均无错误；生成 `frontend/dist/`

- [ ] **Step 6: 验证代理连通**

保证后端已启动（Task 7 Step 5），然后：
```powershell
cd frontend
pnpm dev
```
浏览器访问 `http://127.0.0.1:5173/api/health`。
Expected: 页面显示 `{"status":"ok"}` —— 说明代理生效

- [ ] **Step 7: Commit**

```bash
git add frontend/
git commit -m "chore(frontend): Vite + React + Tailwind v4 脚手架与 API 代理"
```

---

## Task 9: 类型定义与 API 客户端

**Files:**
- Create: `frontend/src/types.ts`
- Create: `frontend/src/api/client.ts`
- Create: `frontend/src/hooks/useModels.ts`

**Interfaces:**
- Consumes: 后端契约（Task 7）
- Produces:
  - `types.ModelInfo` / `PredictResponse` / `ExplainResponse` / `Contrib` / `BatchItem`
  - `types.DEFAULT_THRESHOLD = 0.5`、`types.MIN_THRESHOLD = 0.05`、`types.MAX_THRESHOLD = 0.95`
  - `types.labelOf(probPos: number, threshold: number): "积极" | "消极"`
  - `client.fetchModels()` / `fetchPredict()` / `fetchExplain()` / `fetchBatch()` / `fetchHealth()`
  - `hooks.useModels()` → `{ models, loading, error, reload }`

- [ ] **Step 1: 定义类型**

`frontend/src/types.ts`：
```ts
/** 与后端 API 契约一一对应的类型定义。改后端契约必须同步改这里。 */

export interface ModelInfo {
  id: string;
  layers: number;
  hidden_dim: number;
  embedding_dim: number;
  test_acc: number;
  val_acc: number;
  params: number;
  train_time_sec: number;
}

export interface PredictResponse {
  label: number;
  prob_pos: number;
  prob_neg: number;
  tokens: string[];
}

export interface Contrib {
  /** 词 */
  token: string;
  /** 删词后的积极概率 − 原始积极概率。
   *  负值 = 该词起积极作用；正值 = 起消极作用。 */
  delta: number;
}

export interface ExplainResponse {
  label: number;
  prob_pos: number;
  prob_neg: number;
  /** 完整分词结果，与 predict 返回的 tokens 相同——渲染骨架用这个。
   *  contributions 只覆盖前 CONTRIB_MAX_TOKENS 个位置，不能当词序列用。 */
  tokens: string[];
  contributions: Contrib[];
}

export interface BatchItem {
  text: string;
  prob_pos: number;
  prob_neg: number;
}

/** 判定阈值：把「积极概率」映射成标签的临界值。
 *
 *  阈值是纯前端参数——后端只返回概率，拖动滑块无需重新请求。 */
export const DEFAULT_THRESHOLD = 0.5;
export const MIN_THRESHOLD = 0.05;
export const MAX_THRESHOLD = 0.95;

export type SentimentLabel = "积极" | "消极";

/** 由概率与阈值导出标签。全应用只有这一处定义标签判定规则。 */
export function labelOf(probPos: number, threshold: number = DEFAULT_THRESHOLD): SentimentLabel {
  return probPos >= threshold ? "积极" : "消极";
}

/** 单条批量输入的最大条数，与后端 MAX_BATCH_SIZE 保持一致。 */
export const MAX_BATCH_SIZE = 1000;
```

- [ ] **Step 2: 实现 API 客户端**

`frontend/src/api/client.ts`：
```ts
/** 唯一进行网络请求的模块。组件一律通过这里访问后端。 */

import type {
  BatchItem,
  ExplainResponse,
  ModelInfo,
  PredictResponse,
} from "../types";

/** 相对路径，由 Vite 代理转发到 Flask（见 vite.config.ts）。 */
const BASE = "/api";

/** 后端错误响应统一为 { error: string }。 */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });

  if (!res.ok) {
    let message = `请求失败（HTTP ${res.status}）`;
    try {
      const body = (await res.json()) as { error?: string };
      if (body.error) message = body.error;
    } catch {
      // 响应不是 JSON，沿用默认提示
    }
    throw new Error(message);
  }
  return (await res.json()) as T;
}

export function fetchHealth(): Promise<{ status: string }> {
  return request("/health");
}

export function fetchModels(): Promise<ModelInfo[]> {
  return request("/models");
}

export function fetchPredict(text: string, modelId: string, maxLen?: number): Promise<PredictResponse> {
  return request("/predict", {
    method: "POST",
    body: JSON.stringify({ text, model_id: modelId, max_len: maxLen }),
  });
}

export function fetchExplain(text: string, modelId: string, maxLen?: number): Promise<ExplainResponse> {
  return request("/explain", {
    method: "POST",
    body: JSON.stringify({ text, model_id: modelId, max_len: maxLen }),
  });
}

export function fetchBatch(texts: string[], modelId: string, maxLen?: number): Promise<BatchItem[]> {
  return request("/batch", {
    method: "POST",
    body: JSON.stringify({ texts, model_id: modelId, max_len: maxLen }),
  });
}
```

- [ ] **Step 3: 实现模型列表 hook**

`frontend/src/hooks/useModels.ts`：
```ts
import { useCallback, useEffect, useState } from "react";
import { fetchModels } from "../api/client";
import type { ModelInfo } from "../types";

/** 拉取可用模型列表，并按**验证**准确率降序排列。
 *
 *  用验证集而非测试集排序：排序即「选择」，而测试集只负责报告。
 *  若按测试准确率排序，默认选中的就是 6 个测试准确率的极大值，会被高估。 */
export function useModels() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const list = await fetchModels();
      setModels([...list].sort((a, b) => b.val_acc - a.val_acc));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  return { models, loading, error, reload };
}
```

- [ ] **Step 4: 验证类型检查通过**

Run: `cd frontend; pnpm typecheck`
Expected: 无错误

- [ ] **Step 5: Commit**

```bash
git add frontend/src/types.ts frontend/src/api/ frontend/src/hooks/
git commit -m "feat(frontend): API 类型定义与客户端"
```

---

## Task 10: 模型选择器与推理设置

**Files:**
- Create: `frontend/src/components/ModelSelector.tsx`
- Create: `frontend/src/components/InferenceSettings.tsx`

**Interfaces:**
- Consumes: `types.ModelInfo`、`types.MIN_THRESHOLD/MAX_THRESHOLD/DEFAULT_THRESHOLD`、`hooks.useModels`
- Produces:
  - `<ModelSelector models selectedId onSelect />`
  - `<InferenceSettings threshold onThresholdChange maxLen onMaxLenChange useExplain onUseExplainChange />`

- [ ] **Step 1: 实现 `ModelSelector`**

`frontend/src/components/ModelSelector.tsx`：
```tsx
import type { ModelInfo } from "../types";

interface Props {
  models: ModelInfo[];
  selectedId: string;
  onSelect: (id: string) => void;
}

/** 模型下拉。把 meta.json 里的指标直接摊在选项上——选模型时看得见代价。 */
export function ModelSelector({ models, selectedId, onSelect }: Props) {
  const selected = models.find((m) => m.id === selectedId);

  return (
    <div className="flex flex-col gap-2">
      <label htmlFor="model" className="text-sm font-medium text-slate-700">
        模型
      </label>
      <select
        id="model"
        value={selectedId}
        onChange={(e) => onSelect(e.target.value)}
        className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm
                   focus:border-slate-500 focus:outline-none"
      >
        {models.map((m) => (
          <option key={m.id} value={m.id}>
            {m.layers} 层 · 隐藏 {m.hidden_dim} · 准确率 {(m.test_acc * 100).toFixed(2)}%
          </option>
        ))}
      </select>

      {selected && (
        <dl className="grid grid-cols-3 gap-2 rounded-lg bg-slate-50 p-3 text-xs text-slate-600">
          <div>
            <dt className="text-slate-400">测试准确率</dt>
            <dd className="font-mono text-slate-900">{(selected.test_acc * 100).toFixed(2)}%</dd>
          </div>
          <div>
            <dt className="text-slate-400">参数量</dt>
            <dd className="font-mono text-slate-900">{selected.params.toLocaleString()}</dd>
          </div>
          <div>
            <dt className="text-slate-400">训练耗时</dt>
            <dd className="font-mono text-slate-900">{selected.train_time_sec.toFixed(0)}s</dd>
          </div>
        </dl>
      )}
    </div>
  );
}
```

- [ ] **Step 2: 实现 `InferenceSettings`**

`frontend/src/components/InferenceSettings.tsx`：
```tsx
import { DEFAULT_THRESHOLD, MAX_THRESHOLD, MIN_THRESHOLD } from "../types";

interface Props {
  threshold: number;
  onThresholdChange: (v: number) => void;
  maxLen: number;
  onMaxLenChange: (v: number) => void;
  useExplain: boolean;
  onUseExplainChange: (v: boolean) => void;
}

/** 推理期参数面板。
 *
 *  阈值是纯前端参数——后端只返回概率，拖动滑块不触发任何请求。
 *  MAX_LEN 与逐词贡献会影响推理过程，需重新请求。
 */
export function InferenceSettings({
  threshold,
  onThresholdChange,
  maxLen,
  onMaxLenChange,
  useExplain,
  onUseExplainChange,
}: Props) {
  return (
    <div className="flex flex-col gap-4 rounded-lg border border-slate-200 p-4">
      <div className="flex flex-col gap-2">
        <div className="flex items-baseline justify-between">
          <label htmlFor="threshold" className="text-sm font-medium text-slate-700">
            判定阈值
          </label>
          <span className="font-mono text-sm text-slate-500">{threshold.toFixed(2)}</span>
        </div>
        <input
          id="threshold"
          type="range"
          min={MIN_THRESHOLD}
          max={MAX_THRESHOLD}
          step={0.01}
          value={threshold}
          onChange={(e) => onThresholdChange(Number(e.target.value))}
          className="w-full accent-slate-700"
        />
        <p className="text-xs text-slate-400">
          积极概率 ≥ 阈值即判为积极。阈值只影响标签，不改变概率，拖动即时生效。
          调高阈值会减少「积极」判定（更保守），调低则相反。
        </p>
        <button
          type="button"
          onClick={() => onThresholdChange(DEFAULT_THRESHOLD)}
          className="self-start text-xs text-slate-500 underline hover:text-slate-800"
        >
          恢复默认 0.50
        </button>
      </div>

      <div className="flex flex-col gap-2">
        <label htmlFor="maxlen" className="text-sm font-medium text-slate-700">
          最大词数（MAX_LEN）
        </label>
        <input
          id="maxlen"
          type="number"
          min={5}
          max={1000}
          value={maxLen}
          onChange={(e) => onMaxLenChange(Number(e.target.value))}
          className="rounded-lg border border-slate-300 px-3 py-2 text-sm
                     focus:border-slate-500 focus:outline-none"
        />
        <p className="text-xs text-slate-400">超出部分截断。改变此项会影响分词结果，需重新请求。</p>
      </div>

      <label className="flex items-center gap-2 text-sm text-slate-700">
        <input
          type="checkbox"
          checked={useExplain}
          onChange={(e) => onUseExplainChange(e.target.checked)}
          className="accent-slate-700"
        />
        显示逐词贡献（遮挡法，较慢）
      </label>
    </div>
  );
}
```

- [ ] **Step 3: 验证类型检查通过**

Run: `cd frontend; pnpm typecheck`
Expected: 无错误

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/ModelSelector.tsx frontend/src/components/InferenceSettings.tsx
git commit -m "feat(frontend): 模型选择器与推理设置面板"
```

---

## Task 11: 概率条与单条分析面板

**Files:**
- Create: `frontend/src/components/SentimentGauge.tsx`
- Create: `frontend/src/components/SingleAnalyzer.tsx`

**Interfaces:**
- Consumes: `client.fetchPredict` / `fetchExplain`、`types.labelOf`、`ModelSelector`、`InferenceSettings`
- Produces:
  - `<SentimentGauge probPos probNeg label />`
  - `<SingleAnalyzer modelId maxLen useExplain threshold />`

- [ ] **Step 1: 实现 `SentimentGauge`**

`frontend/src/components/SentimentGauge.tsx`：
```tsx
import type { SentimentLabel } from "../types";

interface Props {
  probPos: number;
  probNeg: number;
  label: SentimentLabel;
}

/** 正负面概率条。用一条水平双色条 + 数值，避免两个独立进度条被误读为互不相关。 */
export function SentimentGauge({ probPos, probNeg, label }: Props) {
  const pct = Math.round(probPos * 100);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline gap-3">
        <span className="text-sm text-slate-500">判定结果</span>
        <span
          className={`rounded-full px-3 py-0.5 text-sm font-medium ${
            label === "积极" ? "bg-emerald-100 text-emerald-800" : "bg-rose-100 text-rose-800"
          }`}
        >
          {label}
        </span>
      </div>

      <div className="flex h-6 overflow-hidden rounded-full bg-slate-100">
        <div
          className="flex items-center justify-start bg-emerald-500 pl-2 text-xs text-white transition-all"
          style={{ width: `${pct}%` }}
        >
          {pct >= 15 && `积极 ${pct}%`}
        </div>
        <div
          className="flex flex-1 items-center justify-end bg-rose-500 pr-2 text-xs text-white transition-all"
          style={{ width: `${100 - pct}%` }}
        >
          {100 - pct >= 15 && `消极 ${100 - pct}%`}
        </div>
      </div>

      <div className="flex justify-between font-mono text-xs text-slate-500">
        <span>积极 {probPos.toFixed(4)}</span>
        <span>消极 {probNeg.toFixed(4)}</span>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: 实现 `SingleAnalyzer`**

`frontend/src/components/SingleAnalyzer.tsx`：
```tsx
import { useState } from "react";
import { fetchExplain, fetchPredict } from "../api/client";
import { type Contrib, labelOf } from "../types";
import { SentimentGauge } from "./SentimentGauge";
import { TokenHighlight } from "./TokenHighlight";

interface Props {
  modelId: string;
  maxLen: number;
  useExplain: boolean;
  threshold: number;
}

const EXAMPLES = [
  "这家酒店位置很好，离地铁站很近，房间也干净，服务态度特别棒",
  "隔音太差了，晚上根本睡不着，前台态度还很冷漠，不会再来了",
  "还行吧，没什么特别的，价格偏高，凑合住一晚",
];

export function SingleAnalyzer({ modelId, maxLen, useExplain, threshold }: Props) {
  const [text, setText] = useState("");
  const [probPos, setProbPos] = useState<number | null>(null);
  const [contributions, setContributions] = useState<Contrib[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function analyze() {
    const input = text.trim();
    if (!input) return;
    setBusy(true);
    setError(null);
    try {
      if (useExplain) {
        // explain 的响应已含标签与概率，无需再调 predict —— 那会白跑一次前向
        const r = await fetchExplain(input, modelId, maxLen);
        setProbPos(r.prob_pos);
        setContributions(r.contributions);
      } else {
        const r = await fetchPredict(input, modelId, maxLen);
        setProbPos(r.prob_pos);
        setContributions(null);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setProbPos(null);
      setContributions(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={4}
        placeholder="输入一条中文评论，例如：这家酒店位置很好，但隔音太差……"
        className="w-full resize-y rounded-lg border border-slate-300 p-3 text-sm
                   focus:border-slate-500 focus:outline-none"
      />

      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={() => void analyze()}
          disabled={busy || !text.trim()}
          className="rounded-lg bg-slate-800 px-4 py-2 text-sm font-medium text-white
                     hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy ? "分析中…" : "分析"}
        </button>

        {EXAMPLES.map((ex, i) => (
          <button
            key={ex}
            type="button"
            onClick={() => setText(ex)}
            className="text-xs text-slate-500 underline hover:text-slate-800"
          >
            示例 {i + 1}
          </button>
        ))}
      </div>

      {error && (
        <p className="rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-700">{error}</p>
      )}

      {probPos !== null && (
        <>
          <SentimentGauge
            probPos={probPos}
            probNeg={1 - probPos}
            label={labelOf(probPos, threshold)}
          />
          {contributions && <TokenHighlight contributions={contributions} />}
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 3: 先写 `TokenHighlight` 的占位实现以便本任务可编译**

> 完整实现放在 Task 12。本步骤只创建最小版本，让本任务能通过类型检查。

`frontend/src/components/TokenHighlight.tsx`：
```tsx
import type { Contrib } from "../types";

interface Props {
  contributions: Contrib[];
}

export function TokenHighlight({ contributions }: Props) {
  return <div className="text-sm text-slate-400">逐词贡献：{contributions.length} 个词</div>;
}
```

- [ ] **Step 4: 验证类型检查通过**

Run: `cd frontend; pnpm typecheck`
Expected: 无错误

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/SentimentGauge.tsx frontend/src/components/SingleAnalyzer.tsx frontend/src/components/TokenHighlight.tsx
git commit -m "feat(frontend): 概率条与单条分析面板"
```

---

## Task 12: 逐词贡献高亮

**Files:**
- Modify: `frontend/src/components/TokenHighlight.tsx`（替换 Task 11 的占位实现）

**Interfaces:**
- Consumes: `types.Contrib`
- Produces: `<TokenHighlight contributions />`（完整实现）

- [ ] **Step 1: 实现完整版**

`frontend/src/components/TokenHighlight.tsx`（整体替换）：
```tsx
import type { Contrib } from "../types";

interface Props {
  contributions: Contrib[];
}

/** 把 |delta| 归一化到 [0,1] 作为着色强度。 */
function normalize(contributions: Contrib[]): Map<string, number> {
  const maxAbs = Math.max(...contributions.map((c) => Math.abs(c.delta)), 1e-9);
  return new Map(contributions.map((c) => [c.token, Math.abs(c.delta) / maxAbs]));
}

/** 依据 delta 取背景色。
 *
 *  delta 定义为「删词后的积极概率 − 原始积极概率」：
 *    delta < 0  → 该词推动「积极」，绿色
 *    delta > 0  → 该词推动「消极」，红色
 *
 *  用发散色阶（绿—灰—红）而非单一色阶，因为贡献是有方向的量，
 *  单色阶会把「强烈积极」和「强烈消极」画成同一个样子。 */
function background(c: Contrib, intensity: number): string {
  const alpha = 0.15 + intensity * 0.55; // 保留一个可见的下限，弱贡献也能被看见
  return c.delta < 0
    ? `color-mix(in oklch, var(--contrib-pos) ${alpha * 100}%, transparent)`
    : `color-mix(in oklch, var(--contrib-neg) ${alpha * 100}%, transparent)`;
}

export function TokenHighlight({ contributions }: Props) {
  const intensity = normalize(contributions);
  const top = contributions.slice(0, 10);

  return (
    <div className="flex flex-col gap-3">
      <div>
        <h3 className="mb-2 text-sm font-medium text-slate-700">逐词贡献（遮挡法）</h3>
        <p className="flex flex-wrap gap-1 text-base leading-relaxed">
          {contributions.map((c, i) => (
            <span
              key={`${c.token}-${i}`}
              style={{ backgroundColor: background(c, intensity.get(c.token) ?? 0) }}
              className="rounded px-1 py-0.5"
              title={`删去该词后积极概率变化 ${c.delta >= 0 ? "+" : ""}${c.delta.toFixed(4)}`}
            >
              {c.token}
            </span>
          ))}
        </p>
        <p className="mt-2 text-xs text-slate-400">
          绿色 = 推高积极判定；红色 = 推高消极判定；颜色越深影响越大。悬停可看具体数值。
        </p>
      </div>

      <div>
        <h3 className="mb-2 text-sm font-medium text-slate-700">影响最大的词</h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left text-xs text-slate-400">
              <th className="py-1 font-normal">词</th>
              <th className="py-1 text-right font-normal">影响</th>
              <th className="py-1 text-right font-normal">方向</th>
            </tr>
          </thead>
          <tbody>
            {top.map((c, i) => (
              <tr key={`${c.token}-${i}`} className="border-b border-slate-100">
                <td className="py-1">{c.token}</td>
                <td className="py-1 text-right font-mono">{Math.abs(c.delta).toFixed(4)}</td>
                <td className="py-1 text-right">
                  <span
                    className={c.delta < 0 ? "text-emerald-700" : "text-rose-700"}
                  >
                    {c.delta < 0 ? "积极" : "消极"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: 验证类型检查通过**

Run: `cd frontend; pnpm typecheck`
Expected: 无错误

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/TokenHighlight.tsx
git commit -m "feat(frontend): 逐词贡献高亮与排序表"
```

---

## Task 13: 批量分析

**Files:**
- Create: `frontend/src/components/ResultTable.tsx`
- Create: `frontend/src/components/BatchAnalyzer.tsx`

**Interfaces:**
- Consumes: `client.fetchBatch`、`types.BatchItem`、`types.labelOf`、`types.MAX_BATCH_SIZE`、`papaparse`
- Produces:
  - `<ResultTable items threshold />`（含导出 CSV）
  - `<BatchAnalyzer modelId maxLen threshold />`

- [ ] **Step 1: 实现 `ResultTable`**

`frontend/src/components/ResultTable.tsx`：
```tsx
import Papa from "papaparse";
import { type BatchItem, labelOf } from "../types";

interface Props {
  items: BatchItem[];
  threshold: number;
}

export function ResultTable({ items, threshold }: Props) {
  /** 导出为 CSV。加 BOM 以便 Excel 正确识别 UTF-8 中文。 */
  function exportCsv() {
    const rows = items.map((it) => ({
      文本: it.text,
      标签: labelOf(it.prob_pos, threshold),
      积极概率: it.prob_pos.toFixed(4),
      消极概率: it.prob_neg.toFixed(4),
    }));
    const csv = Papa.unparse(rows);
    const blob = new Blob([`\uFEFF${csv}`], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "sentiment_results.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  const posCount = items.filter((it) => labelOf(it.prob_pos, threshold) === "积极").length;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <p className="text-sm text-slate-500">
          共 {items.length} 条，积极 {posCount} 条 / 消极 {items.length - posCount} 条
        </p>
        <button
          type="button"
          onClick={exportCsv}
          className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm
                     hover:bg-slate-50"
        >
          导出 CSV
        </button>
      </div>

      <div className="max-h-96 overflow-auto rounded-lg border border-slate-200">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-slate-50">
            <tr className="text-left text-xs text-slate-500">
              <th className="px-3 py-2 font-normal">#</th>
              <th className="px-3 py-2 font-normal">文本</th>
              <th className="px-3 py-2 font-normal">标签</th>
              <th className="px-3 py-2 text-right font-normal">积极概率</th>
            </tr>
          </thead>
          <tbody>
            {items.map((it, i) => {
              const label = labelOf(it.prob_pos, threshold);
              return (
                <tr key={`${i}-${it.text.slice(0, 12)}`} className="border-t border-slate-100">
                  <td className="px-3 py-2 text-slate-400">{i + 1}</td>
                  <td className="max-w-md truncate px-3 py-2" title={it.text}>
                    {it.text}
                  </td>
                  <td className="px-3 py-2">
                    <span
                      className={`rounded-full px-2 py-0.5 text-xs ${
                        label === "积极"
                          ? "bg-emerald-100 text-emerald-800"
                          : "bg-rose-100 text-rose-800"
                      }`}
                    >
                      {label}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-slate-600">
                    {it.prob_pos.toFixed(4)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: 实现 `BatchAnalyzer`**

`frontend/src/components/BatchAnalyzer.tsx`：
```tsx
import Papa from "papaparse";
import { useState } from "react";
import { fetchBatch } from "../api/client";
import { type BatchItem, MAX_BATCH_SIZE } from "../types";
import { ResultTable } from "./ResultTable";

interface Props {
  modelId: string;
  maxLen: number;
  threshold: number;
}

/** 从 CSV 文本里抽出待分析文本列，兼容 text / 文本 / 评论 列名。 */
function extractTexts(csvText: string): string[] {
  const parsed = Papa.parse<Record<string, string>>(csvText.trim(), {
    header: true,
    skipEmptyLines: true,
  });
  const fields = parsed.meta.fields ?? [];
  const key = ["text", "文本", "评论", "content"].find((k) => fields.includes(k));
  const rows = parsed.data;

  if (key) {
    return rows.map((r) => (r[key] ?? "").trim()).filter(Boolean);
  }
  // 无表头：把每行第一列当作文本
  const first = Object.keys(rows[0] ?? {})[0];
  return first ? rows.map((r) => (r[first] ?? "").trim()).filter(Boolean) : [];
}

export function BatchAnalyzer({ modelId, maxLen, threshold }: Props) {
  const [raw, setRaw] = useState("");
  const [items, setItems] = useState<BatchItem[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const texts = raw
    .split("\n")
    .map((s) => s.trim())
    .filter(Boolean);

  async function analyze() {
    if (!texts.length) return;
    if (texts.length > MAX_BATCH_SIZE) {
      setError(`一次最多分析 ${MAX_BATCH_SIZE} 条，当前 ${texts.length} 条`);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      setItems(await fetchBatch(texts, modelId, maxLen));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setItems(null);
    } finally {
      setBusy(false);
    }
  }

  async function onFile(file: File) {
    const content = await file.text();
    const extracted = extractTexts(content);
    if (!extracted.length) {
      setError("未能从文件中解析出文本列（需含 text / 文本 / 评论 列，或无表头的单列文本）");
      return;
    }
    setRaw(extracted.join("\n"));
    setError(null);
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-2">
        <label htmlFor="batch" className="text-sm font-medium text-slate-700">
          每行一条评论
        </label>
        <textarea
          id="batch"
          value={raw}
          onChange={(e) => setRaw(e.target.value)}
          rows={8}
          placeholder={"这家酒店很好，服务周到\n隔音差，晚上被吵醒\n性价比一般，凑合"}
          className="w-full resize-y rounded-lg border border-slate-300 p-3 text-sm
                     focus:border-slate-500 focus:outline-none"
        />
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={() => void analyze()}
          disabled={busy || !texts.length}
          className="rounded-lg bg-slate-800 px-4 py-2 text-sm font-medium text-white
                     hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy ? "分析中…" : `分析（${texts.length} 条）`}
        </button>

        <label className="cursor-pointer text-sm text-slate-500 underline hover:text-slate-800">
          或上传 CSV
          <input
            type="file"
            accept=".csv,.txt"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void onFile(f);
            }}
          />
        </label>
      </div>

      {error && <p className="rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-700">{error}</p>}
      {items && <ResultTable items={items} threshold={threshold} />}
    </div>
  );
}
```

- [ ] **Step 3: 验证类型检查通过**

Run: `cd frontend; pnpm typecheck`
Expected: 无错误

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/ResultTable.tsx frontend/src/components/BatchAnalyzer.tsx
git commit -m "feat(frontend): 批量分析与结果导出"
```

---

## Task 14: 主界面组装、构建托管与端到端联调

**Files:**
- Modify: `frontend/src/App.tsx`（整体替换）
- Modify: `frontend/src/main.tsx`（确认引入 `index.css`）
- Modify: `backend/senti/api.py`（追加静态托管路由）
- Modify: `README.md`、`CLAUDE.md`（新建）

**Interfaces:**
- Consumes: 全部前端组件与 hook
- Produces: 可端到端运行的系统；`README.md` 记录实测结果

- [ ] **Step 1: 组装 App**

`frontend/src/App.tsx`（整体替换）：
```tsx
import { useEffect, useState } from "react";
import { BatchAnalyzer } from "./components/BatchAnalyzer";
import { InferenceSettings } from "./components/InferenceSettings";
import { ModelSelector } from "./components/ModelSelector";
import { SingleAnalyzer } from "./components/SingleAnalyzer";
import { useModels } from "./hooks/useModels";
import { DEFAULT_THRESHOLD } from "./types";

type Tab = "single" | "batch";

export default function App() {
  const { models, loading, error } = useModels();
  const [modelId, setModelId] = useState("");
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  const [maxLen, setMaxLen] = useState(150);
  const [useExplain, setUseExplain] = useState(true);
  const [tab, setTab] = useState<Tab>("single");

  // 模型列表到达后自动选中准确率最高的那个
  useEffect(() => {
    if (!modelId && models.length) setModelId(models[0].id);
  }, [models, modelId]);

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-5xl px-6 py-4">
          <h1 className="text-lg font-semibold text-slate-900">
            网络评论情感分类系统
          </h1>
          <p className="text-xs text-slate-500">
            基于 LSTM 的中文评论情感二分类 · 遮挡法逐词贡献
          </p>
        </div>
      </header>

      <main className="mx-auto flex max-w-5xl flex-col gap-6 px-6 py-6">
        {loading && <p className="text-sm text-slate-500">正在加载模型列表…</p>}
        {error && (
          <p className="rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-700">
            无法连接后端：{error}
            <br />
            请先启动后端：<code>cd backend && ..\\.venv\\Scripts\\python.exe run_api.py</code>
          </p>
        )}

        {!loading && !error && models.length === 0 && (
          <p className="rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-800">
            后端尚无已训练模型。请先运行 <code>python -m senti.train</code> 完成训练。
          </p>
        )}

        {models.length > 0 && (
          <div className="grid gap-6 md:grid-cols-[1fr_320px]">
            <div className="flex flex-col gap-4">
              <div className="flex gap-1 border-b border-slate-200">
                {(["single", "batch"] as Tab[]).map((t) => (
                  <button
                    key={t}
                    type="button"
                    onClick={() => setTab(t)}
                    className={`-mb-px border-b-2 px-4 py-2 text-sm ${
                      tab === t
                        ? "border-slate-800 font-medium text-slate-900"
                        : "border-transparent text-slate-500 hover:text-slate-800"
                    }`}
                  >
                    {t === "single" ? "单条分析" : "批量分析"}
                  </button>
                ))}
              </div>

              {tab === "single" ? (
                <SingleAnalyzer
                  modelId={modelId}
                  maxLen={maxLen}
                  useExplain={useExplain}
                  threshold={threshold}
                />
              ) : (
                <BatchAnalyzer modelId={modelId} maxLen={maxLen} threshold={threshold} />
              )}
            </div>

            <aside className="flex flex-col gap-4">
              <div className="rounded-lg border border-slate-200 bg-white p-4">
                <ModelSelector models={models} selectedId={modelId} onSelect={setModelId} />
              </div>
              <div className="rounded-lg bg-white">
                <InferenceSettings
                  threshold={threshold}
                  onThresholdChange={setThreshold}
                  maxLen={maxLen}
                  onMaxLenChange={setMaxLen}
                  useExplain={useExplain}
                  onUseExplainChange={setUseExplain}
                />
              </div>
            </aside>
          </div>
        )}
      </main>
    </div>
  );
}
```

`frontend/src/main.tsx`（确认第一行是样式引入）：
```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
```

- [ ] **Step 2: 后端托管构建产物（单端口演示）**

在 `backend/senti/api.py` 的 `create_app()` 中，把 `return app` 之前插入：
```python
    # 若前端已构建，则由 Flask 托管 dist/，实现单端口单进程运行，
    # 演示时不必同时开 Vite 与 Flask 两个进程。
    dist = config.PROJECT_ROOT / "frontend" / "dist"
    if dist.exists():
        @app.get("/")
        @app.get("/<path:subpath>")
        def spa(subpath: str = ""):
            from flask import send_from_directory

            target = dist / subpath
            if subpath and target.is_file():
                return send_from_directory(dist, subpath)
            # 其余路径一律回退到 index.html（前端路由接管）
            return send_from_directory(dist, "index.html")

        print(f"已启用前端静态托管：{dist}")
```
并在 `api.py` 顶部 import 区确认 `from senti import config, infer`（`config` 已在 Task 7 引入）。

- [ ] **Step 3: 端到端联调**

```powershell
# 终端 1：构建前端
cd frontend; pnpm build

# 终端 2：启动后端（此时会托管 dist/）
cd backend; ..\.venv\Scripts\python.exe run_api.py
```
浏览器访问 `http://127.0.0.1:5000/`。

Expected：页面加载，模型下拉有 6 个选项，可完成以下检查。

- [ ] **Step 4: 逐项人工验收**

| # | 操作 | 期望 |
|---|------|------|
| 1 | 点「示例 1」（明显正面） | 标签「积极」，绿色占比高 |
| 2 | 点「示例 2」（明显负面） | 标签「消极」，红色占比高 |
| 3 | 拖动阈值到 0.9 | 示例 1 **不再**即时改变概率，但标签可能翻成「消极」；页面**不重新请求** |
| 4 | 关闭「显示逐词贡献」开关，再分析 | 只显示概率条，无高亮 |
| 5 | 打开逐词贡献，分析示例 1 | 出现高亮与「影响最大的词」表；「很好」「干净」等为绿色（积极） |
| 6 | `MAX_LEN` 改为 5，分析长文本 | 参与分析/高亮的词数明显减少 |
| 7 | 切换到准确率最低的模型，重新分析 | 概率可能变化，说明模型确实切了 |
| 8 | 批量页粘贴 3 行文本 → 分析 | 表格 3 行，标签与单条分析一致 |
| 9 | 批量页点「导出 CSV」 | 下载 CSV，用 Excel 打开中文不乱码 |
| 10 | 批量页上传含 `文本` 列的 CSV | 文本被正确抽出 |
| 11 | 停掉后端后点「分析」 | 页面显示错误提示，不白屏 |

- [ ] **Step 5: 写 `README.md`**

沿用兄弟项目惯例，**必须包含实测结果表格与「重新搭建环境」章节**：
````markdown
# 基于 LSTM 的网络评论情感分类系统

河南理工大学计算机学院《神经网络与深度学习》课程设计 · 选题(3)
指导教师：侯蓓蓓

基于 LSTM 的中文评论情感二分类系统：PyTorch 训练模型，Flask 提供推理与
可解释性接口，React 前端提供单条分析、批量分析与参数调节。

## 系统组成

| 层 | 目录 | 职责 |
|----|------|------|
| 数据层 | `backend/senti/data.py` | 语料下载、jieba 分词、词表构建、索引编码 |
| 模型层 | `backend/senti/model.py` | Embedding + LSTM + Linear |
| 训练层 | `backend/senti/train.py` | 训练循环、网格实验、产物落盘 |
| 推理层 | `backend/senti/infer.py` | 单条/批量预测、遮挡法逐词贡献 |
| 接口层 | `backend/senti/api.py` | Flask REST 接口 |
| 前端 | `frontend/` | React + Vite + Tailwind |

## 数据集

ChnSentiCorp（谭松波中文酒店评论语料），来源与划分见 `backend/senti/data.py`。

## 环境

| 组件 | 版本 |
|------|------|
| Python | 3.13.13（项目级 `.venv`，uv 创建） |
| PyTorch | 2.12.0+cu132 |
| Node | v24.19.0 |
| pnpm | 11.7.0 |

**不依赖 torchvision**（本项目无图像任务）。

### 重新搭建环境

```powershell
uv venv .venv --python "D:\settings\Language\Python\Python 3.13.13\python.exe"
uv pip install --python .venv\Scripts\python.exe "d:/settings/Language/Python/库/torch-2.12.0+cu132-cp313-cp313-win_amd64.whl"
uv pip install --python .venv\Scripts\python.exe jieba flask flask-cors pandas matplotlib scikit-learn pytest requests pyarrow
cd frontend; pnpm install
```

> `pyarrow` 由 Task 2 引入（读取原生划分的 parquet 语料），**已并入上面这条安装命令**。

## 运行

```powershell
# 1. 下载语料
cd backend; ..\.venv\Scripts\python.exe -c "from senti import data; data.download()"

# 2. 数据管线体检（必须先通过）
..\.venv\Scripts\python.exe -m senti.train --overfit-test

# 3. 训练 6 组模型（约 <填实际耗时>）
..\.venv\Scripts\python.exe -m senti.train

# 4. 启动后端
..\.venv\Scripts\python.exe run_api.py

# 5. 前端（开发模式，可选）
cd ..\frontend; pnpm dev
```

## 实测结果

> 本表由 `backend/artifacts/meta.json` 生成，与代码实际跑出的一致。

| 模型 | 层数 | 隐藏维度 | 参数量 | 测试准确率 | 训练耗时 |
|------|------|---------|--------|-----------|---------|
| <运行 Task 5 Step 7 后按实际输出填写> |

**数据集**：ChnSentiCorp，训练 <N> 条 / 测试 <N> 条，词表 <N>
**最优模型**：<id>，测试准确率 <N>%（题目门槛 85%）

## 测试

```powershell
cd backend; ..\.venv\Scripts\python.exe -m pytest -v
```

## 设计文档与实现计划

- 设计：`docs/superpowers/specs/2026-10-11-lstm-sentiment-system-design.md`
- 计划：`docs/superpowers/plans/2026-10-11-lstm-sentiment-system.md`
````
**注意**：README 中带尖括号的部分必须替换为 Task 5 实际跑出的数值，
不得留占位符。

- [ ] **Step 6: 写 `CLAUDE.md`**

沿用兄弟项目惯例（给 AI 助手的项目说明）：
````markdown
# CLAUDE.md

本目录是《神经网络与深度学习》课程设计——「基于 LSTM 的网络评论情感分类系统」。
前后端分离：Python 后端（训练+推理）+ React 前端。

## 项目结构

- `backend/senti/` —— 五层划分：`config` / `data` / `model` / `train` / `infer` / `api`
- `backend/artifacts/` —— 训练产物（checkpoint + 词表 + 元数据），不入库
- `frontend/` —— React + Vite + TypeScript + Tailwind v4
- `docs/superpowers/` —— 设计文档与实现计划
- `.venv/` —— 项目专用虚拟环境，**不要用全局解释器替代**

## 常用命令

所有后端命令在 `backend/` 下执行：

```powershell
..\.venv\Scripts\python.exe -m senti.train --overfit-test   # 数据管线体检
..\.venv\Scripts\python.exe -m senti.train                  # 训练全部 6 组
..\.venv\Scripts\python.exe -m senti.train --config 2 128   # 只训一组
..\.venv\Scripts\python.exe run_api.py                      # 启动后端
..\.venv\Scripts\python.exe -m pytest -v                    # 全部测试
```

前端在 `frontend/` 下执行：`pnpm dev` / `pnpm build` / `pnpm typecheck` / `pnpm lint`

## 关键约定（改动前必读）

1. **测试集只用于最终报告指标**。选最优轮次一律用验证集。
   不要写 `if test_acc > best: best = test_acc` —— 那是测试集泄漏。
2. **词表与 checkpoint 必须同源**。`vocab.json` 与 `*.pt` 同生命周期，
   `Predictor.__init__` 有 `vocab_size` 一致性断言。改词表必须重训。
3. **阈值是纯前端参数**。后端只返回概率，`/api/*` 不接收阈值。
   标签判定规则只在 `frontend/src/types.ts` 的 `labelOf` 定义一处。
4. **`MAX_LEN` 改了必须重训**吗？不需要——它只影响推理期的截断，
   但会改变输入表示，故由后端参数接收，不走前端本地计算。
5. **超参数只在 `backend/senti/config.py` 定义**，代码内不得出现字面量。
6. **不装 torchvision**。

## 验证方式

- 后端：`pytest`（纯函数 + 接口契约）
- 数据管线：`--overfit-test`，必须能过拟合
- 前端：`pnpm typecheck` + `pnpm build` + 浏览器人工走通
````

- [ ] **Step 7: 全量测试与构建**

Run:
```powershell
cd backend; ..\.venv\Scripts\python.exe -m pytest -v
cd ..\frontend; pnpm typecheck; pnpm lint; pnpm build
```
Expected: 后端测试全绿；前端无类型错误、无 lint 错误、构建成功

- [ ] **Step 8: Commit**

```bash
git add frontend/src backend/senti/api.py README.md CLAUDE.md
git commit -m "feat: 主界面组装、静态托管与项目文档"
```

---

# 自查记录

## Spec 覆盖核对

| spec 章节 | 对应任务 |
|-----------|---------|
| 1.2 验收标准 A1 数据预处理 | Task 3 |
| 1.2 A2 准确率 ≥85% | Task 5 Step 7 |
| 1.2 A3 人机交互 | Task 10–14 |
| 1.2 A4 可解释性 | Task 6、Task 12 |
| 1.2 A5 批量分析 | Task 6、Task 13 |
| 1.2 A6 参数可调 | Task 5（多 checkpoint）、Task 10、Task 14 |
| 1.2 A7 模块化 | Task 1、4、5、6、7 的文件划分 |
| 1.2 A8 可复现 | Task 5 `set_seed`、Task 14 README 实测表 |
| 4.2 数据层函数表 | Task 2、Task 3 |
| 4.3 模型层 | Task 4 |
| 4.4 训练层函数表 + meta.json | Task 5 |
| 4.5 推理层与遮挡法 | Task 6 |
| 4.6 接口层薄层 | Task 7 |
| 5 接口契约（5 条路由 + 错误码） | Task 7 |
| 6 前端组件结构与交互 | Task 8–14 |
| 6.3 CORS / Vite 代理 | Task 8 |
| 6.4 发散色阶 | Task 12 |
| 7 参数分层（阈值纯前端） | Task 9 `labelOf`、Task 11、Task 14 Step 4 #3 |
| 8 数据来源 | Task 2 |
| 9 环境与依赖（不装 torchvision） | Task 1 Step 2 |
| 10 验证策略 5 层 | Task 2–7 各自测试 + Task 5 过拟合 + Task 14 人工验收 |
| 11 风险对策 | Task 2 Step 2（数据源）、Task 5 Step 7（未达标）、Task 6（词表断言）、Task 12（贡献上限） |

无遗漏项。

## 与 spec 的两处偏离（已确认无冲突）

1. **不创建 `tailwind.config.js`**。spec 目录结构列出了它，但 Tailwind v4 改为
   CSS 优先配置，不再需要该文件。已在 Task 8 Step 1 注明。
2. **`api.py` 追加静态托管路由**。spec 未列该路由，但 spec 6.3 明确写了
   「演示期可由 Flask 托管 `dist/`」，Task 14 把它落实。该路由仅在
   `frontend/dist/` 存在时注册，不影响 Task 7 的接口测试。

## 类型一致性核对

- `Predictor.predict` 返回键 `prob_pos` / `prob_neg` / `label` / `tokens`
  → 前端 `PredictResponse` 一致（Task 9）
- `explain` 返回 `contributions: [{token, delta}]`
  → 前端 `Contrib` 一致（Task 9）
- `list_models` 返回键 `id` / `layers` / `hidden_dim` / `embedding_dim` /
  `test_acc` / `val_acc` / `params` / `train_time_sec`
  → 前端 `ModelInfo` 一致（Task 9）
- `train.split_train_val` 返回 `(DataLoader, DataLoader)`，元素为
  `(ids, lengths, labels)` 三元组 → `make_loader` 与 `evaluate` 均按三元组解包（Task 5）

# 执行交接

计划完成后，按 writing-plans 技能的标准交接流程，在两个执行方式中选择其一：
子代理逐任务执行（推荐）或本会话内批量执行。
