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
UNK_IDX = 1            # <unk> 的索引；实词索引从 2 开始

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
