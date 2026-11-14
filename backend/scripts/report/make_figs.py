# -*- coding: utf-8 -*-
"""生成报告示意图的第 2、3 章 6 张（matplotlib 版，已退役）。

⚠️ 自 2026-10-30 起，这 6 张示意图的权威来源是 drawio/*.drawio，
由 export_drawio.ps1 调用 draw.io 导出到 figs/。本脚本保留作历史备份，
直接运行会覆盖 draw.io 的产物，因此必须显式传 --force。

全部用 matplotlib 的图元手工绘制，不依赖 graphviz，
保证在任何装了 matplotlib 的环境都能复现。
输出目录：figs/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.stdout.reconfigure(encoding="utf-8")

OUT = Path(__file__).resolve().with_name("figs")
OUT.mkdir(exist_ok=True)

DRAWIO = Path(__file__).resolve().with_name("drawio")

# ── 中文字体 ─────────────────────────────────────────────────────────
for name in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
    try:
        matplotlib.font_manager.findfont(name, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [name]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

INK = "#1f2937"
LINE = "#6b7280"
FILL = "#eef2ff"
FILL2 = "#ecfdf5"
FILL3 = "#fef3c7"
FILL4 = "#f3f4f6"


def box(ax, x, y, w, h, text, fill=FILL, fs=10, bold=False, radius=0.02):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={radius}",
        linewidth=1.2, edgecolor=LINE, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.5)


def arrow(ax, p1, p2, style="-|>", ls="-", color=LINE, lw=1.3, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=13,
        linewidth=lw, color=color, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", zorder=1,
        shrinkA=2, shrinkB=2))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10 * h / w)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white",
                pad_inches=0.08)
    plt.close(fig)
    print(f"  {p.name}")


# ── 图 2-1 总体技术路线 ──────────────────────────────────────────────
def fig_roadmap():
    fig, ax = canvas(9, 5.2)
    ax.text(2.5, 5.55, "训练期（离线）", ha="center", fontsize=11,
            fontweight="bold", color="#1d4ed8")
    ax.text(7.5, 5.55, "推理期（在线）", ha="center", fontsize=11,
            fontweight="bold", color="#047857")
    ax.plot([5], [0.15], marker="", ls="")
    ax.axvline(5, ymin=0.05, ymax=0.95, ls="--", lw=1, color="#d1d5db")

    left = [
        (4.55, "数据集/（五个语料）"),
        (3.75, "数据层 data.py\n下载 · 校验 · 分词 · 建表 · 编码"),
        (2.95, "模型层 model.py\nEmbedding + LSTM + Linear"),
        (2.15, "训练层 train.py\n6 组网格 · 验证集选优"),
        (1.35, "产物：*.pt + vocab.json\n+ meta.json"),
    ]
    for y, t in left:
        box(ax, 0.55, y, 3.9, 0.6, t, fill=FILL, fs=9)
    for i in range(len(left) - 1):
        arrow(ax, (2.5, left[i][0]), (2.5, left[i + 1][0] + 0.6))

    right = [
        (4.55, "前端 React\n单条 / 批量 / 参数调节"),
        (3.75, "接口层 api.py\n5 条 REST 路由"),
        (2.95, "推理层 infer.py\n载入产物 · 预测 · 遮挡法"),
        (2.15, "概率 → 阈值 → 标签\n（阈值只在前端）"),
    ]
    for y, t in right:
        box(ax, 5.55, y, 3.9, 0.6, t, fill=FILL2, fs=9)
    for i in range(len(right) - 1):
        arrow(ax, (7.5, right[i][0]), (7.5, right[i + 1][0] + 0.6))

    # 产物 → 推理层（跨边界）
    arrow(ax, (4.45, 1.65), (5.55, 3.25), style="-|>", ls="--",
          color="#b45309", lw=1.5)
    ax.text(4.62, 2.05, "只消费产物", ha="left", va="center", fontsize=8.5,
            color="#b45309",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))

    # 请求与响应：标在分界线下方空白处，避开两个框
    ax.text(5.0, 5.35, "HTTP / JSON", ha="center", va="center", fontsize=8.5,
            color="#047857")
    ax.set_ylim(1.12, 5.72)
    save(fig, "fig_2_1_roadmap.png")


# ── 图 2-2 分层架构 ──────────────────────────────────────────────────
def fig_architecture():
    fig, ax = canvas(7.0, 4.4)
    layers = [
        ("接口层  api.py", "路由 · 参数校验 · JSON 序列化 · 静态托管", FILL3),
        ("推理层  infer.py", "载入产物 · 单条/批量预测 · 遮挡法贡献", FILL2),
        ("训练层  train.py", "训练循环 · 验证选优 · 网格实验 · 曲线", FILL),
        ("模型层  model.py", "Embedding + LSTM + Linear", FILL),
        ("数据层  data.py", "下载 · 校验 · 分词 · 词表 · 编码", FILL),
    ]
    H, STEP, X, W = 0.62, 0.78, 0.55, 5.9
    y = 4.62
    for i, (name, desc, fill) in enumerate(layers):
        box(ax, X, y, W, H, "", fill=fill)
        ax.text(X + 0.28, y + H * 0.64, name, ha="left", va="center",
                fontsize=10, fontweight="bold", color=INK, zorder=3)
        ax.text(X + 0.28, y + H * 0.26, desc, ha="left", va="center",
                fontsize=8.2, color="#4b5563", zorder=3)
        if i < len(layers) - 1:
            arrow(ax, (X + W / 2, y), (X + W / 2, y - (STEP - H) - 0.02))
        y -= STEP

    box(ax, X, 0.62, W, 0.46,
        "config.py —— 路径与超参数常量（全项目唯一来源）", fill=FILL4, fs=8.5)
    box(ax, X, 0.06, W, 0.46,
        "corpora.py —— 多语料下载与统一读取（旁路，不参与推理）",
        fill=FILL4, fs=8.5)
    ax.set_xlim(0.35, 6.75)
    ax.set_ylim(0.0, 5.45)
    save(fig, "fig_2_2_architecture.png")


# ── 图 3-1 数据预处理流程 ────────────────────────────────────────────
def fig_pipeline():
    fig, ax = canvas(9.4, 3.5)
    steps = [
        "语料下载\n（多镜像回退）",
        "解析校验\n（列名 · 标签）",
        "原子改名\n落定",
        "jieba 分词\n去标点",
        "词表构建\n仅训练集",
        "索引化\n截断/填充",
        "DataLoader\n(ids, lengths)",
    ]
    w, gap = 1.18, 0.22
    x = 0.3
    for i, t in enumerate(steps):
        fill = FILL2 if i in (1, 2) else FILL
        box(ax, x, 1.35, w, 0.85, t, fill=fill, fs=8.5)
        if i < len(steps) - 1:
            arrow(ax, (x + w, 1.775), (x + w + gap, 1.775))
        x += w + gap

    # 旁注
    ax.text(0.3 + (w + gap) * 1.5, 0.85,
            "校验失败按「下载失败」处理并换下一个源：\n"
            "否则一个 200 的 HTML 错误页会被当成语料永久固化",
            ha="center", va="top", fontsize=7.8, color="#b45309")
    ax.text(0.3 + (w + gap) * 4.5, 0.85,
            "词表只统计训练集，避免测试集词频泄漏",
            ha="center", va="top", fontsize=7.8, color="#b45309")
    ax.set_ylim(0.25, 2.62)
    save(fig, "fig_3_1_pipeline.png")


# ── 图 3-2 LSTM 单元 ─────────────────────────────────────────────────
def fig_lstm_cell():
    fig, ax = canvas(8.6, 4.2)
    Y = 4.35                      # 细胞状态水平线
    ax.text(1.15, Y, "C(t-1)", ha="center", va="center", fontsize=9.5)
    ax.text(8.45, Y, "C(t)", ha="center", va="center", fontsize=9.5)
    arrow(ax, (1.55, Y), (2.0, Y), lw=2.0, color="#1d4ed8")

    nodes = [(2.2, "×"), (4.6, "+"), (7.55, "×")]
    for x, sym in nodes:
        ax.plot(x, Y, marker="o", ms=13, color="white", mec="#4b5563",
                mew=1.2, zorder=4)
        ax.text(x, Y, sym, ha="center", va="center", fontsize=10,
                color=INK, zorder=5)
    # 连接线段
    for x0, x1 in ((2.0, 2.07), (2.33, 4.47), (4.73, 7.42), (7.68, 8.3)):
        arrow(ax, (x0, Y), (x1, Y), lw=2.0, color="#1d4ed8")

    gates = [
        (2.2, "遗忘门 f", "σ", FILL3),
        (3.55, "输入门 i", "σ", FILL3),
        (5.7, "候选 C′", "tanh", FILL2),
        (7.55, "输出门 o", "σ · tanh", FILL3),
    ]
    BW, BH, BY = 1.25, 0.78, 2.55
    for cx, name, act, fill in gates:
        box(ax, cx - BW / 2, BY, BW, BH, f"{name}\n{act}", fill=fill, fs=8.5)

    # 各门到细胞状态线的连线
    arrow(ax, (2.2, BY + BH), (2.2, Y - 0.14))
    arrow(ax, (3.55, BY + BH), (4.6, Y - 0.14), rad=0.12)
    arrow(ax, (5.7, BY + BH), (4.6, Y - 0.14), rad=-0.12)

    # 输出门向下产出 h(t)
    arrow(ax, (7.55, Y - 0.14), (7.55, BY - 0.02), color="#1d4ed8", lw=1.6)
    ax.text(7.55, 2.15, "h(t) = o ⊙ tanh(C(t))", ha="center", va="center",
            fontsize=8.5, color="#1d4ed8")

    ax.text(4.6, 1.15,
            "细胞状态沿水平线以加法更新，梯度可近乎无衰减地回传",
            ha="center", va="center", fontsize=8, color="#4b5563")
    ax.text(3.55, 4.9, "C(t) = f ⊙ C(t-1) + i ⊙ C′(t)",
            ha="left", va="center", fontsize=8.5, color="#1d4ed8")
    ax.set_xlim(0.7, 9.0)
    ax.set_ylim(0.75, 5.15)
    save(fig, "fig_3_2_lstm_cell.png")


# ── 图 3-3 模型结构 ──────────────────────────────────────────────────
def fig_model():
    fig, ax = canvas(8.6, 5.8)
    seq = [1.25, 2.05, 2.85, 3.65]
    for i, x in enumerate(seq):
        box(ax, x, 0.55, 0.62, 0.5, f"w{i}", fill=FILL4, fs=8)
        arrow(ax, (x + 0.31, 1.05), (x + 0.31, 1.55), color="#9ca3af")

    box(ax, 1.0, 1.55, 4.1, 0.62, "Embedding  (20002 × 128)", fill=FILL, fs=9.5)

    box(ax, 1.0, 2.45, 4.1, 0.72, "LSTM  (层数 × 隐藏维可调)", fill=FILL2, fs=9.5)
    arrow(ax, (3.05, 2.17), (3.05, 2.45))
    arrow(ax, (3.05, 3.17), (3.05, 3.45))

    box(ax, 1.0, 3.45, 4.1, 0.62, "取最后一个真实词的隐藏状态", fill=FILL4, fs=9)
    ax.text(5.35, 3.76, "← 按 lengths gather\n   不用 output[-1]",
            fontsize=7.5, color="#b45309", va="center")

    box(ax, 1.0, 4.35, 4.1, 0.62, "Linear  (hidden → 2)", fill=FILL3, fs=9.5)
    arrow(ax, (3.05, 4.07), (3.05, 4.35))
    arrow(ax, (3.05, 4.97), (3.05, 5.25))
    box(ax, 1.0, 5.25, 1.85, 0.55, "积极 logit", fill="#dbeafe", fs=9)
    box(ax, 3.25, 5.25, 1.85, 0.55, "消极 logit", fill="#fee2e2", fs=9)
    ax.text(5.35, 5.0, "softmax 由交叉熵\n损失内部完成", ha="left",
            va="center", fontsize=7.5, color="#6b7280")

    ax.text(5.35, 1.85, "padding_idx = 0，\n填充位不参与梯度", ha="left",
            va="center", fontsize=7.5, color="#6b7280")
    ax.set_xlim(0.3, 7.2)
    save(fig, "fig_3_3_model.png")


# ── 图 3-4 训练流程 ──────────────────────────────────────────────────
def fig_train():
    fig, ax = canvas(8.0, 5.0)
    box(ax, 1.4, 4.55, 5.2, 0.55, "固定随机种子 · 载入划分", fill=FILL4, fs=9)
    box(ax, 1.4, 3.75, 5.2, 0.6, "训练一轮：前向 → 交叉熵 → 反向 → 更新",
        fill=FILL, fs=9)
    box(ax, 1.4, 2.95, 5.2, 0.6, "在验证集上评估准确率", fill=FILL2, fs=9)
    box(ax, 1.4, 2.15, 5.2, 0.6, "验证准确率创新高？\n是 → 保存当前权重快照",
        fill=FILL3, fs=9)
    box(ax, 1.4, 1.35, 5.2, 0.55, "达到 15 轮？", fill=FILL4, fs=9)
    box(ax, 1.4, 0.55, 5.2, 0.6, "载入最优快照 · 测试集只评估一次",
        fill="#fce7f3", fs=9)

    for y in (4.55, 3.75, 2.95, 2.15, 1.35):
        arrow(ax, (4.0, y), (4.0, y - 0.2))

    # 循环回边
    arrow(ax, (1.4, 1.62), (0.75, 1.62), style="-", lw=1.3)
    arrow(ax, (0.75, 1.62), (0.75, 4.05), style="-", lw=1.3)
    arrow(ax, (0.75, 4.05), (1.4, 4.05))
    ax.text(0.85, 2.85, "否，继续下一轮", fontsize=8, color="#6b7280",
            rotation=90, va="center")

    ax.text(6.85, 3.25, "测试集全程\n不参与任何选择", fontsize=8.5,
            color="#be123c", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fff1f2", ec="#fecdd3"))
    save(fig, "fig_3_4_train.png")


def main() -> None:
    # 守卫：drawio/ 在就说明示意图已改由 draw.io 维护。
    # 不加这个判断的话，随手跑一下就会把 draw.io 导出的图整批覆盖回 matplotlib 版，
    # 而且是静默发生的——生成 docx 之后才会发现图变了。
    if DRAWIO.is_dir() and "--force" not in sys.argv:
        print("drawio/ 目录存在：示意图以 .drawio 为权威来源。")
        print("请改用：pwsh -File export_drawio.ps1")
        print("确实要用 matplotlib 版覆盖 figs/，请加 --force。")
        return

    print("生成示意图：")
    fig_roadmap()
    fig_architecture()
    fig_pipeline()
    fig_lstm_cell()
    fig_model()
    fig_train()
    print(f"\n输出目录：{OUT}")


if __name__ == "__main__":
    main()
