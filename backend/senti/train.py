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

from senti import config, corpora, data, model


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


@torch.no_grad()
def evaluate_full(m: nn.Module, loader: DataLoader, dev: torch.device) -> dict:
    """一次前向算出准确率、**平衡准确率**与多数类基线。

    为什么不能只看准确率：合并语料里积极占 74.4%，一个「无脑判积极」的模型白拿
    74.3% 准确率。**平衡准确率**（两类召回率的均值）对类别比例不敏感，在偏斜数据上
    才是能读的数字；`majority_baseline` 就是那个「无脑判多数类」的分数，用来给
    准确率提供一个可比的参照——报告里写「91%，而多数类基线是 74%」才是有信息的。
    """
    m.eval()
    correct = total = 0
    class_correct = [0, 0]
    class_total = [0, 0]

    for xb, lb, yb in loader:
        x = xb.transpose(1, 0).contiguous().to(dev)
        pred = m(x, lb.to(dev)).argmax(dim=1).cpu()
        correct += (pred == yb).sum().item()
        total += len(yb)
        for cls in (0, 1):
            mask = yb == cls
            class_total[cls] += int(mask.sum().item())
            class_correct[cls] += int((pred[mask] == cls).sum().item())

    recalls = [class_correct[c] / class_total[c] for c in (0, 1) if class_total[c] > 0]
    return {
        "acc": correct / max(total, 1),
        "balanced_acc": sum(recalls) / len(recalls) if recalls else 0.0,
        "majority_baseline": max(class_total) / max(total, 1),
        "class_total": class_total,
    }


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


def corpus_label() -> str:
    """`meta.json` 里 dataset 字段的取值——如实描述本轮到底用了哪套语料。

    这里曾经写死 "ChnSentiCorp"，合并语料之后就成了一句错话：元数据是报告「数据集」
    章节的取值来源，不能让读者以为 89% 是那 9600 条跑出来的。
    """
    if config.CORPUS_PROFILE == "merged":
        return "ChnSentiCorp ∪ " + " ∪ ".join(corpora.CORPORA) + "（合并后全局重划分）"
    return "ChnSentiCorp（原生 train/test 划分）"


def prepare(save_vocab: bool = True) -> dict:
    """载入划分 → 建词表 → 编码，**只做一次**，供全部配置复用。

    为什么必须只做一次：合并语料有 184 万条，编码成 (N, 150) 的 int64 矩阵约 2 GB，
    过一次 jieba 也要几分钟。6 个配置各来一遍既慢又容易把内存吃光——
    这是数据量涨两个数量级之后才暴露出来的问题。
    """
    tr_texts, tr_labels, te_texts, te_labels = corpora.load_training_split()
    vocab = data.build_vocab(tr_texts)
    if save_vocab:
        data.save_vocab(vocab)

    tr_ids, tr_len = data.encode(tr_texts, vocab)
    te_ids, te_len = data.encode(te_texts, vocab)

    return {
        "vocab": vocab,
        "tr_ids": tr_ids,
        "tr_len": tr_len,
        "tr_labels": np.array(tr_labels, dtype=np.int64),
        "te_ids": te_ids,
        "te_len": te_len,
        "te_labels": np.array(te_labels, dtype=np.int64),
        "train_size": len(tr_texts),
        "test_size": len(te_texts),
    }


def train_one(layers: int, hidden: int, prep: dict, quiet: bool = False) -> dict:
    """训练单个配置，返回指标字典并保存 checkpoint。

    每轮在验证集上评估，**保留验证准确率最高的那轮权重**；
    训练结束后用该权重在测试集上评估一次，作为最终指标。

    数据由调用方经 `prepare()` 备好并传入——见那里的说明（只编码一次）。
    """
    set_seed()
    dev = device()
    mid = config.model_id(layers, hidden)
    vocab = prep["vocab"]

    tr_loader, val_loader = split_train_val(prep["tr_ids"], prep["tr_len"], prep["tr_labels"])
    te_loader = make_loader(prep["te_ids"], prep["te_len"], prep["te_labels"], shuffle=False)

    m = model.SentimentLSTM(
        vocab_size=len(vocab) + config.FIRST_WORD_IDX, hidden_dim=hidden, num_layers=layers
    ).to(dev)
    optimizer = optim.Adam(m.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val, best_state, best_val_balanced = -1.0, None, 0.0
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
        val = evaluate_full(m, val_loader, dev)
        history["loss"].append(epoch_loss)
        history["val_acc"].append(val["acc"])

        if val["acc"] > best_val:
            best_val = val["acc"]
            best_val_balanced = val["balanced_acc"]
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}

        if not quiet:
            print(f"  [{mid}] epoch {epoch + 1}/{config.EPOCHS} "
                  f"loss={epoch_loss:.4f} val_acc={val['acc']:.4f} "
                  f"val_balanced={val['balanced_acc']:.4f}")

    # 用验证最优权重报告测试集指标
    assert best_state is not None
    m.load_state_dict(best_state)
    test = evaluate_full(m, te_loader, dev)
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
        "val_balanced_acc": round(best_val_balanced, 4),
        "test_acc": round(test["acc"], 4),
        "test_balanced_acc": round(test["balanced_acc"], 4),
        # 测试集里多数类的占比——准确率的「无脑参照线」。
        # 偏斜数据上必须与准确率一起看：91% 对 74% 的基线和 91% 对 50% 的基线，
        # 说的是完全不同的两件事。
        "majority_baseline": round(test["majority_baseline"], 4),
        "params": model.count_parameters(m),
        "train_time_sec": round(train_time, 1),
        "vocab_size": len(vocab) + config.FIRST_WORD_IDX,
    }
    print(f"  [{mid}] 验证 {best_val:.4f}（平衡 {best_val_balanced:.4f}） / "
          f"测试 {test['acc']:.4f}（平衡 {test['balanced_acc']:.4f}，"
          f"多数类基线 {test['majority_baseline']:.4f}） / "
          f"参数 {metrics['params']:,} / 耗时 {train_time:.0f}s")
    return metrics


def overfit_test(n: int = 40, epochs: int = 30, load=None) -> float:
    """小样本过拟合测试——验证数据管线正确性的最快手段。

    取 n 条样本（不切验证集、不加 dropout），看训练准确率能否冲到接近 100%。
    管线正确时必然能过拟合；若不能，问题在词表/编码/取末词，而不在超参数。

    `load` 是划分来源，缺省用当前 CORPUS_PROFILE 的那套（即真的去查合并语料）。
    单元测试会传入单语料的划分，免得为了过拟合 40 条样本而翻遍 184 万条。

    返回训练集最终准确率。
    """
    set_seed()
    dev = device()

    tr_texts, tr_labels, _, _ = (load or corpora.load_training_split)()
    # 词表从**全量训练集**构建，不能用那 40 条：否则样本里的词几乎全落到 <unk>，
    # 过拟合测试就测不到「取最后一个真实词」这些真正想验的环节。
    # 代价是这一步要过一遍全量分词（合并语料下约几分钟），属一次性的诊断开销。
    vocab = data.build_vocab(tr_texts)
    ids, lengths = data.encode(tr_texts[:n], vocab)
    labels = np.array(tr_labels[:n], dtype=np.int64)

    loader = make_loader(ids, lengths, labels, batch_size=min(n, 16), shuffle=True)

    # hidden/层数刻意取「小而固定」的诊断模型：本测试只关心管线能否过拟合，
    # 不是在搜索超参数，所以这两个数不进 config，也不随 GRID 变化。
    m = model.SentimentLSTM(
        vocab_size=len(vocab) + config.FIRST_WORD_IDX, hidden_dim=64, num_layers=1, dropout=0.0
    ).to(dev)
    optimizer = optim.Adam(m.parameters(), lr=config.LEARNING_RATE)
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
    prep = prepare()
    vocab = prep["vocab"]

    results = []
    for layers, hidden in config.GRID:
        print(f"\n=== 训练 {config.model_id(layers, hidden)} ===")
        results.append(train_one(layers, hidden, prep))

    # 排序与「最优配置」一律按**验证集**：测试集只负责报告，不参与任何选择。
    #
    # 若改用 test_acc 排序，即使每个配置各自都没作弊，「六选一」的最优值也会带上
    # 选择性偏差（winner's curse）——报告里的头条数字随之系统性偏乐观。
    # 各配置的 test_acc 照常写入 meta 并打印，只是不再驱动排序。
    # 不要「顺手」改回按 test_acc 排。
    ranked = sorted(results, key=lambda r: r["val_acc"], reverse=True)

    meta = {
        "dataset": corpus_label(),
        "train_size": prep["train_size"],
        "test_size": prep["test_size"],
        "vocab_size": len(vocab) + config.FIRST_WORD_IDX,
        "max_len": config.MAX_LEN,
        "models": ranked,
    }
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    config.META_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n元数据已保存：{config.META_PATH}")

    best = ranked[0]
    print(f"最优配置（按验证集选出）：{best['model_id']}  "
          f"验证 {best['val_acc']:.4f} / 测试 {best['test_acc']:.4f}")
    print("注：排序依据为验证准确率，测试准确率仅作报告，不参与选择。")
    return meta


def write_meta_for_single(metrics: dict, prep: dict) -> None:
    """单配置训练后，把结果并入 meta.json（已存在同 id 则替换）。"""
    if config.META_PATH.exists():
        meta = json.loads(config.META_PATH.read_text(encoding="utf-8"))
    else:
        meta = {
            "dataset": corpus_label(),
            "train_size": prep["train_size"],
            "test_size": prep["test_size"],
            "vocab_size": len(prep["vocab"]) + config.FIRST_WORD_IDX,
            "max_len": config.MAX_LEN,
            "models": [],
        }
    # 与 run_grid 同一条规则：meta 里 models 一律按**验证**准确率降序，
    # 测试准确率只随行输出、不参与排序。两个写入点都保持该不变式，
    # 否则「哪个入口写的」会决定数组顺序。
    meta["models"] = sorted(
        [m for m in meta["models"] if m["model_id"] != metrics["model_id"]] + [metrics],
        key=lambda r: r["val_acc"],
        reverse=True,
    )
    config.META_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"元数据已更新：{config.META_PATH}")


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

    # prep 在这里建一次、传下去：见 prepare() 的说明（合并语料下不能每个配置重建）
    if args.config:
        layers, hidden = args.config
        if (layers, hidden) not in config.GRID:
            raise SystemExit(f"配置 {layers}/{hidden} 不在 GRID 中：{config.GRID}")
        prep = prepare()
        write_meta_for_single(train_one(layers, hidden, prep), prep)
        return

    run_grid()


if __name__ == "__main__":
    main()
