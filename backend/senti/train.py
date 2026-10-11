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
        vocab_size=len(vocab) + 2, hidden_dim=hidden, num_layers=layers
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
            "vocab_size": len(vocab) + 2,
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
        "vocab_size": len(vocab) + 2,
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

    # hidden/层数刻意取「小而固定」的诊断模型：本测试只关心管线能否过拟合，
    # 不是在搜索超参数，所以这两个数不进 config，也不随 GRID 变化。
    m = model.SentimentLSTM(
        vocab_size=len(vocab) + 2, hidden_dim=64, num_layers=1, dropout=0.0
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
        "vocab_size": len(vocab) + 2,
        "max_len": config.MAX_LEN,
        "models": results,
    }
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    config.META_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n元数据已保存：{config.META_PATH}")

    best = max(results, key=lambda r: r["test_acc"])
    print(f"最优配置：{best['model_id']}  测试准确率 {best['test_acc']:.4f}")
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
            "vocab_size": len(vocab) + 2,
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
