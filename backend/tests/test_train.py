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
