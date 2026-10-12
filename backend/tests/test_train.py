# -*- coding: utf-8 -*-
"""训练层的轻量测试。

训练是数值实验，这里只测「可测的东西」：
工具函数的行为、划分的无泄漏性、以及小样本过拟合这一管线正确性判据。
完整网格训练不在单元测试中跑（分钟级，见 README 的实测流程）。
"""

import numpy as np
import pytest
import torch
from torch import nn

from senti import config, corpora, data, train


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

    这里**显式传入单语料的划分**：过拟合测试验的是「词表 → 编码 → 取末词」这条链路，
    与用哪套语料无关；而默认的合并语料有 184 万条，为一个 40 条的诊断翻一遍是分钟级的。
    """
    acc = train.overfit_test(n=40, epochs=30, load=data.load_split)
    assert acc >= 0.9, f"小样本过拟合仅 {acc:.2%}，管线可能有问题"


# ---------------------------------------------------------------- 指标函数
#
# evaluate_full 是唯一决定「报告的三个数字怎么算」的函数，却一度零覆盖：
# 套件里没有任何测试调用它，于是把 balanced_acc 改成 acc、把 majority_baseline
# 改成常数 0.5，149 条测试照样全绿。下面用**桩模型**构造已知混淆矩阵，
# 把三个数的算式逐个钉死。


class _StubModel(nn.Module):
    """不做任何计算的桩模型：把「期望的预测类别」编码在序列首列。

    `evaluate_full` 只依赖 `m(x, lengths) -> logits`，与模型内部结构无关。
    换成桩模型就能把混淆矩阵设计成任意形状，从而覆盖指标算式的每个分支；
    用真模型则只能被动接受它碰巧给出的那个矩阵。
    """

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: [T, B]；x[0] 是每条样本的期望预测类别（0 或 1）
        first = x[0].float()
        return torch.stack([1.0 - first, first], dim=1) * 10.0


def _confusion_loader(
    neg_total: int, neg_correct: int, pos_total: int, pos_correct: int, batch_size: int
):
    """构造一个混淆矩阵可指定的 DataLoader。

    消极类：前 `neg_correct` 条判对，其余判成积极（错）。
    积极类：前 `pos_correct` 条判对，其余判成消极（错）。
    """
    pred = (
        [0] * neg_correct + [1] * (neg_total - neg_correct)
        + [1] * pos_correct + [0] * (pos_total - pos_correct)
    )
    labels = [0] * neg_total + [1] * pos_total
    n = len(labels)
    token_count = 3
    ids = np.zeros((n, token_count), dtype=np.int64)
    ids[:, 0] = pred                      # 首列即桩模型读到的「预测」
    lengths = np.full(n, token_count, dtype=np.int64)
    return train.make_loader(ids, lengths, np.array(labels, dtype=np.int64), batch_size=batch_size)


def test_evaluate_full_reports_three_distinct_skew_aware_metrics():
    """准确率、平衡准确率、多数类基线必须是三个**互不相同**的数。

    只断言「0 <= x <= 1」是没用的——那对任何实现都成立，也正是这个缺口当初
    骗过了整套测试。这里构造已知混淆矩阵：10 条消极全判对、90 条积极判对 70 条。

        准确率      = 80/100        = 0.80
        平衡准确率  = (1.0 + 70/90)/2 ≈ 0.8889   ← 两类召回率的均值
        多数类基线  = 90/100        = 0.90

    把 `balanced_acc` 写成准确率、或把 `majority_baseline` 写成常数，本条都会失败。
    分批（16）而非一次装入，顺带钉住「跨 batch 累加」是否正确。
    """
    loader = _confusion_loader(
        neg_total=10, neg_correct=10, pos_total=90, pos_correct=70, batch_size=16
    )
    r = train.evaluate_full(_StubModel(), loader, torch.device("cpu"))

    assert r["class_total"] == [10, 90]
    assert r["acc"] == pytest.approx(0.80)
    assert r["balanced_acc"] == pytest.approx((1.0 + 70 / 90) / 2)
    assert r["majority_baseline"] == pytest.approx(0.90)

    # 三者两两不同——否则「三个数要一起读」这句说明就是空话
    assert r["balanced_acc"] != pytest.approx(r["acc"])
    assert r["majority_baseline"] != pytest.approx(r["balanced_acc"])


def test_evaluate_full_handles_a_missing_class_without_dividing_by_zero():
    """某一类一条都没有时不能除零；平衡准确率退化为在场那一类的召回率。

    单语料 profile 下测试集接近平衡，但合并语料重切后完全可能出现某类为 0
    （例如按语料分层切分时）。这里显式构造只有积极类的输入。
    """
    loader = _confusion_loader(
        neg_total=0, neg_correct=0, pos_total=90, pos_correct=70, batch_size=32
    )
    r = train.evaluate_full(_StubModel(), loader, torch.device("cpu"))

    assert r["class_total"] == [0, 90]
    assert r["balanced_acc"] == pytest.approx(70 / 90)
    assert r["majority_baseline"] == pytest.approx(1.0)


def test_prepare_encodes_both_splits_deterministically(monkeypatch):
    """`prepare()` 把「载入划分 → 建词表 → 编码」一次做完，结果必须可复现。

    它是本轮新增的复用入口（六组配置共用一份编码结果），此前无任何测试触达。
    这里把语料来源换成 5 条合成文本，避免为一个结构测试去翻 184 万条真实语料；
    同时显式 `save_vocab=False`，防止把真词表覆盖掉。
    """
    texts = ["很好", "很差", "一般", "很棒", "无聊"]
    labels = [1, 0, 0, 1, 0]
    monkeypatch.setattr(
        train.corpora, "load_training_split", lambda: (texts, labels, texts, labels)
    )

    first = train.prepare(save_vocab=False)
    second = train.prepare(save_vocab=False)

    assert first["train_size"] == len(texts)
    assert first["test_size"] == len(texts)
    assert first["tr_ids"].shape == first["te_ids"].shape
    assert first["tr_labels"].tolist() == labels
    assert first["te_labels"].tolist() == labels
    # 同一输入两次编码必须逐位相同（词表构建与编码都不能引入随机性）
    np.testing.assert_array_equal(first["tr_ids"], second["tr_ids"])
