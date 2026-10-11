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
