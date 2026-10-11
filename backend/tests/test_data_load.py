# -*- coding: utf-8 -*-
"""语料获取与解析的测试。

这些测试依赖真实语料，未下载时会失败——这是有意为之：
数据可得性是本方案的核心风险，必须显式暴露而不是静默跳过。
"""

import pytest
import requests

from senti import config, data


@pytest.fixture(scope="module")
def corpus():
    """确保语料已下载，返回 (train_texts, train_labels, test_texts, test_labels)。"""
    if not data.is_downloaded():
        data.download()
    return data.load_split()


class _FakeResponse:
    """最小化的假响应，只实现 _fetch_validated 用到的接口。

    用来构造「HTTP 200 但内容根本不是语料」这种真实网络里难以稳定复现的输入。
    """

    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status_code = status

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def iter_content(self, chunk_size: int = 1):
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start:start + chunk_size]


# ------------------------------------------------------------------ 语料内容

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


def test_uses_native_train_test_split(corpus):
    """R3a：必须沿用文件自带的划分，不得合并后重切或重排。"""
    train_texts, _, test_texts, _ = corpus
    native_train = data._read_rows(data.train_path())
    native_test = data._read_rows(data.test_path())
    assert len(train_texts) == len(native_train)
    assert len(test_texts) == len(native_test)
    assert train_texts[:5] == [t for t, _ in native_train[:5]]
    assert test_texts[:5] == [t for t, _ in native_test[:5]]


# ------------------------------------------------------------------ 标签解析

def test_label_parsing_accepts_known_encodings():
    """0/1 与 -1/1 两种编码、以及 "1.0" 这类数值写法都要能归一化。"""
    assert data._parse_label(1, 0) == 1
    assert data._parse_label(0, 0) == 0
    assert data._parse_label("1", 0) == 1
    assert data._parse_label("0", 0) == 0
    assert data._parse_label("-1", 0) == 0
    assert data._parse_label("1.0", 0) == 1
    assert data._parse_label(" 0.0 ", 0) == 0


@pytest.mark.parametrize(
    "bad",
    [
        "2", "-2", "positive", "", "nan", None, "1e9",
        # 小数必须被拒：若先取整再校验，1.5 会被截断成 1、0.5 变成 0，
        # 截断值恰好落在白名单里，等于绕过校验又变回静默错标。
        "1.5", "0.5", "-0.5", "1e-9",
    ],
)
def test_unknown_label_raises(bad):
    """标签映射必须失败关闭：把未知取值静默当成积极会持续污染训练集。"""
    with pytest.raises(ValueError, match="标签"):
        data._parse_label(bad, 7)


def test_unknown_label_error_names_the_value_and_record():
    """报错信息要能直接定位到问题记录，否则 9 600 条里找不出那一条。"""
    with pytest.raises(ValueError) as excinfo:
        data._parse_label("positive", 42)
    assert "42" in str(excinfo.value)
    assert "positive" in str(excinfo.value)


# ------------------------------------------------------------------ 下载校验

def test_download_rejects_non_corpus_payload(tmp_path, monkeypatch):
    """HTTP 200 的 HTML 错误页绝不能被当成语料写盘。"""
    html = b"<!DOCTYPE html><html><body>404 Not Found</body></html>"
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: _FakeResponse(html))

    dest = tmp_path / "train.parquet"
    with pytest.raises(ValueError):
        data._fetch_validated("https://example.invalid/train.parquet", dest)

    assert not dest.exists(), "校验失败的内容被写进了目标路径"
    assert not data._tmp_path(dest).exists(), "临时文件没有清理干净"


def test_download_rejects_oversized_payload(tmp_path, monkeypatch):
    """超过字节上限的响应必须被拒，避免异常内容写满磁盘。"""
    monkeypatch.setattr(data, "MAX_CORPUS_BYTES", 1024)
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: _FakeResponse(b"x" * 4096))

    dest = tmp_path / "test.parquet"
    with pytest.raises(ValueError, match="上限"):
        data._fetch_validated("https://example.invalid/test.parquet", dest)

    assert not dest.exists()
    assert not data._tmp_path(dest).exists()


def test_download_rejects_empty_payload(tmp_path, monkeypatch):
    """空响应同样不能落盘。"""
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: _FakeResponse(b""))

    dest = tmp_path / "test.parquet"
    with pytest.raises(ValueError, match="为空"):
        data._fetch_validated("https://example.invalid/test.parquet", dest)

    assert not dest.exists()
    assert not data._tmp_path(dest).exists()
