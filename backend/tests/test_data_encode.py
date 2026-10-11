# -*- coding: utf-8 -*-
"""分词、词表与编码的测试（纯函数，不依赖真实语料）。

输入语料按 jieba 0.42.1 的实测切分结果核对过：**「很好」不是 jieba 的原子词**
（实测切成 ['很', '好']），凡是拿它当词表键的用例，键根本不可能被命中，
断言会以「看起来像实现错了」的方式失败。故词表键统一改用实测确实成词的
「酒店 / 房间」，断言本身未做任何放宽。
"""

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
    """高频词应拿到更小的索引，便于按频次截断词表。

    三个词的出现次数刻意拉开且都 >= min_freq：若沿用「甲 甲 甲 乙 乙 丙」，
    只出现 1 次的「丙」会先被 MIN_FREQ 滤掉，vocab["丙"] 直接 KeyError——
    那测到的就是「过滤」而非本用例要测的「排序」了。
    """
    vocab = data.build_vocab(["甲 甲 甲 甲 乙 乙 乙 丙 丙"])
    assert vocab["甲"] < vocab["乙"] < vocab["丙"]


def test_encode_shapes_and_lengths():
    word2idx = {"酒店": 2, "房间": 3}
    ids, lengths = data.encode(["酒店 房间", "酒店"], word2idx, max_len=5)

    assert ids.shape == (2, 5)
    assert ids.dtype == np.int64
    assert lengths.tolist() == [2, 1]


def test_encode_pads_with_zero():
    word2idx = {"酒店": 2}
    ids, _ = data.encode(["酒店"], word2idx, max_len=4)
    assert ids[0].tolist() == [2, 0, 0, 0]


def test_encode_unknown_word_maps_to_unk():
    word2idx = {"酒店": 2}
    ids, lengths = data.encode(["酒店 咕噜噜"], word2idx, max_len=4)
    assert ids[0][1] == config.UNK_IDX
    assert lengths[0] == 2


def test_encode_truncates_to_max_len():
    word2idx = {"酒店": 2}
    ids, lengths = data.encode(["酒店 酒店 酒店 酒店"], word2idx, max_len=2)
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
    vocab = {"酒店": 2, "房间": 3}
    data.save_vocab(vocab)
    assert data.load_vocab() == vocab
