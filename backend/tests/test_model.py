# -*- coding: utf-8 -*-
"""LSTM 模型结构与形状的测试。

全部用例都只搭随机初始化的小模型（词表 20、隐藏 8），不碰真实语料与训练，
保证秒级反馈。张量索引一律取自 config（PAD_IDX / FIRST_WORD_IDX / OUTPUT_DIM），
避免测试自己重钉一遍 config 拥有的取值。
"""

import torch

from senti import config, model

# 测试用词表规模：够容纳 <pad>、<unk> 与一批实词即可。
# 与 _build 的默认值共用，避免出现「词表 20 却从 20 里取索引」这类隐性错配。
VOCAB_SIZE = 20


def _build(vocab_size=VOCAB_SIZE, hidden_dim=8, num_layers=1):
    torch.manual_seed(0)
    return model.SentimentLSTM(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=num_layers)


def test_forward_output_shape():
    m = _build()
    # 索引从 FIRST_WORD_IDX 起，保证喂进去的都是实词而非 <pad>/<unk>
    x = torch.randint(config.FIRST_WORD_IDX, VOCAB_SIZE, (7, 4))  # (seq_len=7, batch=4)
    lengths = torch.tensor([7, 5, 3, 1])
    out = m(x, lengths)
    assert out.shape == (4, config.OUTPUT_DIM)


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

    # 前 3 位是同一批真实词；其后分别填「不同的非零词」与 <pad>
    ids_a = torch.tensor([[2, 3, 4, 7, 9]]).T   # (5, 1)
    ids_b = torch.tensor([[2, 3, 4, config.PAD_IDX, config.PAD_IDX]]).T

    with torch.no_grad():
        out_a = m(ids_a, lengths)
        out_b = m(ids_b, lengths)

    assert torch.allclose(out_a, out_b, atol=1e-6), (
        "真实长度之后的词位影响了输出，说明取的不是 lengths-1 位的隐藏状态"
    )


def test_padding_index_excluded_from_gradient():
    """<pad> 不参与梯度，否则填充位会学到语义并污染嵌入表。"""
    m = _build()
    assert m.embeddings.padding_idx == config.PAD_IDX
    x = torch.tensor([[2, 3, config.PAD_IDX, config.PAD_IDX]]).T
    out = m(x, torch.tensor([2]))
    out.sum().backward()
    grad = m.embeddings.weight.grad
    assert grad is not None
    assert torch.all(grad[config.PAD_IDX] == 0), "padding_idx 的梯度应为 0"


def test_output_is_logits_not_probabilities():
    """返回的必须是 logits（供 CrossEntropyLoss 使用），不能提前做 softmax。"""
    m = _build().eval()
    x = torch.randint(config.FIRST_WORD_IDX, VOCAB_SIZE, (4, 3))
    with torch.no_grad():
        out = m(x, torch.tensor([4, 4, 4]))
    row_sums = out.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), (
        "输出行和为 1，疑似已经 softmax；应返回 logits"
    )


def test_count_parameters_covers_all_trainable_tensors():
    """用闭式公式独立算一遍参数量。

    这是「层数/维度 → 参数量」对照表与前端展示的唯一计算路径。
    公式漏项或多算都不会抛异常，只会静默报出错误的模型规模，
    所以这里按 nn.LSTM 的张量布局（每层 4 组门 + 两组偏置）手工推导一遍。
    """
    vocab_size, hidden_dim, num_layers = VOCAB_SIZE, 8, 2
    m = _build(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=num_layers)

    # Embedding 权重是 (词表 × 词向量维度) 的矩形，<pad> 行照样占位
    embedding = vocab_size * config.EMBEDDING_DIM
    # 每层 LSTM：4 组门 × (输入→隐藏 + 隐藏→隐藏) + 两组长度 4×hidden 的偏置
    lstm = sum(
        4 * hidden_dim * (config.EMBEDDING_DIM if layer == 0 else hidden_dim)
        + 4 * hidden_dim * hidden_dim
        + 8 * hidden_dim
        for layer in range(num_layers)
    )
    # 输出层：权重 + 偏置
    linear = hidden_dim * config.OUTPUT_DIM + config.OUTPUT_DIM

    assert model.count_parameters(m) == embedding + lstm + linear
