# -*- coding: utf-8 -*-
"""模型层：LSTM 情感二分类模型。

结构沿用课程实验四的 Embedding + LSTM + Linear 三组件方案，
并保留其中一处关键修正（见 forward 的 docstring）。
"""

import torch
import torch.nn as nn

from senti import config


class SentimentLSTM(nn.Module):
    """Embedding + LSTM + Linear 三组件情感分类器。

        Embedding  词索引 → 连续向量（padding_idx=config.PAD_IDX，填充位不参与梯度更新）
        LSTM       向量序列 → 逐时刻的语义表示
        Linear     末时刻语义表示 → output_dim 维 logits（积极 / 消极）

    输出为 **logits** 而非概率：交叉熵损失内部自带 softmax，
    提前做 softmax 会造成数值不稳定与梯度衰减。
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        embedding_dim: int = config.EMBEDDING_DIM,
        dropout: float = config.DROPOUT,
        output_dim: int = config.OUTPUT_DIM,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=config.PAD_IDX)

        # num_layers == 1 时 PyTorch 会忽略 dropout 并给出警告，故显式置 0。
        # 这个 0.0 是语义上的「不启用 dropout」，不是可调超参数，所以不放进 config。
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=False,     # 输入形状为 (seq_len, batch)
            dropout=lstm_dropout,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数：
            x       (seq_len, batch) 词索引矩阵
            lengths (batch,) 每条样本的**真实词数**（不含填充）

        返回：
            (batch, output_dim) logits
        """
        embeds = self.embeddings(x)                # (seq_len, batch, embedding_dim)
        output, _ = self.lstm(embeds)              # (seq_len, batch, hidden_dim)

        # 取每条样本「最后一个真实词」对应的隐藏状态。
        #
        # 不能写成 output[-1]：那固定取第 seq_len 步，而评论普遍短于 MAX_LEN，
        # 末位是 <pad> 填充。实测中该写法会让模型前 5 轮几乎不学习。
        # 正确做法是按真实长度 gather。
        #
        # clamp(min=0) 是防御：lengths 理论上由 data.encode 保证 >=1，
        # 但若真出现 0，这里退化为取第 0 步而不是索引 -1（会越界到末位填充）。
        idx = (
            (lengths - 1)
            .clamp(min=0)
            .view(1, -1, 1)
            .expand(1, output.size(1), output.size(2))
        )
        last = output.gather(0, idx).squeeze(0)    # (batch, hidden_dim)

        return self.fc(last)                       # (batch, output_dim)


def count_parameters(m: nn.Module) -> int:
    """可训练参数量，用于 meta.json 与前端展示。"""
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
