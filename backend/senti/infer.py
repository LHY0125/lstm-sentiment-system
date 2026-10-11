# -*- coding: utf-8 -*-
"""推理层：单条预测、批量预测与逐词贡献分析。

只消费训练产物（checkpoint + 词表 + 元数据），不做任何训练。
「模型可切换」在本层等价于「换一个 checkpoint 文件」。
"""

import json

import numpy as np
import torch

from senti import config, data, model, train

# 后端给出 label 时使用的默认阈值。
# 注意：阈值是「概率 → 标签」的纯映射，最终判定由前端完成，
# 本字段只作参考，不参与任何计算。
DEFAULT_THRESHOLD = 0.5


def list_models() -> list[dict]:
    """读取 meta.json，返回可用模型列表（按 val_acc 降序）。

    排序依据刻意是**验证**准确率而非测试准确率：测试集不参与任何选择，
    否则「六选一」的最优值会带上选择性偏差（winner's curse）。
    test_acc 照常随行返回，供报告与前端展示，只是不驱动排序。

    字段一律用直接取键而非 .get(...)：meta.json 缺字段属于产物损坏，
    应当立刻炸掉；用默认值兜底会让「所有模型并列 0 分」这种错看起来很正常。
    """
    if not config.META_PATH.exists():
        return []
    meta = json.loads(config.META_PATH.read_text(encoding="utf-8"))
    models = [
        {
            "id": m["model_id"],
            "layers": m["layers"],
            "hidden_dim": m["hidden_dim"],
            "embedding_dim": m["embedding_dim"],
            "val_acc": m["val_acc"],
            "test_acc": m["test_acc"],
            "params": m["params"],
            "train_time_sec": m["train_time_sec"],
        }
        for m in meta.get("models", [])
    ]
    return sorted(models, key=lambda m: m["val_acc"], reverse=True)


class Predictor:
    """持有「模型 + 词表」配对，提供推理能力。

    词表与 checkpoint 必须同源：载入时断言 vocab_size 一致，
    否则训练期与推理期用了不同词表，准确率会静默下降而不报错。
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.device = train.device()

        path = config.ckpt_path(model_id)
        if not path.exists():
            raise FileNotFoundError(
                f"找不到模型 {model_id} 的 checkpoint：{path}\n"
                f"可用模型：{[m['id'] for m in list_models()]}"
            )

        self.vocab = data.load_vocab()
        # weights_only=True：checkpoint 只含 str/int/张量字典，无需反序列化任意对象。
        # torch 2.6 起这已是默认值，显式写出以防将来被改回。
        ckpt = torch.load(path, map_location=self.device, weights_only=True)

        # 词表错位的表现是「准确率悄悄掉一截」而不是报错，所以必须在这里挡住。
        expected = len(self.vocab) + config.FIRST_WORD_IDX
        if ckpt["vocab_size"] != expected:
            raise ValueError(
                f"词表与模型不匹配：checkpoint 记录 vocab_size={ckpt['vocab_size']}，"
                f"当前词表为 {len(self.vocab)} 个词（加特殊位应为 {expected}）。\n"
                f"两者必须同源，否则索引整体错位、准确率静默下降。"
                f"请重新运行 `python -m senti.train` 重建词表与 checkpoint。"
            )

        self.model = model.SentimentLSTM(
            vocab_size=ckpt["vocab_size"],
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["layers"],
            embedding_dim=ckpt["embedding_dim"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.max_len = ckpt.get("max_len", config.MAX_LEN)

    # ------------------------------------------------------------ 内部

    @torch.no_grad()
    def _probs_for_ids(self, ids: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """索引矩阵 → 积极类概率。

        参数：
            ids     (batch, max_len) 词索引矩阵
            lengths (batch,) 每条样本的**真实词数**（不含填充）

        返回：
            (batch,) 积极类概率
        """
        x = torch.from_numpy(ids).long().transpose(1, 0).contiguous().to(self.device)
        logits = self.model(x, torch.from_numpy(lengths).long().to(self.device))
        return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    def _ids_for_tokens(self, tokens: list[str], max_len: int) -> tuple[np.ndarray, np.ndarray]:
        """词列表 → ((1, max_len) 索引矩阵, (1,) 真实词数)。

        词表外的词一律落到 config.UNK_IDX；不足 max_len 的部分在后面补
        config.PAD_IDX。用 np.full 而非 np.zeros：填充值必须取自 config，
        否则将来改填充值时这里会静默对不上。
        """
        ids = np.full((1, max_len), config.PAD_IDX, dtype=np.int64)
        seq = [self.vocab.get(w, config.UNK_IDX) for w in tokens[:max_len]]
        ids[0, : len(seq)] = seq
        # lengths 至少为 1，避免空文本在模型里 gather 到索引 -1
        lengths = np.array([max(len(seq), 1)], dtype=np.int64)
        return ids, lengths

    def _probs_for_batch(self, texts: list[str], max_len: int) -> np.ndarray:
        """批量推理，返回积极类概率数组。"""
        ids, lengths = data.encode(texts, self.vocab, max_len=max_len)
        return self._probs_for_ids(ids, lengths)

    # ------------------------------------------------------------ 对外

    def predict(self, text: str, max_len: int | None = None) -> dict:
        """单条预测。返回 {label, prob_pos, prob_neg, tokens}。

        max_len 省略时用 checkpoint 里记录的截断长度，保证推理与训练一致。
        """
        max_len = max_len or self.max_len
        tokens = data.tokenize(text)[:max_len]
        ids, lengths = self._ids_for_tokens(tokens, max_len)
        p = float(self._probs_for_ids(ids, lengths)[0])
        return {
            "label": 1 if p >= DEFAULT_THRESHOLD else 0,
            "prob_pos": p,
            "prob_neg": 1.0 - p,
            "tokens": tokens,
        }

    def batch_predict(self, texts: list[str], max_len: int | None = None) -> list[dict]:
        """批量预测。返回 [{text, prob_pos, prob_neg}]，顺序与输入一致。"""
        max_len = max_len or self.max_len
        # 按 batch_size 分批，避免超长输入一次性占满显存
        probs: list[float] = []
        for i in range(0, len(texts), config.BATCH_SIZE):
            chunk = texts[i : i + config.BATCH_SIZE]
            probs.extend(self._probs_for_batch(chunk, max_len).tolist())
        return [
            {"text": t, "prob_pos": p, "prob_neg": 1.0 - p}
            for t, p in zip(texts, probs)
        ]

    def explain(self, text: str, max_len: int | None = None) -> dict:
        """逐词贡献分析（遮挡法）。

        对每个词位，把它替换为 <unk> 后重新前向，记录积极概率的变化：

            delta = 遮挡该词后的积极概率 − 原始积极概率

            delta < 0  遮住后积极概率下降 → 该词在**推高**积极概率（正向贡献）
            delta > 0  遮住后积极概率上升 → 该词在**压低**积极概率（负向贡献）

        前端的着色依赖这个符号约定：delta < 0 记正向、delta > 0 记负向。

        用遮挡法而非 attention 权重：注意力反映信息的路由，不等于重要性。
        遮挡法问的是反事实问题「缺了这个词，结果会变吗」，对任意模型都成立。

        遮挡**直接改写索引值**为 config.UNK_IDX，而不是先把词列表里的字符串换成
        "<unk>" 再走词表查表。后者只在「词表里恰好没有 <unk> 这个字面词」时才正确；
        一旦某个语料把 <unk> 收进词表，遮挡位会被解析成一个**真实**词的索引，
        贡献度静默算错、不报错、也无从察觉。写索引从根上消除这种碰撞。

        复杂度为 O(N) 次前向；参与分析的词数上限为 CONTRIB_MAX_TOKENS。
        """
        max_len = max_len or self.max_len
        tokens = data.tokenize(text)[: config.CONTRIB_MAX_TOKENS]

        base_ids, lengths = self._ids_for_tokens(tokens, max_len)
        base = float(self._probs_for_ids(base_ids, lengths)[0])

        # 只分析真正进入模型的位置。max_len 比 CONTRIB_MAX_TOKENS 更小时尾部词会被
        # 截断，给它们算贡献恒为 0，那等于谎称「这个词没有影响」。
        n = min(len(tokens), max_len)

        contributions = []
        for i in range(n):
            masked_ids = base_ids.copy()
            # 位置保持不变、只把该位换成 <unk>：这是「该词不可知」的反事实，
            # 与原始序列等长，故各词的 delta 之间可以直接比较。
            masked_ids[0, i] = config.UNK_IDX
            after = float(self._probs_for_ids(masked_ids, lengths)[0])
            contributions.append({"token": tokens[i], "delta": after - base})

        # 按贡献强度降序，前端只展示前若干条
        contributions.sort(key=lambda c: -abs(c["delta"]))
        return {
            "label": 1 if base >= DEFAULT_THRESHOLD else 0,
            "prob_pos": base,
            "prob_neg": 1.0 - base,
            "contributions": contributions,
        }


_PREDICTOR_CACHE: dict[str, Predictor] = {}


def get_predictor(model_id: str) -> Predictor:
    """按 id 取 Predictor，带进程内缓存。

    载入模型要读盘并重建网络，缓存避免每次请求都重载。
    """
    if model_id not in _PREDICTOR_CACHE:
        _PREDICTOR_CACHE[model_id] = Predictor(model_id)
    return _PREDICTOR_CACHE[model_id]
