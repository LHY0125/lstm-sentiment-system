# -*- coding: utf-8 -*-
"""推理层测试。依赖已训练好的 checkpoint（Task 5 的产物）。"""

import pytest

from senti import config, data, infer

# 逐词贡献方向测试所用的混合情感评论。
#
# 这里刻意**不用**计划书原稿的 "非常好 非常好 非常好 酒店 位置 很 差"：
#   1. jieba 不把「非常好」切成一个词，实测是 ['非常', '好']，
#      于是原稿里的 deltas.get("非常好", 0) 取不到键、静默返回默认值 0，
#      断言退化成「差的 delta > 0」，等于什么都没验。
#   2. 原稿字符串里「好」出现 3 次，而 {token: delta} 字典会保留最后一次写入；
#      贡献列表又按 |delta| 降序排，于是留下的恰好是「最弱的那一次」出现。
#
# 采用计划书 Step 5 人工检视用的那条评论，理由是信号强度：
# 该模型对「位置很好但隔音太差」这类短句会输出接近 0 的积极概率（实测 0.0006），
# softmax 已饱和，所有 delta 被压缩到小数点后第四位（好 −0.0005 / 差 +0.0039），
# 方向对但余量只有 0.004，任何微小的数值扰动都可能让这条断言翻面。
# 下面这条评论的积极概率是 0.4491、未饱和，实测余量 0.516（好 −0.364 / 差 +0.152）。
EXPLAIN_TEXT = "这家酒店位置很好，离地铁站近，但房间隔音太差，晚上吵得睡不着"


@pytest.fixture(scope="module")
def predictor():
    models = infer.list_models()
    if not models:
        pytest.skip("尚无 checkpoint，先运行：python -m senti.train")
    # 选「最好模型」一律按**验证集**：测试集不参与任何选择，否则最优值会带上
    # 选择性偏差（winner's curse）。test_acc 仍随响应返回，只是不驱动排序。
    best = max(models, key=lambda m: m["val_acc"])
    return infer.get_predictor(best["id"])


def test_list_models_reads_meta():
    models = infer.list_models()
    assert models, "meta.json 里应当至少有一个模型"
    for m in models:
        assert {"id", "layers", "hidden_dim", "val_acc", "test_acc", "params"} <= set(m)


def test_list_models_sorted_by_val_acc():
    """排序依据是验证准确率，不是测试准确率。"""
    accs = [m["val_acc"] for m in infer.list_models()]
    assert accs == sorted(accs, reverse=True), (
        "list_models 必须按 val_acc 降序返回；测试准确率只作报告，不参与选择。"
    )


def test_predict_probabilities_sum_to_one(predictor):
    r = predictor.predict("这个酒店非常好，服务很周到")
    assert abs(r["prob_pos"] + r["prob_neg"] - 1.0) < 1e-6


def test_predict_returns_label_and_tokens(predictor):
    r = predictor.predict("房间很干净")
    assert r["label"] in (0, 1)
    assert isinstance(r["tokens"], list) and r["tokens"]


def test_predict_positive_and_negative_differ(predictor):
    """明显褒贬的两句话应给出不同倾向，否则模型实际没学到极性。"""
    good = predictor.predict("非常好，非常满意，强烈推荐")
    bad = predictor.predict("太差了，非常失望，再也不来了")
    assert good["prob_pos"] > bad["prob_pos"]


def test_batch_predict_length_matches_input(predictor):
    texts = ["很好", "很差", "一般"]
    out = predictor.batch_predict(texts)
    assert len(out) == len(texts)
    assert [o["text"] for o in out] == texts


def test_batch_matches_single(predictor):
    """批量与单条结果必须一致，否则说明批处理路径有 bug。"""
    texts = ["服务很好", "房间很脏"]
    batch = predictor.batch_predict(texts)
    for item in batch:
        single = predictor.predict(item["text"])
        assert abs(single["prob_pos"] - item["prob_pos"]) < 1e-4


def test_explain_returns_contributions_for_each_token(predictor):
    r = predictor.explain("这个酒店非常好")
    assert r["contributions"]
    n_tokens = len(predictor.predict("这个酒店非常好")["tokens"])
    assert len(r["contributions"]) == n_tokens


def test_explain_direction_is_correct(predictor):
    """遮挡强正向词后，积极概率应当下降 → delta < 0。

    delta 定义为「遮挡该词后的积极概率 − 原始积极概率」，
    故正向词 delta < 0、负向词 delta > 0。

    被测词必须先断言「确实出现在分词结果里」：原稿写法
    `deltas.get("非常好", 0)` 在键不存在时静默返回 0，
    会让这条断言在什么都没测到的情况下通过。
    """
    tokens = data.tokenize(EXPLAIN_TEXT)
    r = predictor.explain(EXPLAIN_TEXT)
    deltas = {c["token"]: c["delta"] for c in r["contributions"]}

    for token in ("好", "差"):
        assert token in tokens, f"{token!r} 不在分词结果中：{tokens}"
        assert token in deltas, f"{token!r} 不在贡献结果中：{sorted(deltas)}"

    assert deltas["差"] > deltas["好"], (
        f"负向词的 delta 应大于正向词，实测 差={deltas['差']:+.4f} "
        f"好={deltas['好']:+.4f}（分词：{tokens}）"
    )


def test_explain_respects_token_limit(predictor):
    long_text = "很好 " * 300
    r = predictor.explain(long_text)
    assert len(r["contributions"]) <= config.CONTRIB_MAX_TOKENS


def test_vocab_size_mismatch_raises(predictor, monkeypatch):
    """词表与 checkpoint 不同源时必须报错，而不是静默把准确率拉低。"""
    monkeypatch.setattr(data, "load_vocab", lambda: {"好": config.FIRST_WORD_IDX})
    with pytest.raises(ValueError, match="词表"):
        infer.Predictor(predictor.model_id)


def test_unknown_model_raises(predictor):
    with pytest.raises((FileNotFoundError, ValueError)):
        infer.Predictor("不存在的模型")
