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

# 用于「explain 与 predict 必须一致」的长文本夹具：前 100 词是中性填充，
# 极性只出现在第 101 词之后。这样「截 100 词」与「截 max_len 词」会给出
# **相反**的结论，缺陷才暴露得出来。
#
# 实测（lstm_l1_h256）：共 107 词，截 100 词 p=0.036347→消极、
# 截 max_len 词 p=0.996449→积极，标签直接相反。
#
# 刻意不用「同一句话重复 N 遍」当夹具：实测那种文本两种截断长度只差 1.2e-06，
# 标签也相同，用它对「截断长度不一致」这个缺陷几乎没有分辨力——
# 这正是本缺陷长期潜伏的原因。
LONG_TEXT = "酒店 " * 100 + "很好 非常满意 强烈推荐 很干净"


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


def test_explain_contributions_preserve_sentence_order(predictor):
    """contributions 必须与原句词序一致——前端要靠它渲染高亮句子。

    排序客户端可做（免费），从乱序还原原句序不可能（重新分词要 jieba，只在服务端）。
    故服务端返回的顺序必须是保序的。

    本条覆盖「文本短于两条截断上限」的情形：此时 predict 与 explain 看到的是
    同一个词列表，两者应当**全等**。长于 CONTRIB_MAX_TOKENS 的前缀情形
    由下一条测试专门覆盖——不要在这里用断言放宽来「兼容」两者。
    """
    text = "这家酒店位置很好，但房间隔音太差"
    tokens = predictor.predict(text)["tokens"]
    # 前提断言：把「本条只覆盖短文本」写成可执行的条件，
    # 将来有人换成长文本时这里会响亮失败，而不是悄悄比较两个不同的列表。
    assert len(tokens) <= config.CONTRIB_MAX_TOKENS, (
        f"本测试前提是文本不超过 {config.CONTRIB_MAX_TOKENS} 词，实测 {len(tokens)} 词；"
        f"超限时 explain 只分析前缀，应改用下面的前缀测试。"
    )

    contribs = predictor.explain(text)["contributions"]
    assert [c["token"] for c in contribs] == tokens


def test_explain_contributions_are_prefix_when_text_over_limit(predictor):
    """文本长于 CONTRIB_MAX_TOKENS 时，contributions 是 predict 词列表的**前缀**。

    explain 的分析上限是 CONTRIB_MAX_TOKENS、predict 的截断上限是 max_len，
    当前配置下 100 < 150，故超限文本的尾部词根本不参与分析——这是有意的
    复杂度控制（O(N) 次前向），不是丢词。把这条差异钉死，避免有人为了
    「让两边全等」而把 explain 的上限抬到 max_len、悄悄把代价乘上 1.5 倍。
    """
    text = "这家酒店位置很好，但房间隔音太差，" * 40
    tokens = predictor.predict(text)["tokens"]
    assert len(tokens) > config.CONTRIB_MAX_TOKENS, "前提：文本必须超过分析上限"

    contribs = predictor.explain(text)["contributions"]
    n = min(len(tokens), config.CONTRIB_MAX_TOKENS)
    assert [c["token"] for c in contribs] == tokens[:n]


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


@pytest.mark.parametrize(
    "text, expect_over_limit",
    [
        ("这家酒店位置很好，但房间隔音太差", False),
        (LONG_TEXT, True),
    ],
    ids=["短文本", "超100词长文本"],
)
def test_explain_agrees_with_predict(predictor, text, expect_over_limit):
    """同一输入，explain 与 predict 的结论必须完全一致。

    spec §6.2 规定开启逐词贡献时前端**只**调 /api/explain（不再调 /api/predict），
    所以 explain 若给出与 predict 不同的概率或标签，用户看到的就是
    与「单条预测」自相矛盾的答案，前端无从解释、也无从修正。

    实测 ChnSentiCorp 测试集 1200 条：135 条（11.25%）超过分析上限，
    修复前其中 134 条的 prob_pos 与 predict 不同、14 条 label 直接相反。

    长文本这一路是**关键**：短文本下两条路径恰好等价，看不出任何问题。
    """
    single = predictor.predict(text)
    explained = predictor.explain(text)

    # 前提写成可执行断言：长文本用例必须真的超过分析上限，
    # 否则它与短文本用例重复，这条参数化对回归完全无感。
    assert (len(single["tokens"]) > config.CONTRIB_MAX_TOKENS) is expect_over_limit, (
        f"用例前提被破坏：实测 {len(single['tokens'])} 词，"
        f"期望「超过上限」={expect_over_limit}。请调整文本长度，不要放宽断言。"
    )

    if expect_over_limit:
        # 夹具的**分辨力**也要断言：它必须让「截 CONTRIB_MAX_TOKENS 词」与
        # 「截 max_len 词」给出不同结论，否则即使 explain 又退回用短序列，
        # 这条用例也测不出来（重复句夹具就是这样失效的）。
        # 这里只走 predict 的公开接口，与 explain 的实现无关。
        truncated = predictor.predict(text, max_len=config.CONTRIB_MAX_TOKENS)
        assert truncated["label"] != single["label"], (
            "夹具失去分辨力：截到 CONTRIB_MAX_TOKENS 与截到 max_len 得到相同标签，"
            "本用例对「explain 用了更短的输入」已无感觉。请更换长文本夹具。"
        )

    # 先断结论再断附加字段：结论不一致是更严重的缺陷，让它先报错。
    assert explained["label"] == single["label"], (
        f"label 不一致：explain={explained['label']} predict={single['label']}"
    )
    assert explained["prob_pos"] == single["prob_pos"]
    assert explained["prob_neg"] == single["prob_neg"]

    # explain 必须自带完整词表：前端靠它渲染高亮句子，而它已不再调 /api/predict。
    assert explained["tokens"] == single["tokens"]

    # contributions 的上限是前向次数预算，不能随之放开。
    assert len(explained["contributions"]) <= config.CONTRIB_MAX_TOKENS


def test_vocab_size_mismatch_raises(predictor, monkeypatch):
    """词表与 checkpoint 不同源时必须报错，而不是静默把准确率拉低。"""
    monkeypatch.setattr(data, "load_vocab", lambda: {"好": config.FIRST_WORD_IDX})
    with pytest.raises(ValueError, match="词表"):
        infer.Predictor(predictor.model_id)


def test_unknown_model_raises(predictor):
    with pytest.raises((FileNotFoundError, ValueError)):
        infer.Predictor("不存在的模型")
