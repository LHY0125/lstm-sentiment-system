# -*- coding: utf-8 -*-
"""接口层测试。用 Flask test client，无需起服务。"""

import pytest
from werkzeug.exceptions import RequestEntityTooLarge

from senti import api, config, infer

# 「explain 与 predict 必须一致」所用的短文本：词数远小于两条截断上限，
# 此时两条路径看到的是同一个词列表。
SHORT_TEXT = "这家酒店位置很好，但房间隔音太差"

# 同一条硬约束的长文本夹具：前 100 词是中性填充，极性只出现在第 101 词之后。
# 这样「截 100 词」与「截 max_len 词」会给出**相反**的结论，缺陷才暴露得出来。
#
# 刻意不用「同一句话重复 N 遍」当夹具：实测那种文本在两截断长度下只差 1.2e-06、
# 标签也相同，对「截断长度不一致」几乎没有分辨力——这正是该缺陷长期潜伏的原因。
#
# 实测（lstm_l1_h256）：共 107 词，截 100 词 p=0.036347→消极、
# 截 max_len 词 p=0.996449→积极，标签直接相反。
LONG_TEXT = "酒店 " * 100 + "很好 非常满意 强烈推荐 很干净"


@pytest.fixture(scope="module")
def client():
    models = infer.list_models()
    if not models:
        pytest.skip("尚无 checkpoint，先运行：python -m senti.train")
    app = api.create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(scope="module")
def best_model_id():
    """选「最好模型」一律按**验证集**。

    与 test_infer.py 的同款夹具同理：测试集不参与任何选择，
    按 test_acc 挑模型就是把测试集引入选择，最优值会带上选择性偏差
    （winner's curse）。test_acc 照常随 /api/models 返回，只是不驱动选择。
    """
    models = infer.list_models()
    return max(models, key=lambda m: m["val_acc"])["id"]


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_models_list(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    data = r.get_json()
    assert isinstance(data, list) and data
    assert {"id", "layers", "hidden_dim", "test_acc", "params"} <= set(data[0])


def test_models_list_preserves_val_acc_order(client):
    """接口层只做序列化，不得按 test_acc 重排。

    排序依据是**验证**准确率（infer.list_models 已保证）。接口层是第二个可能
    把顺序改掉的地方——一旦在这里按 test_acc 排，前端下拉框的默认选项就变成了
    「测试集上最好的那个」，选择性偏差从接口处漏回来。
    """
    accs = [m["val_acc"] for m in client.get("/api/models").get_json()]
    assert accs == sorted(accs, reverse=True), (
        "接口层改变了模型顺序；排序必须原样来自 infer.list_models（按 val_acc 降序）。"
    )


def test_predict_ok(client, best_model_id):
    r = client.post("/api/predict", json={"text": "这个酒店非常好", "model_id": best_model_id})
    assert r.status_code == 200
    d = r.get_json()
    assert abs(d["prob_pos"] + d["prob_neg"] - 1.0) < 1e-6
    assert d["label"] in (0, 1)
    assert isinstance(d["tokens"], list)


def test_predict_missing_text_returns_400(client, best_model_id):
    r = client.post("/api/predict", json={"model_id": best_model_id})
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_predict_empty_text_returns_400(client, best_model_id):
    r = client.post("/api/predict", json={"text": "   ", "model_id": best_model_id})
    assert r.status_code == 400


def test_predict_too_long_returns_400(client, best_model_id):
    r = client.post(
        "/api/predict",
        json={"text": "好" * (config.MAX_TEXT_CHARS + 1), "model_id": best_model_id},
    )
    assert r.status_code == 400


def test_predict_unknown_model_returns_404(client):
    """文本合法、模型不存在 → 404（而不是 400）。

    这条同时钉住了**校验顺序**：参数校验在模型校验之前。
    顺序反了会把「模型不存在」伪装成参数错误，前端无从提示用户换模型。
    两个字段都缺的情形由下一条覆盖（应当报 400 而非 404）。
    """
    r = client.post("/api/predict", json={"text": "很好", "model_id": "不存在"})
    assert r.status_code == 404
    assert "error" in r.get_json()


def test_predict_missing_both_reports_400_not_404(client):
    """校验顺序：两个字段都缺时报 400（参数错误）而非 404。

    「你少了 text」是用户能立刻修好的问题；报 404 会把人引向
    「模型列表坏了」这个错误方向。
    """
    r = client.post("/api/predict", json={})
    assert r.status_code == 400, r.get_json()


def test_explain_ok(client, best_model_id):
    r = client.post("/api/explain", json={"text": "房间很干净，服务也好", "model_id": best_model_id})
    assert r.status_code == 200
    d = r.get_json()
    assert d["contributions"]
    assert set(d["contributions"][0]) == {"token", "delta"}


def test_batch_ok(client, best_model_id):
    texts = ["很好", "很差", "一般般"]
    r = client.post("/api/batch", json={"texts": texts, "model_id": best_model_id})
    assert r.status_code == 200
    d = r.get_json()
    assert len(d) == len(texts)
    assert [x["text"] for x in d] == texts


def test_batch_rejects_non_list(client, best_model_id):
    r = client.post("/api/batch", json={"texts": "很好", "model_id": best_model_id})
    assert r.status_code == 400


def test_batch_rejects_empty_list(client, best_model_id):
    r = client.post("/api/batch", json={"texts": [], "model_id": best_model_id})
    assert r.status_code == 400


def test_batch_rejects_oversized_payload(client, best_model_id):
    """单次批量上限，防止一次请求打爆显存。"""
    r = client.post("/api/batch", json={"texts": ["好"] * 1001, "model_id": best_model_id})
    assert r.status_code == 400


def test_batch_rejects_blank_item(client, best_model_id):
    """数组里混进空白串同样要挡下。

    只校验「是不是数组」不够：一条空串会让分词结果为空，
    推理层得靠 lengths 兜底成 1 才不越界——那是防御，不该当正常路径用。
    """
    r = client.post("/api/batch", json={"texts": ["好", "   "], "model_id": best_model_id})
    assert r.status_code == 400


def test_batch_rejects_overlong_item(client, best_model_id):
    """逐条长度上限与单条接口一致，不能靠批量绕过 MAX_TEXT_CHARS。"""
    r = client.post(
        "/api/batch",
        json={"texts": ["好", "好" * (config.MAX_TEXT_CHARS + 1)], "model_id": best_model_id},
    )
    assert r.status_code == 400


def test_inference_failure_returns_500_json(client, best_model_id, monkeypatch):
    """推理层抛出未预期异常时，必须转成 500 + JSON，且不泄漏内部细节。

    只回异常**类型名**：对排查够用，又不会把栈信息交给前端。
    响应体做全等断言，「多带了字段」也算失败——契约就是这一个键。
    """

    def boom(_model_id):
        raise RuntimeError("模拟崩溃")

    monkeypatch.setattr(infer, "get_predictor", boom)
    r = client.post("/api/predict", json={"text": "很好", "model_id": best_model_id})
    assert r.status_code == 500
    assert r.get_json() == {"error": "推理失败：RuntimeError"}


def test_unknown_path_returns_404_json(client):
    """未知路径必须是 404 + JSON，而不是 Flask 默认的 HTML 404 页。

    前端的 request() 统一 await res.json()；一个 HTML 体在那里会抛解析错误，
    界面上只剩 "Unexpected token '<'"，真正的原因（路径写错了）被整个盖掉。
    「错误响应统一为 JSON」只有在**所有**响应都是 JSON 时才成立。
    """
    r = client.get("/api/nope")
    assert r.status_code == 404
    assert r.content_type.startswith("application/json"), f"实测 {r.content_type}"
    assert "error" in r.get_json()


def test_wrong_method_returns_405_json(client):
    """方法不对（POST 路由被 GET）必须是 405 + JSON，而不是 HTML 405 页。

    这条与上一条是两件事：404 是「路径不存在」，405 是「路径存在但方法不对」，
    前端 typo 掉一个方法时，它得能区分这两种情况。

    `Allow` 头一并钉住：405 换掉响应体时最容易丢掉的就是它，
    而 RFC 9110 要求 405 必须带 `Allow`——丢了就是违反 HTTP 语义，
    与「体的格式」无关，属于另一类错误。
    只断言「包含 POST」而不是「等于 POST」：CORS 会给每条路由注册 OPTIONS
    以支持预检，实测该头是 `POST, OPTIONS`，这是对的，不该被测试挡掉。
    """
    r = client.get("/api/predict")
    assert r.status_code == 405
    assert r.content_type.startswith("application/json"), f"实测 {r.content_type}"
    assert "error" in r.get_json()
    allow = r.headers.get("Allow", "")
    assert "POST" in allow, f"405 丢了 Allow 头，实测 {allow!r}"


def test_unhandled_error_outside_inference_returns_500_json(client, monkeypatch):
    """不经过推理层的故障（如 meta.json 损坏）也必须回 JSON。

    /api/models 没有自己的 try/except，异常会一路冒到应用层。
    这正是「兜底处理器」存在的意义：否则前端的单一错误路径只能覆盖一半接口。
    """

    def boom():
        raise RuntimeError("模拟 meta.json 损坏")

    monkeypatch.setattr(infer, "list_models", boom)
    r = client.get("/api/models")
    assert r.status_code == 500
    assert r.content_type.startswith("application/json"), f"实测 {r.content_type}"
    assert r.get_json() == {"error": "推理失败：RuntimeError"}


def test_framework_http_error_keeps_its_status_code(client, monkeypatch):
    """框架级的客户端错误必须保留自己的状态码，不能被兜底处理器压成 500。

    413/415 这类错误是「请求有问题」，报 500「推理失败」等于把客户端错误
    谎报成服务端错误：前端会以为该重试/报警，而用户改一下请求就好了。
    为此 HTTPException 按**类**注册（见 api.py 的注释），本用例把这条钉死。
    """

    def boom():
        raise RequestEntityTooLarge()

    monkeypatch.setattr(infer, "list_models", boom)
    r = client.get("/api/models")
    assert r.status_code == 413, f"实测被压成了 {r.status_code}：{r.get_json()}"
    assert r.content_type.startswith("application/json"), f"实测 {r.content_type}"
    assert "error" in r.get_json()


def test_max_len_param_is_honored(client, best_model_id):
    """max_len 应影响分词截断长度。"""
    long_text = "很好 " * 50
    a = client.post(
        "/api/predict", json={"text": long_text, "model_id": best_model_id, "max_len": 5}
    ).get_json()
    assert len(a["tokens"]) <= 5


@pytest.mark.parametrize(
    "text, expect_over_limit",
    [(SHORT_TEXT, False), (LONG_TEXT, True)],
    ids=["短文本", "超100词长文本"],
)
def test_explain_agrees_with_predict(client, best_model_id, text, expect_over_limit):
    """同一输入，两个端点的 label / prob_pos / prob_neg / tokens 必须完全一致。

    spec §5 把这条写成了硬约束；§6.2 又规定开启逐词贡献时前端**只**调
    /api/explain（不再调 /api/predict），所以一旦两者分叉，用户看到的就是
    与「单条预测」自相矛盾的结论，前端无从解释、也无从修正。

    这一层是**第二个**能引入分叉的地方：infer 里已修过一次的截断长度不一致
    （explain 用 CONTRIB_MAX_TOKENS、predict 用 max_len，实测 1200 条测试集中
    134 条概率不同、14 条标签相反），在接口层可以由「一个路由传默认值、
    另一个不传」之类的差异重新长出来。四个字段分开断言，
    失败时能直接指出是哪一项分了叉。

    长文本用例是**关键**：短文本下两条路径恰好等价，看不出任何问题。
    """
    payload = {"text": text, "model_id": best_model_id}
    single = client.post("/api/predict", json=payload).get_json()
    explained = client.post("/api/explain", json=payload).get_json()

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
        # 只走接口，不碰任何私有实现。
        truncated = client.post(
            "/api/predict", json={**payload, "max_len": config.CONTRIB_MAX_TOKENS}
        ).get_json()
        assert truncated["label"] != single["label"], (
            "夹具失去分辨力：截到 CONTRIB_MAX_TOKENS 与截到 max_len 得到相同标签，"
            "本用例对「explain 用了更短的输入」已无感觉。请更换长文本夹具。"
        )

    # 四个字段逐个断言，失败时能立刻看出是哪一项分叉。
    assert explained["label"] == single["label"], (
        f"label 不一致：explain={explained['label']} predict={single['label']}"
    )
    assert explained["prob_pos"] == single["prob_pos"], (
        f"prob_pos 不一致：explain={explained['prob_pos']!r} predict={single['prob_pos']!r}"
    )
    assert explained["prob_neg"] == single["prob_neg"], (
        f"prob_neg 不一致：explain={explained['prob_neg']!r} predict={single['prob_neg']!r}"
    )
    assert explained["tokens"] == single["tokens"], (
        "tokens 不一致：前端用 explain 的 tokens 作渲染骨架，"
        "它必须与 predict 逐词相同，否则高亮句与概率对不上。"
    )

    # contributions 的上限是「前向次数预算」，不是输入长度，不能随之放开。
    assert len(explained["contributions"]) <= config.CONTRIB_MAX_TOKENS
