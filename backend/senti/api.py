# -*- coding: utf-8 -*-
"""接口层：Flask 路由。

薄层——只做参数校验、调用推理层、序列化响应，不含任何业务逻辑。
"""

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from senti import config, infer

# 单次批量请求的条数上限，防止一次请求打爆显存。
# 放在这里而不是 config：它约束的是「一次请求装多少条」，属请求管道；
# config 里住的是数据/模型/推理的超参数。
MAX_BATCH_SIZE = 1000

# 请求里 max_len 的可接受上限。刻意远大于 config.MAX_LEN（默认为它）：
# max_len 是前端可调的截断旋钮，允许试到比默认更长，但必须有天花板，
# 否则一条 max_len=10**9 的请求会让词索引矩阵直接吃掉全部内存。
MAX_LEN_LIMIT = 1000

# 框架自己抛的 HTTP 错误的中文说明。只给「前端真会撞上」的两个写专门措辞，
# 其余状态码走下面的通用句式——保住码值比编一句像样的话更重要。
_HTTP_ERROR_MESSAGES = {
    404: "接口不存在",
    405: "请求方法不允许",
}

# 错误响应的统一形状：(JSON 体, HTTP 状态码)。
ErrorResponse = tuple[Response, int]


def _error(message: str, status: int) -> ErrorResponse:
    """统一的错误响应。所有失败路径都必须经过这里，形状才不会分叉。"""
    return jsonify({"error": message}), status


def _payload() -> dict:
    """请求体，非 JSON 对象一律当空字典处理。

    统一在这里兜底（而不是各路由各写一次 `get_json(silent=True) or {}`），
    是为了让「没有 body / JSON 坏掉 / body 是数组」这三种情况也走到参数校验、
    返回契约内的 {"error": ...}，而不是让 `payload.get` 抛 AttributeError
    变成一个 HTML 的 500——那会破坏「错误响应统一为 JSON」这条约定。
    """
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def _validate_text(payload: dict) -> tuple[str | None, ErrorResponse | None]:
    """校验 text，返回 (text, None) 或 (None, 错误响应)。"""
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, _error("text 必须是非空字符串", 400)
    if len(text) > config.MAX_TEXT_CHARS:
        return None, _error(f"text 超长（上限 {config.MAX_TEXT_CHARS} 字符）", 400)
    return text, None


def _validate_model(payload: dict) -> tuple[str | None, ErrorResponse | None]:
    """校验 model_id 是否存在于可用模型列表中。

    可用集合从 infer.list_models() 现取，而不是在启动时缓存：
    checkpoint 是离线产物，训练完新模型后不需要重启服务就能被识别。
    """
    model_id = payload.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        return None, _error("model_id 必须是非空字符串", 400)
    available = {m["id"] for m in infer.list_models()}
    if model_id not in available:
        return None, _error(f"模型不存在：{model_id}；可用模型：{sorted(available)}", 404)
    return model_id, None


def _max_len(payload: dict) -> int:
    """解析可选的 max_len，非法值回退到配置默认值。

    刻意不因为「超范围」而报 400：max_len 是个可选的调参旋钮，
    传了乱七八糟的值时「用默认值继续算」比「整条请求失败」更符合使用预期。
    代价是超范围的值会被静默忽略，前端无法从响应里看出自己的值没生效。
    """
    value = payload.get("max_len")
    # bool 是 int 的子类，True 会通过 isinstance 检查——显式排除，
    # 否则 max_len=true 会被当成 max_len=1 这种谁也想不到的截断长度。
    if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= MAX_LEN_LIMIT:
        return value
    return config.MAX_LEN


def create_app() -> Flask:
    """构造 Flask 应用。

    用工厂函数而不是模块级 app：测试可以直接 create_app() 拿实例跑
    test_client，不必起服务、也不会在不同测试之间共享路由状态。
    """
    app = Flask(__name__)
    # 开发期前端经 Vite 代理访问，本不需要 CORS；
    # 保留它以便直接用浏览器或 curl 调试。
    CORS(app)

    @app.get("/api/health")
    def health() -> Response:
        return jsonify({"status": "ok"})

    @app.get("/api/models")
    def models() -> Response:
        # 原样转发 infer 的顺序（按 val_acc 降序）：此处重排会让
        # 「按测试集挑模型」的选择性偏差从接口层漏回前端。
        return jsonify(infer.list_models())

    @app.post("/api/predict")
    def predict() -> ErrorResponse | Response:
        payload = _payload()
        # 先校验文本、再校验模型：两个字段都缺时报 400（参数错误）而非 404，
        # 因为「你少了 text」是用户能立刻修好的问题，
        # 报 404 会把人引向「模型列表坏了」这个错误方向。
        text, err = _validate_text(payload)
        if err:
            return err
        model_id, err = _validate_model(payload)
        if err:
            return err
        try:
            result = infer.get_predictor(model_id).predict(text, _max_len(payload))
        except Exception as exc:  # noqa: BLE001 —— 在边界处统一转 500
            # 只回异常**类型名**：对排查够用，又不把栈信息泄漏给前端。
            return _error(f"推理失败：{type(exc).__name__}", 500)
        return jsonify(result)

    @app.post("/api/explain")
    def explain() -> ErrorResponse | Response:
        payload = _payload()
        text, err = _validate_text(payload)
        if err:
            return err
        model_id, err = _validate_model(payload)
        if err:
            return err
        try:
            # 与 /api/predict 走同一套默认值（_max_len），
            # 否则同一输入两个端点会给出不同概率与标签——spec §5 的硬约束。
            result = infer.get_predictor(model_id).explain(text, _max_len(payload))
        except Exception as exc:  # noqa: BLE001
            return _error(f"推理失败：{type(exc).__name__}", 500)
        return jsonify(result)

    @app.post("/api/batch")
    def batch() -> ErrorResponse | Response:
        payload = _payload()
        texts = payload.get("texts")
        # 数组本身的形状先于模型校验：这样「texts 传错类型」不会被
        # 「模型不存在」的 404 盖掉，用户看到的错误指向真正的问题。
        if not isinstance(texts, list):
            return _error("texts 必须是字符串数组", 400)
        if not texts:
            return _error("texts 不能为空", 400)
        if len(texts) > MAX_BATCH_SIZE:
            return _error(f"单次批量上限 {MAX_BATCH_SIZE} 条，当前 {len(texts)} 条", 400)
        if not all(isinstance(t, str) and t.strip() for t in texts):
            return _error("texts 中每一条都必须是非空字符串", 400)
        # 逐条长度上限与单条接口一致，避免用批量接口绕过 MAX_TEXT_CHARS。
        if any(len(t) > config.MAX_TEXT_CHARS for t in texts):
            return _error(f"存在超长文本（单条上限 {config.MAX_TEXT_CHARS} 字符）", 400)

        model_id, err = _validate_model(payload)
        if err:
            return err
        try:
            result = infer.get_predictor(model_id).batch_predict(texts, _max_len(payload))
        except Exception as exc:  # noqa: BLE001
            return _error(f"推理失败：{type(exc).__name__}", 500)
        return jsonify(result)

    # ------------------------------------------------------------ 错误信封
    # 上面各路由只覆盖「走到视图函数之后」的失败。框架自己产生的响应
    # （未知路径、方法不对、请求体过大……）默认是 text/html，
    # 而前端的 request() 统一 await res.json()——一个 HTML 体就会让那里抛
    # 解析错误，界面上出现 "Unexpected token '<'"，真正的错误信息被整个盖掉。
    # 「错误响应统一为 JSON」只有在**所有**响应都是 JSON 时才成立，
    # 所以必须在应用层再兜一道。

    @app.errorhandler(HTTPException)
    def _http_error(err: HTTPException) -> ErrorResponse:
        """框架级 HTTP 错误 → 原状态码 + JSON。

        按**类**注册（而不是逐个 @errorhandler(404)/@errorhandler(405)）：
        Werkzeug 的 HTTPException 是一个大类，逐个注册只能覆盖列举到的码，
        其余（413 请求体过大、415 媒体类型不支持……）会掉进下面的兜底处理器，
        被贴上 500「推理失败」的标签——把客户端错误谎报成服务端错误。
        实测：按状态码注册 404/405 时，413 得到
        `500 {"error": "推理失败：RequestEntityTooLarge"}`；按类注册后得到
        `413 {"error": "请求被拒绝（HTTP 413）"}`。

        类注册同时也是安全的：Flask 沿异常的 MRO 就近查找处理器，
        HTTPException 比 Exception 更近，故框架错误不会落进兜底。

        **框架给出的响应头必须跟过来**：405 的 `Allow`、401 的 `WWW-Authenticate`
        都住在 `get_headers()` 里，换掉响应体时很容易顺手把它们丢掉——
        而 RFC 9110 要求 405 必须带 `Allow`，缺了就是违反 HTTP 语义。
        实测：不搬运时 `Allow` 为 `None`（Werkzeug 默认响应里是 `POST`）。
        `Content-Type` / `Content-Length` 例外，那两个必须留给 jsonify 的结果，
        否则 JSON 体被标成 text/html，前端的 res.json() 照样炸。
        """
        message = _HTTP_ERROR_MESSAGES.get(err.code, f"请求被拒绝（HTTP {err.code}）")
        # err.code 理论上可为 None（Werkzeug 对无码异常按 500 处理），显式兜一下，
        # 免得把一个 None 塞进 HTTP 状态码。
        response, status = _error(message, err.code or 500)
        for key, value in err.get_headers():
            if key not in ("Content-Type", "Content-Length"):
                response.headers[key] = value
        return response, status

    @app.errorhandler(Exception)
    def _unhandled(err: Exception) -> ErrorResponse:
        """兜底：任何未预期异常都变成 JSON 500。

        口径与三处路由内的 500 完全一致——只回异常**类型名**，
        既不回 str(err) 也不回栈，避免把内部细节交给前端。
        副作用（有意为之）：meta.json 损坏这类「不经过推理」的故障
        也会得到 JSON 而不是 HTML，前端的单一错误路径得以成立。
        """
        return _error(f"推理失败：{type(err).__name__}", 500)

    return app
