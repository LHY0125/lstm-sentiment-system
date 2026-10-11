# -*- coding: utf-8 -*-
"""多语料适配层的单元与集成测试。

**不联网**：下载相关的用例都打到一个本进程内的 HTTP 服务端上（`serve` 夹具），
所以既能驱动真实的 `_fetch` / `_fetch_validated` / `_download_one`，又不会因为
公网抖动而随机变红。真语料（`数据集/` 下）不参与测试。
"""

import io
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from senti import corpora

# 一个对 waimai_10k 而言**合法**的语料体，且长度跨过多个 64 KiB 数据块，
# 这样续传、截断这些行为才有东西可测。
CORPUS_BODY = b"label,review\n" + b"1,\xe5\xa5\xbd\xe5\x90\x83\n" * 20000
HTML_PAGE = b"<!DOCTYPE html>\n<html><body>404 Not Found</body></html>\n"
# 约 3.6 MB、跨 55 个数据块：用来模拟「每次连接只推进一块」的链路，块数刻意 >40，
# 以便钉住 MAX_ATTEMPTS 不能退回那个不够用的旧值。
BIG_BODY = b"label,review\n" + b"1,\xe5\xa5\xbd\xe5\x90\x83\n" * 400_000
# 约 810 KB、跨 13 个数据块：配合「每 3 次请求抖一次」用来验证停滞计数会被推进清零。
MEDIUM_BODY = b"label,review\n" + b"1,\xe5\xa5\xbd\xe5\x90\x83\n" * 90_000


class _CorpusHandler(BaseHTTPRequestHandler):
    """按 class 属性 `script` 的剧本应答，用来对真实下载路径做对抗性实验。

    剧本键：
      body            要发送的字节
      content_length  显式声明的 Content-Length（可故意写错）
      total           Content-Range 里自称的总长（缺省用真实长度）
      send_length     是否发送 Content-Length（False = 什么长度都不给）
      support_range   是否支持 Range（False = 恒返 200）
      seen            每次请求记录下的 Range 头（由夹具初始化）
    """

    script: dict = {}

    def log_message(self, *args):   # 静音：否则每个请求都往 stderr 打一行
        return

    def do_GET(self):
        s = type(self).script
        s.setdefault("seen", []).append(self.headers.get("Range"))
        body = s.get("body", b"")
        rng = self.headers.get("Range")
        use_range = bool(rng) and s.get("support_range", True)
        start = int(rng.split("=", 1)[1].split("-", 1)[0]) if use_range else 0

        payload = body[start:]
        if s.get("one_chunk"):
            payload = payload[:corpora.CHUNK]   # 模拟「一次连接只推进一个数据块」的链路

        # 第 2 次请求起，每 fail_every 次回一个空响应，模拟链路抖动。
        # 必须仍走 206（带正确的 Content-Range 总长），否则客户端会以 wb 重写、把半成品毁掉。
        s["requests"] = s.get("requests", 0) + 1
        if s.get("fail_every") and s["requests"] > 1 and s["requests"] % s["fail_every"] == 0:
            payload = b""
            s.setdefault("failed", []).append(s["requests"])

        if use_range:
            total = s.get("total")
            self.send_response(206)
            self.send_header(
                "Content-Range",
                f"bytes {start}-{start + len(payload) - 1}/{total if total is not None else len(body)}",
            )
            self.send_header("Content-Length", str(len(payload)))
        else:
            self.send_response(200)
            if s.get("content_length") is not None:
                self.send_header("Content-Length", str(s["content_length"]))
            elif s.get("send_length", True):
                self.send_header("Content-Length", str(len(payload)))

        self.send_header("Content-Type", "text/csv")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True


@pytest.fixture
def serve():
    """起一个本机可控服务端，返回 (url, script)。"""
    _CorpusHandler.script = {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CorpusHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/corpus.csv"
    try:
        yield url, _CorpusHandler.script
    finally:
        server.shutdown()
        server.server_close()
        _CorpusHandler.script = {}


# ---------------------------------------------------------------- 登记表

def test_registry_is_wellformed():
    """登记表的每个条目都自洽：有镜像、有标签映射、读取方式已知、字节数已填。"""
    for key, spec in corpora.CORPORA.items():
        assert spec.key == key, f"{key} 的 spec.key 与字典键不一致"
        assert spec.files, f"{key} 没有任何文件"
        for file_spec in spec.files:
            assert file_spec.relpath, f"{key} 的文件没有落盘名"
            assert file_spec.urls, f"{key}/{file_spec.relpath} 没有任何镜像"
            for url in file_spec.urls:
                assert url.startswith("https://hf-mirror.com/datasets/"), url
            # 没有它，服务端又不给长度时，截断的文件无法被识破
            assert file_spec.expected_bytes > 0, f"{key}/{file_spec.relpath} 没填 expected_bytes"
        assert spec.label_map, f"{key} 没有声明标签映射"
        assert spec.reader in {"csv", "parquet", "zip-csv"}, f"{key} 读取方式未知"
        # 丢弃档与映射档必须互斥，否则同一条标签会有两种解释
        assert not (set(spec.label_map) & set(spec.drop_labels)), f"{key} 的丢弃档与映射档重叠"


def test_corpus_dir_unknown_key_raises():
    with pytest.raises(KeyError):
        corpora.corpus_dir("不存在的语料")


# ---------------------------------------------------------------- 编码嗅探

def test_sniff_encoding_utf8_when_sample_splits_a_character(tmp_path):
    """回归测试：样本切口落在多字节汉字中间时，不得误判成「非 UTF-8」。

    这是实现时真实踩到的 bug——waimai_10k.csv 整文件是合法 UTF-8，却因为
    65536 字节的切口劈开了一个汉字，被判成「既非 UTF-8 也非 GBK」。
    """
    path = tmp_path / "x.csv"
    path.write_bytes(b"a" * 4 + "中".encode("utf-8"))   # 第 5 个字节起是「中」的首字节
    # sample=5 → 样本正好 = b"aaaa" + 「中」的第一个字节，序列不完整
    assert corpora._sniff_encoding(path, sample=5) == "utf-8-sig"


def test_sniff_encoding_detects_gbk(tmp_path):
    path = tmp_path / "gbk.csv"
    path.write_bytes("标签,评论\n1,很好吃\n".encode("gbk"))
    assert corpora._sniff_encoding(path) == "gbk"


def test_sniff_encoding_rejects_undecodable(tmp_path):
    path = tmp_path / "bin.csv"
    # 0xFF 在 UTF-8 与 GBK 里都不是合法起始字节
    path.write_bytes(b"\xff\xfe\xff\xfe" * 100)
    with pytest.raises(ValueError, match="无法判断编码"):
        corpora._sniff_encoding(path)


# ---------------------------------------------------------------- 标签（失败关闭）

@pytest.mark.parametrize(
    "raw,expected",
    [("1", 1), ("0", 0), (1, 1), (0, 0), ("1.0", 1), (" 0.0 ", 0)],
)
def test_label_key_accepts_declared_values(raw, expected):
    assert corpora._label_key(raw, {"1": 1, "0": 0}, "t", 0) == expected


@pytest.mark.parametrize("raw", ["2", "-1", "positive", "", None, "1.5"])
def test_label_key_fails_closed_on_unknown(raw):
    """未知/非整数标签一律报错。静默错标会持续污染训练集且事后极难发现。"""
    with pytest.raises(ValueError):
        corpora._label_key(raw, {"1": 1, "0": 0}, "t", 7)


def test_label_key_mentions_record_number():
    with pytest.raises(ValueError, match="第 42 条"):
        corpora._label_key("9", {"1": 1, "0": 0}, "t", 42)


@pytest.mark.parametrize("star,expected", [("1", 0), ("2", 0), ("4", 1), ("5", 1)])
def test_dmsc_star_maps_to_binary(star, expected):
    assert corpora._label_key(star, corpora.STAR_TO_LABEL, "t", 0, (corpora.STAR_NEUTRAL,)) == expected


def test_dmsc_three_stars_is_dropped_not_guessed():
    """3 星是中性，必须**丢弃**而不是硬塞进某一类——那是替数据编标签。"""
    assert corpora.STAR_NEUTRAL not in corpora.STAR_TO_LABEL
    assert corpora._label_key("3", corpora.STAR_TO_LABEL, "t", 0, (corpora.STAR_NEUTRAL,)) is None


def test_dmsc_unexpected_star_fails_closed():
    with pytest.raises(ValueError, match="取值非法"):
        corpora._label_key("6", corpora.STAR_TO_LABEL, "t", 0, (corpora.STAR_NEUTRAL,))


# ---------------------------------------------------------------- 读取

def test_iter_csv_reads_rows_and_skips_blank_text():
    spec = corpora.CORPORA["online_shopping_10_cats"]
    text = "cat,label,review\n书籍,1,很好看\n书籍,0,\n手机,0,很卡\n"
    rows = list(corpora._iter_csv(io.StringIO(text), spec, "t.csv", None))
    assert [r.label for r in rows] == [1, 0]          # 空评论被跳过
    assert [r.group for r in rows] == ["书籍", "手机"]
    assert rows[1].text == "很卡"


def test_iter_csv_skips_neutral_star_rows():
    """DMSC 形状：3 星整行不产出，1/2/4/5 星映射到 0/1。"""
    spec = corpora.CORPORA["DMSC"]
    text = "Star,Comment,Movie_Name_CN\n5,好看,甲\n3,一般,甲\n2,难看,乙\n"
    rows = list(corpora._iter_csv(io.StringIO(text), spec, "DMSC.csv", None))
    assert [r.label for r in rows] == [1, 0]
    assert [r.group for r in rows] == ["甲", "乙"]


def test_iter_csv_validates_label_even_when_text_is_blank():
    """空文本不能成为绕过失败关闭的通道。"""
    spec = corpora.CORPORA["waimai_10k"]
    with pytest.raises(ValueError, match="取值非法"):
        list(corpora._iter_csv(io.StringIO("label,review\n9,\n"), spec, "t.csv", None))


def test_iter_csv_strips_bom_from_header_and_values():
    """表头或值里的 BOM 必须被清掉：`str.strip()` 不把 U+FEFF 当空白。"""
    spec = corpora.CORPORA["waimai_10k"]
    text = "﻿label,review\n1,﻿好吃\n"
    rows = list(corpora._iter_csv(io.StringIO(text), spec, "t.csv", None))
    assert len(rows) == 1 and rows[0].text == "好吃"


def test_iter_csv_honours_limit():
    spec = corpora.CORPORA["waimai_10k"]
    text = "label,review\n" + "".join(f"1,第{i}条\n" for i in range(50))
    rows = list(corpora._iter_csv(io.StringIO(text), spec, "t.csv", limit=10))
    assert len(rows) == 10


def test_iter_csv_reports_missing_columns():
    spec = corpora.CORPORA["waimai_10k"]
    # 列名按「文本 → 标签」的顺序识别，故这里先报文本列
    with pytest.raises(ValueError, match="文本列无法识别"):
        list(corpora._iter_csv(io.StringIO("foo,bar\n1,2\n"), spec, "t.csv", None))


def test_zip_single_csv_reads_without_extracting(tmp_path):
    """zip 直读：不落盘解压，且按 DMSC 的列名与星级派生标签。"""
    archive = tmp_path / "x.csv.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("inner.csv", "Star,Comment,Movie_Name_CN\n4,好看,甲\n2,难看,乙\n")

    spec = corpora.CORPORA["DMSC"]
    rows = list(corpora._iter_rows(archive, spec))
    assert [r.label for r in rows] == [1, 0]
    assert rows[0].group == "甲"
    # 只留下 zip 本身，没有解压产物
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.csv.zip"]


def test_zip_with_two_csv_members_is_rejected(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a.csv", "label,review\n1,x\n")
        zf.writestr("b.csv", "label,review\n1,y\n")
    with zipfile.ZipFile(archive) as zf, pytest.raises(ValueError, match="恰好 1 个"):
        corpora._single_csv_name(zf)


# ---------------------------------------------------------------- _fetch（真实服务端）

def test_total_from_content_range():
    assert corpora._total_from_content_range("bytes 100-999/1000") == 1000
    assert corpora._total_from_content_range("bytes 0-0/1") == 1
    assert corpora._total_from_content_range("bytes 0-99/*") is None
    assert corpora._total_from_content_range(None) is None
    assert corpora._total_from_content_range("garbage") is None


@pytest.mark.parametrize(
    "server_total,declared,expected",
    [
        (1000, 999, 1000),   # 服务端优先
        (None, 999, 999),    # 服务端没给 → 用登记表
        (None, 0, None),     # 两边都没有 → 放弃长度校验
    ],
)
def test_resolve_expected(server_total, declared, expected):
    assert corpora._resolve_expected(server_total, declared) == expected


def test_fetch_downloads_a_complete_file(serve, tmp_path):
    url, script = serve
    script["body"] = CORPUS_BODY
    tmp = tmp_path / "x.part.csv"
    corpora._fetch(url, tmp, 0)
    assert tmp.read_bytes() == CORPUS_BODY


def test_fetch_detects_truncation_via_content_length(serve, tmp_path):
    """服务端声明得比实发多 → 判为中断（否则截断文件会被当成语料固化），
    且**保留**已收到的半成品供续传。

    构造要点：必须让服务端先送出**若干个完整数据块**再断。urllib3 是按 Content-Length
    整块读的，若不足一块就断，它会连已读到的字节一起丢掉抛 IncompleteRead——
    那样一个字节都留不下，也就没有「半成品」可测（这是分块读的固有性质，不是缺陷）。
    """
    url, script = serve
    script["body"] = CORPUS_BODY
    script["content_length"] = len(CORPUS_BODY) + 100_000      # 声称还有更多
    tmp = tmp_path / "x.part.csv"
    with pytest.raises(corpora.IncompleteTransfer):
        corpora._fetch(url, tmp, 0)
    size = tmp.stat().st_size
    assert 0 < size < len(CORPUS_BODY)      # 收到了一部分，且确实不完整


def test_fetch_detects_truncation_when_server_gives_no_length(serve, tmp_path):
    """回归：服务端既不回 Content-Length 也不回 Content-Range 时，
    必须靠登记表里声明的字节数兜底——否则截断文件会被当成语料固化。"""
    url, script = serve
    script["body"] = CORPUS_BODY[:1000]
    script["send_length"] = False
    tmp = tmp_path / "x.part.csv"
    with pytest.raises(corpora.IncompleteTransfer):
        corpora._fetch(url, tmp, len(CORPUS_BODY))
    assert tmp.stat().st_size == 1000


def test_fetch_resumes_with_range_and_matches_byte_for_byte(serve, tmp_path):
    url, script = serve
    script["body"] = CORPUS_BODY
    tmp = tmp_path / "x.part.csv"
    tmp.write_bytes(CORPUS_BODY[:70000])      # 预置半成品
    corpora._fetch(url, tmp, 0)
    assert script["seen"][0] == "bytes=70000-"   # 确实带了 Range
    assert tmp.read_bytes() == CORPUS_BODY       # 拼接结果与原文逐字节一致


def test_fetch_restarts_cleanly_when_server_ignores_range(serve, tmp_path):
    """服务端不支持续传时必须以 wb 重写，绝不能在半成品后面追加。"""
    url, script = serve
    script["body"] = CORPUS_BODY
    script["support_range"] = False
    tmp = tmp_path / "x.part.csv"
    tmp.write_bytes(b"garbage left by another source")
    corpora._fetch(url, tmp, 0)
    assert tmp.read_bytes() == CORPUS_BODY


def test_fetch_rejects_more_bytes_than_expected(serve, tmp_path):
    url, script = serve
    script["body"] = CORPUS_BODY
    script["total"] = 500            # 自称总共只有 500 字节
    tmp = tmp_path / "x.part.csv"
    tmp.write_bytes(CORPUS_BODY[:100])   # 有半成品才会走 206 分支
    with pytest.raises(ValueError, match="多于预期"):
        corpora._fetch(url, tmp, 0)
    assert not tmp.exists()          # 不可信，已丢弃


def test_fetch_rejects_empty_response(serve, tmp_path):
    url, script = serve
    script["body"] = b""
    tmp = tmp_path / "x.part.csv"
    with pytest.raises(ValueError, match="响应为空"):
        corpora._fetch(url, tmp, 0)


def test_fetch_enforces_max_file_bytes(serve, tmp_path, monkeypatch):
    """异常响应不能把磁盘写满。"""
    url, script = serve
    script["body"] = CORPUS_BODY
    monkeypatch.setattr(corpora, "MAX_FILE_BYTES", 1000)
    tmp = tmp_path / "x.part.csv"
    with pytest.raises(ValueError, match="超过单文件上限"):
        corpora._fetch(url, tmp, 0)
    assert not tmp.exists()


# ---------------------------------------------------------------- _fetch_validated（失败处置）

def test_fetch_validated_discards_tmp_when_content_is_unusable(serve, tmp_path):
    """内容坏（比如 200 的 HTML 错误页）必须**丢弃**临时文件，
    否则续传只会把一个坏文件补完，而 is_downloaded() 之后永远为真。"""
    url, script = serve
    script["body"] = HTML_PAGE
    dest = tmp_path / "waimai_10k.csv"
    with pytest.raises(corpora.UnusableContent):
        corpora._fetch_validated(url, dest, corpora.CORPORA["waimai_10k"], 0)
    assert not corpora._tmp_path(dest).exists()
    assert not dest.exists()


def test_fetch_validated_keeps_tmp_when_transfer_is_interrupted(serve, tmp_path):
    """传输中断必须**保留**临时文件——这是弱网下拿下大文件的唯一活路。

    与上一条恰好相反，所以这里用真实的传输中断来驱动（而不是打桩抛异常）：
    打桩的异常发生在 try 之外，断言对任何异常都成立，等于没测。
    """
    url, script = serve
    script["body"] = CORPUS_BODY
    script["content_length"] = len(CORPUS_BODY) + 100_000
    dest = tmp_path / "waimai_10k.csv"
    with pytest.raises(corpora.IncompleteTransfer):
        corpora._fetch_validated(url, dest, corpora.CORPORA["waimai_10k"], 0)
    assert 0 < corpora._tmp_path(dest).stat().st_size < len(CORPUS_BODY)
    assert not dest.exists()


def test_fetch_validated_accepts_a_good_download(serve, tmp_path):
    """正例：完整且能解析 → 原子改名落定，临时文件消失。"""
    url, script = serve
    script["body"] = CORPUS_BODY
    dest = tmp_path / "waimai_10k.csv"
    corpora._fetch_validated(url, dest, corpora.CORPORA["waimai_10k"], len(CORPUS_BODY))
    assert dest.read_bytes() == CORPUS_BODY
    assert not corpora._tmp_path(dest).exists()


# ---------------------------------------------------------------- _download_one（续传与重试）

def _file_spec(url: str) -> corpora.FileSpec:
    return corpora.FileSpec(relpath="x.csv", urls=(url,), expected_bytes=len(CORPUS_BODY))


def test_download_one_resumes_a_partial_from_the_same_source(serve, tmp_path):
    """回归：上个进程留下的半成品必须被接上，而不是被无条件删掉重来。"""
    url, script = serve
    script["body"] = CORPUS_BODY
    dest = tmp_path / "x.csv"
    tmp = corpora._tmp_path(dest)
    tmp.write_bytes(CORPUS_BODY[:70000])
    corpora._src_marker(tmp).write_text(url, encoding="utf-8")

    corpora._download_one("waimai_10k", _file_spec(url), dest, corpora.CORPORA["waimai_10k"])

    assert script["seen"][0] == "bytes=70000-"       # 接着上次的位置要
    assert dest.read_bytes() == CORPUS_BODY
    assert not corpora._src_marker(tmp).exists()      # 标记已清理


def test_download_one_restarts_when_partial_came_from_another_source(serve, tmp_path):
    """换源时不能接着用：不同源的字节拼在一起是能通过长度校验的垃圾。"""
    url, script = serve
    script["body"] = CORPUS_BODY
    dest = tmp_path / "x.csv"
    tmp = corpora._tmp_path(dest)
    tmp.write_bytes(b"bytes left by a different mirror")
    corpora._src_marker(tmp).write_text("https://hf-mirror.com/datasets/别的源/x.csv", encoding="utf-8")

    corpora._download_one("waimai_10k", _file_spec(url), dest, corpora.CORPORA["waimai_10k"])

    assert script["seen"][0] is None                  # 第一次请求不带 Range，从头来
    assert dest.read_bytes() == CORPUS_BODY


def test_download_one_gives_up_after_repeated_no_progress(serve, tmp_path, monkeypatch):
    """源已经死了就别硬耗：连续 STALL_LIMIT 次零推进即放弃。"""
    url, script = serve
    script["body"] = b""                      # 每次都失败且不留半成品
    monkeypatch.setattr(corpora, "STALL_LIMIT", 2)
    dest = tmp_path / "x.csv"

    with pytest.raises(RuntimeError, match="全部下载源均失败"):
        corpora._download_one("waimai_10k", _file_spec(url), dest, corpora.CORPORA["waimai_10k"])

    # 2 次无进展就停，而不是打满 MAX_ATTEMPTS
    assert len(script["seen"]) == 2


def test_download_one_does_not_retry_a_source_whose_content_is_unusable(serve, tmp_path):
    """内容不对时重下多少次都一样——必须立刻放弃该源，别白耗带宽。"""
    url, script = serve
    script["body"] = HTML_PAGE
    dest = tmp_path / "x.csv"

    with pytest.raises(RuntimeError, match="全部下载源均失败"):
        corpora._download_one("waimai_10k", _file_spec(url), dest, corpora.CORPORA["waimai_10k"])

    assert len(script["seen"]) == 1           # 只请求了一次


def test_download_one_survives_a_link_that_advances_one_chunk_at_a_time(serve, tmp_path):
    """回归 Important-3：链路每次连接只推进一个数据块时，必须靠续传一路跑完。

    这条钉住的是 `MAX_ATTEMPTS` 足够大——把上限退回 40（旧值）它会立刻变红。
    注意它**不**覆盖「停滞计数在推进时清零」：在单调推进的场景里那个重置是行为等价的
    （计数初值为 0，只在无进展时自增），所以这里观察不到差别，那条另有用例覆盖。
    """
    url, script = serve
    script["body"] = BIG_BODY
    script["one_chunk"] = True
    script["content_length"] = len(BIG_BODY)
    dest = tmp_path / "x.csv"

    corpora._download_one(
        "waimai_10k",
        corpora.FileSpec("x.csv", (url,), expected_bytes=len(BIG_BODY)),
        dest,
        corpora.CORPORA["waimai_10k"],
    )

    assert dest.read_bytes() == BIG_BODY
    attempts = len(script["seen"])
    assert attempts == -(-len(BIG_BODY) // corpora.CHUNK)     # 每次恰好推进一块
    assert attempts > 40, "这条用例的意义就在于块数要多于旧上限，否则钉不住 MAX_ATTEMPTS"


def test_download_one_forgives_earlier_stalls_once_progress_resumes(serve, tmp_path):
    """间歇性失败不该累积成「源已死」：只要还在推进，之前的停滞就该被清零。

    服务端每次连接推进一块，但每第 3 次请求回一次空响应。若实现**不**在推进时重置
    停滞计数，无进展就会一路累加、到第 3 次误判放弃——而它其实一直在前进。
    这条正是 `test_..._one_chunk_at_a_time` 那条 docstring 里说明它覆盖不到的行为。
    """
    url, script = serve
    script["body"] = MEDIUM_BODY
    script["one_chunk"] = True
    script["content_length"] = len(MEDIUM_BODY)
    script["fail_every"] = 3
    dest = tmp_path / "x.csv"

    corpora._download_one(
        "waimai_10k",
        corpora.FileSpec("x.csv", (url,), expected_bytes=len(MEDIUM_BODY)),
        dest,
        corpora.CORPORA["waimai_10k"],
    )

    assert dest.read_bytes() == MEDIUM_BODY
    # 确实发生了多次「无进展」，但下载仍然完成——否则这条用例证明不了什么
    assert len(script["failed"]) >= 3
    assert corpora.STALL_LIMIT <= len(script["failed"])


# ---------------------------------------------------------------- 端到端（本地目录）

def test_load_rows_against_a_local_corpus_dir(tmp_path, monkeypatch):
    """把 DATA_ROOT 指向临时目录，走完 is_downloaded → load_rows 的完整路径。"""
    monkeypatch.setattr(corpora.config, "DATA_ROOT", tmp_path)
    assert not corpora.is_downloaded("waimai_10k")

    target = tmp_path / "waimai_10k"
    target.mkdir()
    (target / "waimai_10k.csv").write_text(
        "label,review\n1,很快很好吃\n0,凉了很难吃\n1,配送员态度好\n", encoding="utf-8"
    )

    assert corpora.is_downloaded("waimai_10k")
    rows = corpora.load_rows("waimai_10k")
    assert len(rows) == 3
    assert [r.label for r in rows] == [1, 0, 1]
    assert rows[0].group == ""      # 没有品类列时 group 为空串，不是 None


def test_iter_rows_streams_without_loading_everything(tmp_path, monkeypatch):
    monkeypatch.setattr(corpora.config, "DATA_ROOT", tmp_path)
    target = tmp_path / "waimai_10k"
    target.mkdir()
    (target / "waimai_10k.csv").write_text(
        "label,review\n" + "".join(f"1,第{i}条\n" for i in range(100)), encoding="utf-8"
    )
    rows = corpora.iter_rows("waimai_10k")
    assert next(rows).text == "第0条"        # 生成器：取一条不必读完整个文件


def test_load_rows_requires_download(tmp_path, monkeypatch):
    monkeypatch.setattr(corpora.config, "DATA_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="语料未下载"):
        corpora.load_rows("waimai_10k")
