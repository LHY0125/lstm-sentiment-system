# -*- coding: utf-8 -*-
"""多语料适配层的单元测试。

**不联网**：网络路径（_fetch）在这里用桩替换，真正下载过的产物不在测试里依赖——
否则测试会因为链路抖动而随机变红。
"""

import io
import zipfile

import pytest

from senti import corpora


# ---------------------------------------------------------------- 登记表

def test_registry_is_wellformed():
    """登记表的每个条目都自洽：有镜像、有标签映射、读取方式已知。"""
    for key, spec in corpora.CORPORA.items():
        assert spec.key == key, f"{key} 的 spec.key 与字典键不一致"
        assert spec.files, f"{key} 没有任何文件"
        for file_spec in spec.files:
            assert file_spec.relpath, f"{key} 的文件没有落盘名"
            assert file_spec.urls, f"{key}/{file_spec.relpath} 没有任何镜像"
            for url in file_spec.urls:
                assert url.startswith("https://hf-mirror.com/datasets/"), url
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
    """DMSC 形状：3 星整行不产出，1/5 星分别映射到 0/1。"""
    spec = corpora.CORPORA["DMSC"]
    text = "Star,Comment,Movie_Name_CN\n5,好看,甲\n3,一般,甲\n2,难看,乙\n"
    rows = list(corpora._iter_csv(io.StringIO(text), spec, "DMSC.csv", None))
    assert [r.label for r in rows] == [1, 0]
    assert [r.group for r in rows] == ["甲", "乙"]


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
    assert spec.reader == "zip-csv"
    rows = list(corpora._iter_rows(archive, spec))
    assert [r.label for r in rows] == [1, 0]
    assert rows[0].group == "甲"
    assert not list(tmp_path.iterdir()) or len(list(tmp_path.iterdir())) == 1  # 只有 zip，没有解压产物


def test_zip_with_two_csv_members_is_rejected(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a.csv", "label,review\n1,x\n")
        zf.writestr("b.csv", "label,review\n1,y\n")
    with zipfile.ZipFile(archive) as zf, pytest.raises(ValueError, match="恰好 1 个"):
        corpora._single_csv_name(zf)


# ---------------------------------------------------------------- 下载：长度解析与失败处置

@pytest.mark.parametrize(
    "header,expected",
    [
        ("bytes 100-999/1000", 1000),
        ("bytes 0-0/1", 1),
        ("bytes 0-99/*", None),
        (None, None),
        ("garbage", None),
    ],
)
def test_total_from_content_range(header, expected):
    assert corpora._total_from_content_range(header) == expected


def test_fetch_validated_discards_tmp_when_content_is_unusable(tmp_path, monkeypatch):
    """内容坏（比如 200 的 HTML 错误页）必须丢弃临时文件，
    否则续传只会把一个坏文件补完，而 is_downloaded() 之后永远为真。"""
    dest = tmp_path / "waimai_10k.csv"
    html = b"<!DOCTYPE html>\n<html><body>404 Not Found</body></html>\n"
    monkeypatch.setattr(corpora, "_fetch", lambda url, tmp: tmp.write_bytes(html))
    with pytest.raises(ValueError, match="解析"):
        corpora._fetch_validated("https://x", dest, corpora.CORPORA["waimai_10k"])
    assert not corpora._tmp_path(dest).exists()
    assert not dest.exists()


def test_fetch_validated_keeps_tmp_on_transfer_failure(tmp_path, monkeypatch):
    """传输中断必须保留临时文件——这是弱网下拿下大文件的唯一活路。"""

    def fake_fetch(url, tmp):
        tmp.write_bytes(b"partial bytes")
        raise corpora.IncompleteTransfer("boom")

    dest = tmp_path / "waimai_10k.csv"
    monkeypatch.setattr(corpora, "_fetch", fake_fetch)
    with pytest.raises(corpora.IncompleteTransfer):
        corpora._fetch_validated("https://x", dest, corpora.CORPORA["waimai_10k"])
    assert corpora._tmp_path(dest).read_bytes() == b"partial bytes"


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


def test_load_rows_requires_download(tmp_path, monkeypatch):
    monkeypatch.setattr(corpora.config, "DATA_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="语料未下载"):
        corpora.load_rows("waimai_10k")
