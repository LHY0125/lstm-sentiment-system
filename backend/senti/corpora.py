# -*- coding: utf-8 -*-
"""多语料适配层：下载并读取 ChnSentiCorp 之外的互联网评论语料。

与 data.py 的分工（重要，别把两者混起来）
------------------------------------------------------------------
`data.py` 是**主训练语料**（ChnSentiCorp）的专用数据层：它建的词表、切分的
训练/测试集与 checkpoint 一一对应，被模型、训练与推理层直接依赖。

本模块**不是**第二个 data.py。它只做一件事：把别的语料下载到本地，并统一成
「(文本, 标签)」——为跨域实验、数据准备与报告的数据集章节服务。
因此这里刻意**不**提供 tokenize / build_vocab / encode：那些属于主管线，
而跨域重建词表意味着全部 checkpoint 都要重训，不在本模块的职责范围内。

标签统一约定与 data.py 一致：**1=积极 / 0=消极**。
每个语料的原始编码在 `CorpusSpec.label_map` 里逐条声明，未知取值一律报错
（失败关闭）——静默错标会持续污染训练集且事后极难发现，data.py 里写过这条
理由，这里同一条规矩。

用本地 zip 直读、不落盘解压（DMSC 解压后可达数百 MB），所以磁盘上只留压缩包。
"""

import argparse
import codecs
import csv
import io
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np
import pyarrow.parquet as pq
import requests

from senti import config, data

# ---------------------------------------------------------------- 下载参数

CONNECT_TIMEOUT = 30   # 建连超时
READ_TIMEOUT = 180     # 单次接收超时；DMSC 是 143 MB，弱网下必须放宽
STALL_LIMIT = 3        # 连续多少次「一点没推进」就判定该源不可用
# 单个源的尝试**防呆上限**，正常情况下远不会触及：
# 真正的停止条件是「连续 STALL_LIMIT 次无进展」（源一死 3 次就放弃），
# 而只要还在推进就该继续。这个数按最坏情形定——若链路每次连接只送出一个数据块（CHUNK），
# 143 MB 的 DMSC 需要 ceil(150609472 / 65536) = 2299 次才下得完；上限必须远大于它，
# 否则就会出现「明明在推进却被判定放弃」。（评审者用本地服务端实测复现过这条，
# 并验了 5000 下第 71 次即完成一块一次的那种链路。）
MAX_ATTEMPTS = 5000
CHUNK = 1 << 16        # 64 KiB
# 单个文件的字节硬上限，防止异常响应把磁盘填满。
# DMSC 是 143.6 MB，这里留了约 3.5 倍余量。
MAX_FILE_BYTES = 512 * 1024 * 1024

# 校验用的抽样行数：解析校验只需要证明「文件确实能当语料读」，
# 不必把 387 MB 的 CSV 全解一遍。完整性由服务端声明的总字节数、
# 或登记表里的 expected_bytes 兜住（见 _fetch 与 _resolve_expected）。
VALIDATE_ROWS = 200
SNIFF_BYTES = 1 << 16   # 编码嗅探读取的样本大小


class IncompleteTransfer(Exception):
    """传输中断（收到的字节少于服务端声明的总长）。

    单独一个类型，是为了把它和「内容不可用」区分开：前者保留临时文件续传，
    后者必须丢弃重来（见 _fetch_validated）。
    """


class UnusableContent(ValueError):
    """下载完整、但内容不是可用语料（列名对不上、标签越界等）。

    单独一个类型是为了**不再重试同一个源**：内容不对，重下多少次都一样，
    而 DMSC 一次就是 143 MB。它是 ValueError 的子类，故调用方原有的
    `except ValueError` 仍然成立。"""


class Row(NamedTuple):
    """一条统一后的样本。`group` 是语料自带的分组（如电商品类），没有则为空串。"""

    text: str
    label: int
    group: str


@dataclass(frozen=True)
class FileSpec:
    """一个语料文件：落盘相对路径 + 镜像列表（按顺序回退）。"""

    relpath: str
    urls: tuple[str, ...]
    # 该文件的字节数（自 hf-mirror 的 /tree 接口核对，与本地实测一致）。
    # 用途是**兜底**：服务端既没给 Content-Length 也没给 Content-Range 时，
    # 没有它就分不出「下完了」和「被截断了」——而截断的 CSV 照样能解析出前若干行，
    # 于是坏文件会被当成语料固化下来。登记表里每个文件都必须填这一项。
    expected_bytes: int = 0


@dataclass(frozen=True)
class CorpusSpec:
    """一个语料的完整描述。"""

    key: str                 # 短名，同时是子目录名
    title: str               # 中文标题（进报告用）
    domain: str              # 领域
    files: tuple[FileSpec, ...]
    # 原始标签取值（字符串，比较前会先规范成整数字符串）→ 统一编码
    label_map: dict[str, int]
    # 读取方式：csv / parquet / zip-csv（zip 里含单个 CSV，直读不落盘）
    reader: str
    text_key: str | None = None    # 显式指定列名；None 表示自动识别
    label_key: str | None = None
    group_key: str | None = None
    # 需要**丢弃**的标签取值（如 DMSC 的 3 星中性）。
    # 把中性硬塞进二分类是在造标签，不是在标注；宁可少要数据，也不制造假样本。
    drop_labels: tuple[str, ...] = ()
    note: str = ""


# 列名自动识别时的候选（按优先级）。显式指定优先于这里。
_TEXT_KEYS = ("review", "text", "content", "sentence", "评论", "文本", "内容")
_LABEL_KEYS = ("label", "sentiment", "polarity", "标签", "情感")
_GROUP_KEYS = ("cat", "category", "类别", "品类", "领域")

_HF = "https://hf-mirror.com/datasets"

# DMSC 的原始标签是「1–5 星打分」，没有现成的二分类列，必须派生。
# 业界惯例：4–5 星 → 积极，1–2 星 → 消极；3 星是中性，**丢弃**。
# 把中性硬塞进二分类等于替数据编标签，宁可少要样本也不制造假样本。
STAR_TO_LABEL = {"1": 0, "2": 0, "4": 1, "5": 1}
STAR_NEUTRAL = "3"


def _resolve(repo: str, path: str) -> str:
    """hf-mirror 的 resolve 直链。"""
    return f"{_HF}/{repo}/resolve/main/{path}"


# ---------------------------------------------------------------- 语料登记表

# 每个文件的镜像都经过核对：**同名、同格式、同体积**（来自 hf-mirror 的
# /tree 接口）。刻意不把「体积不同的同名文件」当镜像——那等于用不同数据兜底，
# 比没有兜底更坏。DMSC 只有单一来源，如实标注。
#
# 这些语料不是训练用主料，可得性风险远低于 ChnSentiCorp，故镜像从简。
CORPORA: dict[str, CorpusSpec] = {
    "waimai_10k": CorpusSpec(
        key="waimai_10k",
        title="外卖平台评价",
        domain="外卖/餐饮服务",
        files=(
            FileSpec(
                "waimai_10k.csv",
                (
                    _resolve("XiangPan/waimai_10k", "waimai_10k.csv"),
                    _resolve("dirtycomputer/waimai_10k", "waimai_10k.csv"),
                ),
                expected_bytes=919_380,
            ),
        ),
        label_map={"1": 1, "0": 0},
        reader="csv",
        note="二分类；与酒店评论同属「服务评价」但用语差异大（配送/骑手/难吃）。",
    ),
    "weibo_senti_100k": CorpusSpec(
        key="weibo_senti_100k",
        title="微博情感语料",
        domain="社交媒体短文本",
        files=(
            FileSpec(
                "weibo_senti_100k.csv",
                (
                    _resolve("dirtycomputer/weibo_senti_100k", "weibo_senti_100k.csv"),
                    _resolve("hlsk/weibo_senti_100k", "weibo_senti_100k.csv"),
                ),
                expected_bytes=19_699_818,
            ),
        ),
        label_map={"1": 1, "0": 0},
        reader="csv",
        note="二分类；不是商品评论而是社交短文本，风格与其余语料差得最远。",
    ),
    "online_shopping_10_cats": CorpusSpec(
        key="online_shopping_10_cats",
        title="电商 10 品类评论",
        domain="电商（服装/数码/家电等 10 类）",
        files=(
            FileSpec(
                "online_shopping_10_cats.csv",
                (
                    _resolve("XiangPan/online_shopping_10_cats_62k", "online_shopping_10_cats.csv"),
                    _resolve("dirtycomputer/online_shopping_10_cats", "online_shopping_10_cats.csv"),
                ),
                expected_bytes=11_299_695,
            ),
        ),
        label_map={"1": 1, "0": 0},
        reader="csv",
        group_key="cat",
        note="二分类 + 自带品类列，适合做「同一模型跨品类」的对照。",
    ),
    "DMSC": CorpusSpec(
        key="DMSC",
        title="豆瓣电影短评",
        domain="影视评论",
        files=(
            FileSpec(
                "DMSC.csv.zip",
                (_resolve("BerlinWang/DMSC", "DMSC.csv.zip"),),
                expected_bytes=150_609_472,
            ),
        ),
        label_map=STAR_TO_LABEL,
        drop_labels=(STAR_NEUTRAL,),
        reader="zip-csv",
        text_key="Comment",
        label_key="Star",
        group_key="Movie_Name_CN",
        note="二分类由 1–5 星派生（3 星中性丢弃）；apache-2.0，四者中唯一有明确许可；体量最大。",
    ),
}


# ---------------------------------------------------------------- 路径

def corpus_dir(key: str) -> Path:
    """某个语料的落盘目录（`数据集/<key>/`）。未知 key 立刻报错，避免拼错路径。"""
    _spec(key)
    return config.DATA_ROOT / key


def _spec(key: str) -> CorpusSpec:
    if key not in CORPORA:
        raise KeyError(f"未知语料 {key!r}；已登记：{sorted(CORPORA)}")
    return CORPORA[key]


def is_downloaded(key: str) -> bool:
    """该语料的全部文件都已落盘且非空。"""
    spec = _spec(key)
    return all(
        (corpus_dir(key) / f.relpath).exists() and (corpus_dir(key) / f.relpath).stat().st_size > 0
        for f in spec.files
    )


# ---------------------------------------------------------------- 下载

def _tmp_path(dest: Path) -> Path:
    """临时文件名保留原后缀，解析校验时才能按后缀选对解析器。"""
    return dest.with_name(f"{dest.stem}.part{dest.suffix}")


def _size(path: Path) -> int:
    """文件字节数；不存在算 0。"""
    return path.stat().st_size if path.exists() else 0


def _resolve_expected(server_total: int | None, declared: int) -> int | None:
    """定出「这个文件应该有多少字节」。

    服务端声明的总长优先（权威且随上游更新而变）；服务端没给时退回登记表里
    声明的字节数；两者都没有才放弃长度校验（`declared` 为 0 也表示「没填」）。

    只信服务端会留一个洞：服务端既不给 Content-Length 也不给 Content-Range 时，
    截断的文件无法被识别——评审者用本地服务端实测复现过。故必须有登记表兜底。
    """
    if server_total is not None:
        return server_total
    return declared or None


def _fetch(url: str, tmp: Path, declared: int) -> None:
    """把 url 取到 tmp，支持断点续传，并校验收到的字节数。

    为什么必须校验总长度：本机链路会中途断开，续传后若「少了一截」，CSV 往往
    仍能解析出前若干行——于是坏文件会被当成好语料固化下来，而 is_downloaded()
    之后永远为真。用总字节数一比即可发现。
    """
    start = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Accept-Encoding": "identity"}   # 避免压缩传输让长度对不上
    if start:
        headers["Range"] = f"bytes={start}-"

    with requests.get(
        url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), stream=True, headers=headers
    ) as resp:
        if start and resp.status_code == 206:
            expected = _resolve_expected(
                _total_from_content_range(resp.headers.get("Content-Range")), declared
            )
            mode, written = "ab", start
        else:
            # 服务器不支持续传（或没要求续传）：从头写，mode=wb 会截断旧内容
            resp.raise_for_status()
            expected = _resolve_expected(
                _int_or_none(resp.headers.get("Content-Length")), declared
            )
            mode, written = "wb", 0

        too_big = False
        try:
            with tmp.open(mode) as fh:
                for chunk in resp.iter_content(chunk_size=CHUNK):
                    written += len(chunk)
                    if written > MAX_FILE_BYTES:
                        too_big = True
                        break
                    fh.write(chunk)
        except requests.RequestException as exc:
            # 传输在读完之前断掉：urllib3 会**抢先**抛出来——Content-Length 与实际不符时
            # 它抛 ChunkedEncodingError，我的长度比较根本轮不到执行。这里统一成本模块的
            # IncompleteTransfer：「半成品保留、下次带 Range 续传」这条语义对调用方应当是
            # 单一的，不该让它去分辨底层异常类型。
            if expected is not None and _size(tmp) < expected:
                raise IncompleteTransfer(
                    f"传输中断：{type(exc).__name__}: {exc}（将续传）"
                ) from exc
            raise

    if too_big:
        # 注意必须在 with 之外删：Windows 上删一个仍被打开的文件会 PermissionError
        tmp.unlink(missing_ok=True)
        raise ValueError(f"超过单文件上限 {MAX_FILE_BYTES} 字节，疑似异常内容")
    if written == 0:
        raise ValueError("响应为空")
    if expected is not None and written > expected:
        # 多出来的字节说明这个临时文件已经不可信，丢掉重来（不能在此基础上续传）
        tmp.unlink(missing_ok=True)
        raise ValueError(f"收到 {written} 字节，多于预期 {expected} 字节，已丢弃")
    if expected is not None and written < expected:
        # 少了一截：**保留**临时文件，下一次尝试会带 Range 头续传。
        # 这是弱网下载大文件的唯一活路（DMSC 有 143 MB，链路会中途重置）。
        raise IncompleteTransfer(f"传输中断：已收 {written} / {expected} 字节（将续传）")


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _total_from_content_range(value: str | None) -> int | None:
    """`bytes 100-999/1000` → 1000。取不到返回 None。"""
    if not value or "/" not in value:
        return None
    return _int_or_none(value.rsplit("/", 1)[1].strip())


def _fetch_validated(url: str, dest: Path, spec: CorpusSpec, expected_bytes: int) -> None:
    """下载 → 抽样解析校验 → 原子改名。

    失败时**删不删临时文件取决于失败发生在哪一层**，这条区分很关键：
      · 网络/传输层失败（连接重置、中断）→ **保留**临时文件，下次续传；
      · 内容层失败（解析不出语料）→ **删除**临时文件，否则续传只会把一个坏文件补完。
    与 data.py 同一条理由：若把响应体直接写进目标路径，一个 200 的 HTML 错误页
    就会被固化成语料，而 is_downloaded() 之后永远为真。
    """
    tmp = _tmp_path(dest)
    _fetch(url, tmp, expected_bytes)   # 网络层：异常直接上抛，tmp 原样保留供续传

    try:
        rows = list(_iter_rows(tmp, spec, limit=VALIDATE_ROWS))
    except Exception as exc:  # noqa: BLE001 —— 统一转成「这个源的内容不可用」
        tmp.unlink(missing_ok=True)
        raise UnusableContent(f"下载内容无法解析为语料：{type(exc).__name__}: {exc}") from exc
    if not rows:
        tmp.unlink(missing_ok=True)
        raise UnusableContent("解析后没有任何样本")

    tmp.replace(dest)


def download(key: str, force: bool = False) -> Path:
    """下载一个语料，返回其目录。已存在且 force=False 时幂等返回。"""
    spec = _spec(key)
    dest_dir = corpus_dir(key)

    if is_downloaded(key) and not force:
        print(f"[{key}] 已存在，跳过：{dest_dir}")
        return dest_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    for file_spec in spec.files:
        dest = dest_dir / file_spec.relpath
        if dest.exists() and dest.stat().st_size > 0 and not force:
            print(f"[{key}] 已存在，跳过：{dest.name}")
            continue
        _download_one(key, file_spec, dest, spec)

    return dest_dir


def _src_marker(tmp: Path) -> Path:
    """记录「这个临时文件是哪个源留下的」，用于跨进程续传时判断能不能接着用。"""
    return tmp.with_name(tmp.name + ".src")


def _download_one(key: str, file_spec: FileSpec, dest: Path, spec: CorpusSpec) -> None:
    """按优先级逐个源下载一个文件；全部失败才抛错。

    重试的停止条件不是「次数」而是「有没有进展」：每次续传若推进了字节数就继续，
    连续 STALL_LIMIT 次一点没推进才判定该源不可用。理由是本机链路会中途重置，
    143 MB 的 DMSC 需要很多次续传才拿得下来，用固定次数会既不够（大文件）
    又浪费（源已经死了还在重试）。MAX_ATTEMPTS 只是防呆硬上限。
    """
    last_error: Exception | None = None
    tmp = _tmp_path(dest)
    marker = _src_marker(tmp)

    for url in file_spec.urls:
        # 只有当临时文件**确实来自同一个源**时才接着用。换源时必须清空：
        # 不同源的字节拼在一起就是垃圾，而且还能通过长度校验。
        # （早先无条件清空的写法把「换源」和「跨进程续传」一起干掉了——
        #   上个进程留下的半成品会在第一次尝试前就被删掉。评审者实测复现过。）
        resumed = tmp.exists() and marker.exists() and marker.read_text(encoding="utf-8") == url
        if not resumed:
            tmp.unlink(missing_ok=True)
        marker.write_text(url, encoding="utf-8")

        stalls = 0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            before = tmp.stat().st_size if tmp.exists() else 0
            try:
                print(f"[{key}] 下载 {dest.name} ← {url}（第 {attempt} 次尝试）")
                _fetch_validated(url, dest, spec, file_spec.expected_bytes)
                print(f"[{key}] 已保存：{dest}（{dest.stat().st_size} 字节）")
                marker.unlink(missing_ok=True)
                return
            except UnusableContent as exc:
                # 内容不对，重下多少次都一样——直接放弃这个源，别白耗带宽
                print(f"[{key}]   内容不可用，放弃该源：{exc}")
                last_error = exc
                break
            except Exception as exc:  # noqa: BLE001 —— 记录最后一个错误，视进展决定是否继续
                last_error = exc
                after = tmp.stat().st_size if tmp.exists() else 0
                gained = after - before
                if gained > 0:
                    stalls = 0
                    print(f"[{key}]   失败：{type(exc).__name__}: {exc}｜本次推进 {gained} 字节，继续续传")
                else:
                    stalls += 1
                    print(f"[{key}]   失败：{type(exc).__name__}: {exc}｜无进展 {stalls}/{STALL_LIMIT}")
                if stalls >= STALL_LIMIT:
                    print(f"[{key}] 连续 {STALL_LIMIT} 次无进展，放弃该源")
                    break
        else:
            print(f"[{key}] 该源已达尝试上限 {MAX_ATTEMPTS}，放弃")

    # 全部源都失败时不清理 `.part` 与 `.src`：留着它们，下次运行就能从断点接着下。
    # （即使 `.part` 已被内容校验删掉、只剩 `.src`，下次也会因为「临时文件不存在」而重来，无害。）
    raise RuntimeError(f"[{key}] {dest.name} 的全部下载源均失败，最后一个错误：{last_error}")


def download_all(keys: list[str] | None = None, force: bool = False) -> list[Path]:
    """下载全部（或指定的）语料。单个失败不阻断其余，最后统一汇报。"""
    targets = keys or list(CORPORA)
    done: list[Path] = []
    failed: list[str] = []
    for key in targets:
        try:
            done.append(download(key, force=force))
        except Exception as exc:  # noqa: BLE001 —— 一个语料失败不应拖垮其余的
            print(f"[{key}] 失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            failed.append(key)
    if failed:
        print(f"\n以下语料未能完成：{failed}（可稍后重跑，已下好的不会重下）", file=sys.stderr)
    return done


# ---------------------------------------------------------------- 读取

def _sniff_bytes(raw: bytes) -> str:
    """按样本字节猜编码。中文语料常见 UTF-8（可能带 BOM）与 GBK 两种。

    必须用**增量解码器**而不是 `raw.decode(enc)`：样本按字节切，切口很可能正好落在
    一个多字节汉字中间，整段严格解码会因此假失败——实测 waimai_10k 就是这样被判成
    「既非 UTF-8 也非 GBK」，而它整文件都是合法 UTF-8。增量解码器把末尾那截不完整
    的序列留在缓冲区里，不当作错误（final=False）。
    """
    for enc in ("utf-8-sig", "gbk"):
        try:
            codecs.getincrementaldecoder(enc)().decode(raw)
            return enc
        except UnicodeDecodeError:
            continue
    raise ValueError("无法判断编码（既非 UTF-8 也非 GBK）")


def _sniff_encoding(path: Path, sample: int = SNIFF_BYTES) -> str:
    with path.open("rb") as fh:      # 必须关闭：本项目的 pytest 把 ResourceWarning 当错误
        raw = fh.read(sample)
    try:
        return _sniff_bytes(raw)
    except ValueError:
        raise ValueError(f"无法判断编码（既非 UTF-8 也非 GBK）：{path}") from None


def _open_csv_text(path: Path) -> io.TextIOBase:
    return path.open("r", encoding=_sniff_encoding(path), newline="")


BOM = "﻿"


def _deboom(lines) -> Iterator[str]:
    """逐行抹掉行首的 BOM。

    语料里**真的存在**行首 BOM（weibo 的部分记录）；`utf-8-sig` 只吃掉文件开头那一个，
    行内的会原样落进字段名或字段值——表头若带 BOM，列名就再也认不出来。
    """
    for line in lines:
        yield line[len(BOM):] if line.startswith(BOM) else line


def _clean_text(raw: object) -> str:
    """去掉值里混进的 BOM 与首尾空白。

    注意 `str.strip()` **不**把 U+FEFF 当空白字符，所以必须显式去掉。
    """
    return str(raw).replace(BOM, "").strip()


def _single_csv_name(zf: zipfile.ZipFile) -> str:
    """zip 里唯一的 .csv 成员名。多个或没有都视为异常（不猜）。"""
    names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    if len(names) != 1:
        raise ValueError(f"zip 里的 CSV 成员不是恰好 1 个：{names}")
    return names[0]


def _pick(keys: tuple[str, ...], fields, explicit: str | None, what: str) -> str:
    """选列：显式指定优先，否则按候选顺序自动识别。找不到就报错（不猜）。"""
    if explicit is not None:
        if explicit not in fields:
            raise ValueError(f"指定的{what}列 {explicit!r} 不存在，实际列：{list(fields)}")
        return explicit
    found = next((k for k in keys if k in fields), None)
    if found is None:
        raise ValueError(f"{what}列无法识别：{list(fields)}（候选：{keys}）")
    return found


def _label_key(
    raw: object,
    label_map: dict[str, int],
    where: str,
    record_no: int,
    drop: tuple[str, ...] = (),
) -> int | None:
    """原始标签 → 0/1；属于 `drop` 的档位返回 None（由调用方跳过）。

    未知取值一律报错（失败关闭）。先判「是不是整数」再取整：直接 int(float(x))
    会把 1.5 截断成 1，截断值正好落在白名单里，等于绕过校验、又变回静默错标
    （同 data.py）。
    """
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{where} 第 {record_no} 条标签无法解析：{raw!r}") from None
    if not value.is_integer():
        raise ValueError(f"{where} 第 {record_no} 条标签不是整数：{raw!r}")
    normalized = str(int(value))
    if normalized in drop:
        return None      # 明确丢弃，不是错误
    if normalized not in label_map:
        raise ValueError(
            f"{where} 第 {record_no} 条标签取值非法：{raw!r}"
            f"（本语料已声明的取值：{sorted(label_map)}，丢弃档：{sorted(drop)}）"
        )
    return label_map[normalized]


def _iter_rows(path: Path, spec: CorpusSpec, limit: int | None = None) -> Iterator[Row]:
    """按 spec 描述的格式读出统一后的样本。limit 用于抽样校验。

    每个分支都必须**关闭文件**：本项目的 pytest 把警告升级为错误，泄漏一个句柄
    就会让测试变红；在 Windows 上还会让随后的 unlink 报 WinError 32（文件被占用）。
    """
    if spec.reader == "csv":
        with _open_csv_text(path) as handle:
            yield from _iter_csv(handle, spec, path.name, limit)
    elif spec.reader == "parquet":
        yield from _iter_parquet(path, spec, limit)
    elif spec.reader == "zip-csv":
        # zip 直读，不落盘解压：DMSC 解压后可达数百 MB，磁盘上只留压缩包
        with zipfile.ZipFile(path) as zf:
            name = _single_csv_name(zf)
            with zf.open(name) as member:
                # 先嗅探编码再回到开头：压缩包里的 CSV 未必是 UTF-8
                encoding = _sniff_bytes(member.read(SNIFF_BYTES))
                member.seek(0)
                with io.TextIOWrapper(member, encoding=encoding, newline="") as text:
                    yield from _iter_csv(text, spec, path.name, limit)
    else:
        raise ValueError(f"未知的读取方式：{spec.reader!r}")


def _iter_csv(handle, spec: CorpusSpec, name: str, limit: int | None) -> Iterator[Row]:
    reader = csv.DictReader(_deboom(handle))
    fields = reader.fieldnames or []
    text_col = _pick(_TEXT_KEYS, fields, spec.text_key, "文本")
    label_col = _pick(_LABEL_KEYS, fields, spec.label_key, "标签")
    group_col = _pick(_GROUP_KEYS, fields, spec.group_key, "分组") if spec.group_key else None

    for i, record in enumerate(reader):
        if limit is not None and i >= limit:
            return
        # 先校验标签再判空文本：反过来会让「空文本 + 非法标签」被当成正常跳过，
        # 而不是触发失败关闭——那正是这条规矩要防的静默。
        label = _label_key(record.get(label_col), spec.label_map, name, i, spec.drop_labels)
        text = _clean_text(record.get(text_col, ""))
        if label is None or not text:
            continue      # 明确丢弃的档位（如 DMSC 的 3 星中性）／空评论
        yield Row(
            text=text,
            label=label,
            group=_clean_text(record.get(group_col, "")) if group_col else "",
        )


def _iter_parquet(path: Path, spec: CorpusSpec, limit: int | None) -> Iterator[Row]:
    table = pq.read_table(path)
    fields = table.column_names
    text_col = _pick(_TEXT_KEYS, fields, spec.text_key, "文本")
    label_col = _pick(_LABEL_KEYS, fields, spec.label_key, "标签")
    group_col = _pick(_GROUP_KEYS, fields, spec.group_key, "分组") if spec.group_key else None

    texts = table.column(text_col).to_pylist()
    labels = table.column(label_col).to_pylist()
    groups = table.column(group_col).to_pylist() if group_col else [""] * len(texts)

    for i, (text, label, group) in enumerate(zip(texts, labels, groups)):
        if limit is not None and i >= limit:
            return
        # 与 _iter_csv 同序：先校验标签，再判空文本（见那里的注释）
        unified = _label_key(label, spec.label_map, path.name, i, spec.drop_labels)
        cleaned = _clean_text(text)
        if unified is None or not cleaned:
            continue
        yield Row(
            text=cleaned,
            label=unified,
            group=_clean_text(group) if group_col else "",
        )


def iter_rows(key: str) -> Iterator[Row]:
    """**流式**读出该语料的全部样本。

    DMSC 解压后是 387 MB、百万行级的 CSV，一次性读进列表会吃掉几百 MB 内存，
    所以遍历大语料请用这个，不要用 load_rows()。
    """
    spec = _spec(key)
    if not is_downloaded(key):
        raise FileNotFoundError(f"语料未下载：{key}（请先 download({key!r}) 或 download_all()）")
    for file_spec in spec.files:
        yield from _iter_rows(corpus_dir(key) / file_spec.relpath, spec)


def load_rows(key: str) -> list[Row]:
    """读出全部样本为列表。小语料够用；DMSC 这类请改用 iter_rows()。"""
    return list(iter_rows(key))


# ---------------------------------------------------------------- 合并划分

def load_merged_split() -> tuple[list[str], list[int], list[str], list[int]]:
    """主语料 + 全部扩充语料**合并后全局重划分**，返回 (训练文本, 训练标签, 测试文本, 测试标签)。

    与 `data.load_split()` 同形，可直接替换训练入口里的那一句。

    为什么全局重切而不是各语料各留一份：本 profile 的定位是「用尽可能多的互联网评论
    训一个通用模型」，因此测试集也必须来自同一个混合分布——否则量出来的是
    「在 ChnSentiCorp 上有多准」，而不是「在多领域上有多准」。

    划分**按文本随机**而非按语料分层：各语料的类别比例差异极大（DMSC 77.5% 积极、
    外卖 33.4% 积极），全局随机切分能让三个集合的类别分布一致，这正是「同一分布里切三份」
    所要求的。种子固定，可复现。
    """
    texts: list[str] = []
    labels: list[int] = []

    tr_texts, tr_labels, te_texts, te_labels = data.load_split()
    texts.extend(tr_texts)
    texts.extend(te_texts)
    labels.extend(tr_labels)
    labels.extend(te_labels)
    print(f"  主语料 ChnSentiCorp：{len(tr_texts) + len(te_texts)} 条")

    for key in CORPORA:
        rows = list(iter_rows(key))
        texts.extend(row.text for row in rows)
        labels.extend(row.label for row in rows)
        print(f"  扩充语料 {key}：{len(rows)} 条")

    if not texts:
        raise ValueError("没有任何语料，无法合并划分")

    order = np.random.RandomState(config.RANDOM_SEED).permutation(len(texts))
    n_test = max(1, int(len(texts) * config.TEST_RATIO))
    test_idx, train_idx = order[:n_test], order[n_test:]

    def take(idx: np.ndarray) -> tuple[list[str], list[int]]:
        return [texts[i] for i in idx], [labels[i] for i in idx]

    train_texts, train_labels = take(train_idx)
    test_texts, test_labels = take(test_idx)

    tr_pos = sum(train_labels) / len(train_labels)
    te_pos = sum(test_labels) / len(test_labels)
    print(
        f"合并语料：共 {len(texts)} 条 → 训练 {len(train_texts)}（积极 {tr_pos:.1%}）"
        f" / 测试 {len(test_texts)}（积极 {te_pos:.1%}）"
    )
    return train_texts, train_labels, test_texts, test_labels


def load_training_split() -> tuple[list[str], list[int], list[str], list[int]]:
    """训练入口使用的划分：按 `config.CORPUS_PROFILE` 选择单语料或合并语料。"""
    if config.CORPUS_PROFILE == "merged":
        return load_merged_split()
    return data.load_split()


# ---------------------------------------------------------------- 汇总

SAMPLE_ROWS = 3        # 汇总时打印的样例条数
GROUP_PREVIEW = 12     # 分组名最多列几个（DMSC 的电影名有成千上万个）


def summarize(keys: list[str] | None = None) -> None:
    """打印每个语料的条数、标签分布与抽样，用于人工核对。"""
    for key in keys or list(CORPORA):
        spec = _spec(key)
        print(f"\n=== {key} · {spec.title}（{spec.domain}）===")
        if not is_downloaded(key):
            print("  未下载")
            continue

        total = 0
        pos = 0
        groups: set[str] = set()
        samples: list[Row] = []
        for row in iter_rows(key):      # 流式：大语料也不吃内存
            total += 1
            pos += row.label
            if row.group:
                groups.add(row.group)
            if len(samples) < SAMPLE_ROWS:
                samples.append(row)

        if total == 0:
            print("  可读样本为 0（文件在，但解析不出任何样本）")
            continue
        print(f"  条数：{total}　积极 {pos}（{pos / total:.1%}）／消极 {total - pos}")
        if groups:
            shown = sorted(groups)[:GROUP_PREVIEW]
            tail = f" …（共 {len(groups)} 个）" if len(groups) > GROUP_PREVIEW else ""
            print(f"  分组：{'、'.join(shown)}{tail}")
        for row in samples:
            head = row.text[:40].replace("\n", " ")
            print(f"  · [{row.label}] {head}…")
        if spec.note:
            print(f"  备注：{spec.note}")


def main() -> int:
    # Windows 控制台常是 GBK，而语料里可能出现它编不了的字符（如 BOM U+FEFF）。
    # 不处理的话，一条 print 就能让整个工具崩掉——README 里那条核对命令正是这样挂的。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")   # 保留控制台编码，只把编不了的字符换掉
        except (AttributeError, ValueError):       # 非文本流 / 已关闭时忽略
            pass

    parser = argparse.ArgumentParser(description="多语料下载与汇总（ChnSentiCorp 之外）")
    parser.add_argument("--keys", nargs="*", help="只处理这些语料（默认全部）")
    parser.add_argument("--force", action="store_true", help="已存在也重新下载")
    parser.add_argument("--no-download", action="store_true", help="只汇总，不下载")
    args = parser.parse_args()

    targets = args.keys or list(CORPORA)
    if not args.no_download:
        download_all(targets, force=args.force)
    summarize(targets)

    # 有语料没到位就返回非零，让脚本与 CI 能发现（原来恒返 0，等于永远报成功）
    missing = [key for key in targets if not is_downloaded(key)]
    if missing:
        print(f"\n以下语料不可用：{missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
