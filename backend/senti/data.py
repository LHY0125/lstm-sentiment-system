# -*- coding: utf-8 -*-
"""数据层：语料下载、解析、分词、词表构建、索引化。

对外只暴露「文本 → 张量」所需的能力，不感知模型与训练。
"""

from pathlib import Path

import pyarrow.parquet as pq
import requests

from senti import config

TIMEOUT = 60  # 单个语料文件最大约 2 MB，60 秒足够，弱网下也不至于卡死

# 单个语料文件的字节上限。语料共约 2.4 MB，留足两个数量级余量；
# 作用是防止异常响应（比如被重定向到某个大文件）把磁盘写满。
MAX_CORPUS_BYTES = 64 * 1024 * 1024

TRAIN_FILENAME = "train.parquet"
TEST_FILENAME = "test.parquet"

# 由 Task 2 Step 2 的探测结果固化；文件名取自 hf-mirror 的 /api/datasets 接口，
# 不是猜的。每个文件配多个镜像，按顺序回退——语料可得性是本方案的核心风险。
#
# 只下载 train 与 test：按 spec §8，验证集由训练集按 config.VAL_RATIO 现切
# （那段逻辑在 Task 5），所以语料自带的 validation 划分刻意不用、也不下载。
#
# 三个镜像仓库里同名文件逐字节相同（已实测 sha256 一致），选了下载量最高的
# lansinuote 做主源。注意它们同处 hf-mirror.com，能防的是「某个仓库被删/改名」，
# 防不了整站故障。
CORPUS_FILES: dict[str, tuple[str, ...]] = {
    TRAIN_FILENAME: (
        "https://hf-mirror.com/datasets/lansinuote/ChnSentiCorp/resolve/main/data/train-00000-of-00001-02f200ca5f2a7868.parquet",
        "https://hf-mirror.com/datasets/liming751218/ChnSentiCorp/resolve/main/data/train-00000-of-00001-02f200ca5f2a7868.parquet",
    ),
    TEST_FILENAME: (
        "https://hf-mirror.com/datasets/lansinuote/ChnSentiCorp/resolve/main/data/test-00000-of-00001-5372924f059fe767.parquet",
        "https://hf-mirror.com/datasets/liming751218/ChnSentiCorp/resolve/main/data/test-00000-of-00001-5372924f059fe767.parquet",
    ),
}

# 标签取值 → 统一编码（1=积极 / 0=消极）。只认这三个值，见 _parse_label。
_LABEL_MAP = {"0": 0, "-1": 0, "1": 1}


def train_path() -> Path:
    return config.DATA_DIR / TRAIN_FILENAME


def test_path() -> Path:
    return config.DATA_DIR / TEST_FILENAME


def _corpus_paths() -> list[Path]:
    return [config.DATA_DIR / name for name in CORPUS_FILES]


def _tmp_path(dest: Path) -> Path:
    """临时文件名保留原后缀，校验时才能按后缀选对解析器。"""
    return dest.with_name(f"{dest.stem}.part{dest.suffix}")


def is_downloaded() -> bool:
    """语料全部文件都已落盘且非空。"""
    return all(p.exists() and p.stat().st_size > 0 for p in _corpus_paths())


def download(force: bool = False) -> Path:
    """下载语料全部文件，返回语料所在目录。

    已存在且 force=False 时直接返回，保证幂等。
    """
    if is_downloaded() and not force:
        print(f"语料已存在，跳过下载：{config.DATA_DIR}")
        return config.DATA_DIR

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, urls in CORPUS_FILES.items():
        dest = config.DATA_DIR / name
        # 逐个文件判断，这样「训练集在、测试集缺」时只补缺的那个
        if dest.exists() and dest.stat().st_size > 0 and not force:
            print(f"已存在，跳过：{dest}")
            continue
        _download_one(name, urls, dest)

    return config.DATA_DIR


def _download_one(name: str, urls: tuple[str, ...], dest: Path) -> None:
    """按优先级逐个源下载一个文件，全部失败才抛错。"""
    last_error: Exception | None = None
    for url in urls:
        try:
            print(f"尝试下载 {name}：{url}")
            _fetch_validated(url, dest)
            print(f"已保存：{dest}（{dest.stat().st_size} bytes）")
            return
        except Exception as exc:  # noqa: BLE001 —— 逐个源回退，记录最后一个错误
            last_error = exc
            print(f"  失败：{type(exc).__name__}: {exc}")

    raise RuntimeError(f"{name} 的所有下载源均失败，最后一个错误：{last_error}")


def _fetch_validated(url: str, dest: Path) -> None:
    """下载并校验，全部通过后才原子改名落定。

    校验不通过按「和网络失败一样」处理，交给上层换下一个源。这一点很关键：
    若直接把响应体写进目标路径，一个 200 的 HTML 错误页就会被当成语料固化下来，
    而 is_downloaded() 之后会永远返回 True，坏数据再也换不掉。
    """
    tmp = _tmp_path(dest)
    try:
        size = 0
        with requests.get(url, timeout=TIMEOUT, stream=True) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    size += len(chunk)
                    if size > MAX_CORPUS_BYTES:
                        raise ValueError(
                            f"响应超过 {MAX_CORPUS_BYTES} 字节上限，疑似异常内容"
                        )
                    fh.write(chunk)

        if size == 0:
            raise ValueError("响应为空")
        # 解析校验：列名能识别、标签合法、且确实解析出样本。
        # 统一包成 ValueError：pyarrow 对坏文件抛的异常类型不止一种，
        # 但它们都只意味着「这个源不可用」，应该交给上层换下一个源，
        # 而不是把底层异常类型泄漏出去。
        try:
            rows = _read_rows(tmp)
        except Exception as exc:
            raise ValueError(f"下载内容无法解析为语料：{type(exc).__name__}: {exc}") from exc
        if not rows:
            raise ValueError("解析后没有任何样本")

        tmp.replace(dest)
    except BaseException:
        # 校验失败或中途异常都不能留下半成品，否则下次会被当成「已下载」
        tmp.unlink(missing_ok=True)
        raise


def _parse_label(raw: object, record_no: int) -> int:
    """把原始标签统一成 1=积极 / 0=消极；无法识别就报错。

    这里刻意「失败关闭」。语料里出现意料之外的取值时必须炸掉：
    静默把未知标签当成积极会持续污染训练集，而且事后极难发现。
    """
    try:
        # 先过一遍 float，让 "1.0" / 0.0 这类数值写法也能归一化成 "1" / "0"
        normalized = str(int(float(str(raw).strip())))
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"语料第 {record_no} 条记录标签无法解析：{raw!r}") from None

    if normalized not in _LABEL_MAP:
        raise ValueError(
            f"语料第 {record_no} 条记录标签取值非法：{raw!r}（期望 0 / 1 / -1）"
        )
    return _LABEL_MAP[normalized]


def _read_rows(path: Path | None = None) -> list[tuple[str, int]]:
    """读取单个语料文件，返回 (文本, 标签) 列表。

    兼容两种列名：text/label（英文）与 文本/标签（中文）。
    path 省略时读训练集，方便交互式排查。
    """
    path = path or train_path()
    if not path.exists():
        raise FileNotFoundError(f"找不到语料：{path}\n请先调用 download()")

    table = pq.read_table(path)
    fields = table.column_names
    text_key = next((k for k in ("text", "文本", "review", "content") if k in fields), None)
    label_key = next((k for k in ("label", "标签", "sentiment") if k in fields), None)
    if text_key is None or label_key is None:
        raise ValueError(f"语料列名无法识别：{fields}（需要 text/label 或 文本/标签）")

    texts = table.column(text_key).to_pylist()
    labels = table.column(label_key).to_pylist()
    return [(str(text), _parse_label(label, i)) for i, (text, label) in enumerate(zip(texts, labels))]


def load_split() -> tuple[list[str], list[int], list[str], list[int]]:
    """载入语料并沿用其自带的 train/test 划分（spec §8 / Ruling R3a）。

    不再合并重切：原生划分才能让测试集指标与公开的 ChnSentiCorp 基线对齐。
    语料自带的 validation 划分同样不使用——验证集由训练集按 config.VAL_RATIO
    现切（Task 5），避免引入第三个来源。

    返回 (train_texts, train_labels, test_texts, test_labels)。
    """
    print("使用划分方式：语料自带 train/test 划分（未重切）")

    train_rows = _read_rows(train_path())
    test_rows = _read_rows(test_path())
    if not train_rows or not test_rows:
        raise ValueError("语料为空")

    train_texts = [t for t, _ in train_rows]
    train_labels = [y for _, y in train_rows]
    test_texts = [t for t, _ in test_rows]
    test_labels = [y for _, y in test_rows]
    print(f"语料载入：训练 {len(train_texts)} 条 / 测试 {len(test_texts)} 条")
    return train_texts, train_labels, test_texts, test_labels
