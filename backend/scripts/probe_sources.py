# -*- coding: utf-8 -*-
"""探测 ChnSentiCorp 的可用下载源。

已知网络事实（2026-10-11 实测）：
    hf-mirror.com              可达
    raw.githubusercontent.com  可达
    gitee.com                  可达
    huggingface.co             不可达

逐个探测候选源，打印状态码与内容长度，选定可用源后再固化为 download()。

2026-10-11 的探测结论：
    - huggingface.co 不可达，但 hf-mirror.com 可用，其 /api/datasets/<repo>
      能列出真实文件名（siblings），因此不必猜路径。
    - ChnSentiCorp 的「原生 train/test 划分」只以 parquet / arrow 形式存在；
      不存在带划分的 CSV。唯一存在的 CSV 是单一未划分文件
      ChnSentiCorp_htl_all.csv（7 766 条，酒店域子集，积极占比 0.685）。
    - 按 spec §8「沿用语料自带 train/test 划分」，最终选定 lansinuote/ChnSentiCorp
      的 parquet（train 9 600 / test 1 200，积极占比 0.500 / 0.507），
      并为此新增依赖 pyarrow。lansinuote、liming751218、liqqqqy 三个仓库的同名
      parquet 逐字节相同，故互为回退镜像。
    - 语料自带的 validation 划分不使用：按 spec §8，验证集由训练集按
      config.VAL_RATIO 现切（Task 5）。
"""

import sys

import requests

# Windows 控制台默认 GBK，中文说明会乱码成一片问号，容易看错是哪一行成功；
# 显式切到 UTF-8，保证探测结论可读。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TIMEOUT = 20

CANDIDATES: list[tuple[str, str]] = [
    # (说明, URL)
    ("HF镜像 seamew/ChnSentiCorp train",
     "https://hf-mirror.com/datasets/seamew/ChnSentiCorp/resolve/main/data/train-00000-of-00001.parquet"),
    ("HF镜像 seamew/ChnSentiCorp 文件列表",
     "https://hf-mirror.com/api/datasets/seamew/ChnSentiCorp"),
    ("HF镜像 lansinuote/ChnSentiCorp 文件列表",
     "https://hf-mirror.com/api/datasets/lansinuote/ChnSentiCorp"),
    ("GitHub raw 常见镜像（示例路径）",
     "https://raw.githubusercontent.com/lansinuote/NLPBeginnerTutorials/main/README.md"),
    ("HF镜像 数据集搜索 API",
     "https://hf-mirror.com/api/datasets?search=ChnSentiCorp"),
    # ---- 以下为探测过程中补充的候选，用于覆盖「原生 train/test 两文件」的可能形态 ----
    ("HF镜像 dirtycomputer/ChnSentiCorp 文件列表",
     "https://hf-mirror.com/api/datasets/dirtycomputer/ChnSentiCorp"),
    ("HF镜像 seamew/ChnSentiCorp test",
     "https://hf-mirror.com/datasets/seamew/ChnSentiCorp/resolve/main/data/test-00000-of-00001.parquet"),
    ("HF镜像 seamew/ChnSentiCorp validation",
     "https://hf-mirror.com/datasets/seamew/ChnSentiCorp/resolve/main/data/validation-00000-of-00001.parquet"),
    # ---- 最终选定的源（Ruling R3a：用原生划分，故取 parquet）。 ----
    # ---- 放进来是为了「重跑本脚本即可复现定源结论」，否则脚本只记录了 ----
    # ---- 一堆 404 的猜测路径，看不出到底哪个源成功了。               ----
    ("【选定】lansinuote train（2 156 438 字节，9 600 条）",
     "https://hf-mirror.com/datasets/lansinuote/ChnSentiCorp/resolve/main/data/train-00000-of-00001-02f200ca5f2a7868.parquet"),
    ("【选定】lansinuote test（275 253 字节，1 200 条）",
     "https://hf-mirror.com/datasets/lansinuote/ChnSentiCorp/resolve/main/data/test-00000-of-00001-5372924f059fe767.parquet"),
    ("【选定·备用】liming751218 train（与 lansinuote 逐字节相同）",
     "https://hf-mirror.com/datasets/liming751218/ChnSentiCorp/resolve/main/data/train-00000-of-00001-02f200ca5f2a7868.parquet"),
    ("【选定·备用】liming751218 test（与 lansinuote 逐字节相同）",
     "https://hf-mirror.com/datasets/liming751218/ChnSentiCorp/resolve/main/data/test-00000-of-00001-5372924f059fe767.parquet"),
    # ---- 备选的单一未划分 CSV（R3a 之前选的源；保留以备回溯对比） ----
    ("【备选·未采用】dirtycomputer ChnSentiCorp_htl_all CSV（2 904 858 字节）",
     "https://hf-mirror.com/datasets/dirtycomputer/ChnSentiCorp_htl_all/resolve/main/ChnSentiCorp_htl_all.csv"),
]


def main() -> int:
    for desc, url in CANDIDATES:
        try:
            r = requests.get(url, timeout=TIMEOUT, allow_redirects=True)
            print(f"[{r.status_code}] {desc}")
            print(f"        {url}")
            print(f"        content-length={len(r.content)} bytes")
            if r.status_code == 200 and len(r.content) < 4000:
                print(f"        预览: {r.text[:300]!r}")
        except Exception as exc:  # noqa: BLE001 —— 探测脚本，任何异常都只打印
            print(f"[ERR] {desc}\n        {url}\n        {type(exc).__name__}: {exc}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
