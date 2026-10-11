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
    - ChnSentiCorp 的「原生 train/test 划分」只以 parquet / arrow 形式存在，
      读取需要 pyarrow（本环境未安装）；带划分的 CSV 不存在。
    - 唯一存在的 CSV 是单一未划分文件 ChnSentiCorp_htl_all.csv（7 766 条）。
      故 download() 固化为该 CSV —— 依赖零新增，划分改走固定种子 9:1 重切。
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
    # ---- 最终选定的两个源。放进来是为了「重跑本脚本即可复现定源结论」，----
    # ---- 否则脚本只记录了一堆 404 的猜测路径，看不出到底哪个源成功了。  ----
    ("【选定】HF镜像 dirtycomputer/ChnSentiCorp_htl_all CSV（2 904 858 字节）",
     "https://hf-mirror.com/datasets/dirtycomputer/ChnSentiCorp_htl_all/resolve/main/ChnSentiCorp_htl_all.csv"),
    ("【选定·备用】GitHub raw SophonPlus/ChineseNlpCorpus 同一文件",
     "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv"),
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
