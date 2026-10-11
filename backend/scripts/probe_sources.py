# -*- coding: utf-8 -*-
"""探测 ChnSentiCorp 的可用下载源。

已知网络事实（2026-10-11 实测）：
    hf-mirror.com              可达
    raw.githubusercontent.com  可达
    gitee.com                  可达
    huggingface.co             不可达

逐个探测候选源，打印状态码与内容长度，选定可用源后再固化为 download()。
"""

import sys

import requests

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
