# -*- coding: utf-8 -*-
"""裁掉界面截图四周的空白（以左上角像素作为背景色基准）。

界面背景不是纯白，所以不能按「白色」判断，改为与背景色差异阈值比较。
只处理 figs/fig_4_*.png，其余图不动。
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageChops

sys.stdout.reconfigure(encoding="utf-8")

FIGS = Path(__file__).resolve().with_name("figs")
PAD = 12          # 裁剪后保留的边距（像素）
TOL = 6           # 与背景色的容差


def crop_one(p: Path) -> None:
    im = Image.open(p).convert("RGB")
    w, h = im.size
    # 取样点放在左下角——那是页面空白区，能代表真正的页面背景色
    bg = im.getpixel((6, h - 6))

    # 与背景色的逐像素差异
    diff = ImageChops.difference(im, Image.new("RGB", im.size, bg))
    gray = diff.convert("L").point(lambda v: 255 if v > TOL else 0)
    box = gray.getbbox()
    if not box:
        print(f"  {p.name}: 全同背景，跳过")
        return

    left = max(0, box[0] - PAD)
    top = max(0, box[1] - PAD)
    right = min(w, box[2] + PAD)
    bottom = min(h, box[3] + PAD)
    if (left, top, right, bottom) == (0, 0, w, h):
        print(f"  {p.name}: 无需裁剪")
        return

    im.crop((left, top, right, bottom)).save(p)
    print(f"  {p.name}: {w}x{h} -> {right - left}x{bottom - top}")


def main() -> None:
    print("裁剪界面截图：")
    for p in sorted(FIGS.glob("fig_4_*.png")):
        crop_one(p)


if __name__ == "__main__":
    main()
