# -*- coding: utf-8 -*-
"""matplotlib 中文字体配置。

Windows 上 matplotlib 默认字体不含中文字形，直接画中文会显示成方块。
这里按优先级探测系统已装的中文字体，找到后设为默认字体。
"""

import matplotlib
from matplotlib import font_manager

# 候选字体：按优先级排列，前三项为 Windows 自带
CJK_FONT_CANDIDATES = [
    "Microsoft YaHei",   # 微软雅黑，Win7+ 自带
    "SimHei",            # 黑体，Windows 自带
    "SimSun",            # 宋体，Windows 自带
    "Noto Sans CJK SC",  # 跨平台开源
    "Source Han Sans SC",
]


def _find_available_font() -> str | None:
    """返回系统上第一个可用的中文字体名，找不到则返回 None。"""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_FONT_CANDIDATES:
        if name in installed:
            return name
    return None


def setup_chinese_font() -> str | None:
    """设为 Agg 后端（只保存图片、不弹窗）并配置中文字体。

    返回实际使用的中文字体名，未找到可用字体时返回 None。
    必须在 import matplotlib.pyplot 之前调用。
    """
    matplotlib.use("Agg")

    font_name = _find_available_font()
    if font_name:
        matplotlib.rcParams["font.sans-serif"] = [font_name]
        print(f"matplotlib 中文字体：{font_name}")
    else:
        print(f"[警告] 未找到中文字体（已探测 {CJK_FONT_CANDIDATES}），图中中文可能显示为方块")

    # 使用中文字体后负号会显示异常，需显式关闭
    matplotlib.rcParams["axes.unicode_minus"] = False
    return font_name
