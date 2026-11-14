# -*- coding: utf-8 -*-
"""课程设计报告生成器：把 content_part*.py 的内容写进学院模板的副本。

设计原则
--------
- **改 document.xml，并按格式规范补页眉页脚部件**。模板的封面、评分表、样式表、
  分节符原样保留；模板本身没有页眉，这里新建页眉/页脚部件并挂到各节上，
  同时把章节分节改成 oddPage，满足「每章开始于奇数页」。
- 内容与排版分离：本文件只负责「怎么排」，文字全部来自 content_part*.py。
- 图片按需注册：缺图时跳过并告警，不中断构建，便于先出文字稿。

用法
----
    python build_report.py                 # 生成报告 docx
    python build_report.py --check         # 只做自检，不写文件
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

import pyhighlight

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]                       # backend/scripts/report -> 项目根
TEMPLATE = PROJECT / "docs" / "课程设计报告模板 - 神经网络与深度学习.docx"
OUTPUT = PROJECT / "docs" / "《神经网络与深度学习》课程设计报告.docx"

# ── 封面字段（按需修改）────────────────────────────────────────────────
COVER = {
    "题目": "基于 LSTM 的网络评论情感分类系统设计与实现",
    "学生姓名": "刘航宇",
    "专业班级": "",          # 留空则保留下划线，打印前手填
    "学生学号": "",
    "教师姓名": "侯蓓蓓",
    "日期": "2026 年 10 月 30 日",
}

# 是否自动填写封面信息。
# 默认 False —— 按用户要求保留模板原样的空下划线，打印前手写。
# 想自动填时改成 True，上面 COVER 字典里的值会写进封面。
FILL_COVER = False

# ── XML 基础 ──────────────────────────────────────────────────────────
NS_W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def esc(s: str) -> str:
    """XML 文本转义。"""
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;"))


def _png_size(p: Path) -> tuple[int, int] | None:
    """读 PNG 的 IHDR 取 (宽, 高)。不装 Pillow 也能拿到尺寸。

    IHDR 固定是第一个块：签名 8 字节 + 长度 4 + 类型 4 + 宽 4 + 高 4，
    所以宽高分别落在 [16:20] 和 [20:24]，大端无符号整数。
    """
    try:
        head = p.open("rb").read(24)
    except OSError:
        return None
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return (int.from_bytes(head[16:20], "big"),
            int.from_bytes(head[20:24], "big"))


def _svg_size(p: Path) -> tuple[float, float] | None:
    """读 SVG 根元素的 width/height（或 viewBox），拿真实宽高比。

    draw.io 导出的是 `width="1062px" height="722px"`；若只有 viewBox，
    就用 viewBox 的宽高。Word 的 wp:extent 是显示尺寸，必须按这个比例算，
    否则 SVG 会被非等比拉伸。
    """
    try:
        head = p.read_text(encoding="utf-8", errors="ignore")[:16000]
    except OSError:
        return None
    # draw.io 的根 <svg> 标签带一个巨大的 content="..."（整张图的 mxfile），
    # 所以不能先匹配完整的 <svg ...> 再取属性；width/height 在 content 之前，
    # 直接在文件头里找第一组根属性即可。
    mw = re.search(r'\bwidth="([\d.]+)(?:px)?"', head)
    mh = re.search(r'\bheight="([\d.]+)(?:px)?"', head)
    if mw and mh:
        try:
            w, h = float(mw.group(1)), float(mh.group(1))
            if w > 0 and h > 0:
                return w, h
        except ValueError:
            pass
    mv = re.search(r'\bviewBox="([-\d.\s,]+)"', head)
    if mv:
        parts = [x for x in re.split(r"[\s,]+", mv.group(1).strip()) if x]
        if len(parts) == 4:
            try:
                w, h = float(parts[2]), float(parts[3])
                if w > 0 and h > 0:
                    return w, h
            except ValueError:
                pass
    return None


# 中文与英文/数字之间不留空格（用户的行文习惯）。
# 只清洗 CJK↔ASCII 的边界：纯 ASCII 内部的空间（公式、英文短语）原样保留，
# 代码块走 raw=True 完全不动，否则会和仓库里的真实源码对不上。
_CJK = r"\u3400-\u9fff"   # CJK Unified Ideographs; Python re interprets \uXXXX in patterns
_SPACE_CJK_TO_ASCII = re.compile(rf"(?<=[{_CJK}])[ \t]+(?=[A-Za-z0-9])")
_SPACE_ASCII_TO_CJK = re.compile(rf"(?<=[A-Za-z0-9])[ \t]+(?=[{_CJK}])")


def tight(s: str) -> str:
    """去掉中文与英文/数字之间的空格。

    封面与参考文献专用：前者要保持模板的下划线排版，后者要保留
    「[4] 作者」这类国标引用格式的空格。
    """
    return _SPACE_ASCII_TO_CJK.sub("", _SPACE_CJK_TO_ASCII.sub("", s))


# 正文清洗（比封面严格）：CJK（含全角标点）与任意「西文 token」之间都不留空格。
# 西文 token 包括 ASCII 可见字符、希腊字母、箭头与常用数学符号；
# 这样「clamp(min=0) 防御」「编码成 (N, 150) 的矩阵」「概率 → 标签」
# 都会收紧成中文排版习惯的样子。纯 ASCII 内部的空格（英文短语、版本号）不动。
_CJK_BODY = r"、-〿㐀-鿿！-｠￠-￦"
_TOKEN = (r"A-Za-z0-9_\x21-\x2f\x3a-\x40\x5b-\x60\x7b-\x7e"
          r"Ͱ-Ͽ←-⇿"
          r"×÷−≤≥≠≈∑∏⊙")
_SPACE_CJK_TO_TOKEN = re.compile(rf"(?<=[{_CJK_BODY}])[ \t]+(?=[{_TOKEN}])")
_SPACE_TOKEN_TO_CJK = re.compile(rf"(?<=[{_TOKEN}])[ \t]+(?=[{_CJK_BODY}])")
_SPACE_AFTER_PUNCT = re.compile(rf"(?<=[。！？；：，、])[ \t]+(?=[{_CJK_BODY}])")
_SPACE_IN_BRACKET = re.compile(r"(?<=[（【《「『])[ \t]+|[ \t]+(?=[）】》」』])")
# 「1 = 积极」「准确率 ≥ 85%」：把紧挨中文的运算符两侧空格一并去掉
_SPACE_OP_RIGHT = re.compile(
    rf"(?<=[A-Za-z0-9])[ \t]+([=+\-*/<>])[ \t]*(?=[{_CJK_BODY}])")
_SPACE_OP_LEFT = re.compile(
    rf"(?<=[{_CJK_BODY}])[ \t]*([=+\-*/<>])[ \t]+(?=[A-Za-z0-9])")
# 乘号、比较号、箭头两侧若有一侧是中文，两侧空格一起删（层数 × 3 → 层数×3）
_SYM = r"×÷−≤≥≠≈∑∏⊙←-⇿"
_SPACE_SYM_R = re.compile(
    rf"(?<=[{_CJK_BODY}])[ \t]*([{_SYM}])[ \t]*(?=[A-Za-z0-9])")
_SPACE_SYM_L = re.compile(
    rf"(?<=[A-Za-z0-9])[ \t]*([{_SYM}])[ \t]*(?=[{_CJK_BODY}])")
_SPACE_SYM_C = re.compile(
    rf"(?<=[{_CJK_BODY}])[ \t]*([{_SYM}])[ \t]*(?=[{_CJK_BODY}])")


def tight_body(s: str) -> str:
    """正文用：清掉中文与西文 token 之间的空格（不碰纯 ASCII 内部）。"""
    s = _SPACE_CJK_TO_TOKEN.sub("", s)
    s = _SPACE_TOKEN_TO_CJK.sub("", s)
    s = _SPACE_AFTER_PUNCT.sub("", s)
    s = _SPACE_IN_BRACKET.sub("", s)
    s = _SPACE_OP_RIGHT.sub(r"\1", s)
    s = _SPACE_OP_LEFT.sub(r"\1", s)
    s = _SPACE_SYM_R.sub(r"\1", s)
    s = _SPACE_SYM_L.sub(r"\1", s)
    s = _SPACE_SYM_C.sub(r"\1", s)
    return s


_WT = re.compile(r"(<w:t(?: [^>]*)?>)([^<]*)(</w:t>)")


def tight_xml_texts(xml: str) -> str:
    """对一段 XML 里所有 w:t 的文字做 tight()，标签原样保留。

    用于封面：那几块是从模板直接拷来的原文，不经过 run()，
    但排版规则应当全文一致。
    """
    return _WT.sub(lambda m: m.group(1) + tight(m.group(2)) + m.group(3), xml)


# 格式规范：中文用宋体/黑体，数字与英文用 Times New Roman。
# 这两者必须分别写进 rFonts 的 eastAsia 与 ascii/hAnsi，否则 Word 会把
# ascii 也套成中文字体，数字和英文看起来就是「宋体的数字」。
LATIN_OF = {
    "宋体": "Times New Roman",
    "黑体": "黑体",               # 一级标题整体黑体，编号也跟着黑体
    "Times New Roman": "Times New Roman",
}
# 公式等场景指定「拉丁用 TNR」时，中文仍应落在宋体上，不能跟着 TNR 走。
EAST_OF = {"Times New Roman": "宋体"}


def run(text: str, *, bold=False, size=None, font=None, latin=None, color=None,
        underline=False, raw=False) -> str:
    """构造一个 w:r。size 单位是半磅（小四=24，五号=21，四号=28）。

    正文默认走 tight_body()：中文与英文/数字之间不留空格。
    raw=True 用于代码、公式与参考文献，保留原有空格。
    """
    if not raw:
        text = tight_body(text)
    rpr = []
    if font:
        latin = latin or LATIN_OF.get(font, font)
        east = EAST_OF.get(font, font)
        rpr.append(f'<w:rFonts w:ascii="{latin}" w:hAnsi="{latin}" '
                   f'w:eastAsia="{east}" w:cs="{latin}" w:hint="eastAsia"/>')
    if bold:
        rpr.append("<w:b/>")
    if underline:
        rpr.append('<w:u w:val="single"/>')
    if color:
        rpr.append(f'<w:color w:val="{color}"/>')
    if size:
        rpr.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
    rpr_xml = f"<w:rPr>{''.join(rpr)}</w:rPr>" if rpr else ""
    return (f"<w:r>{rpr_xml}"
            f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r>')


def runs_from_text(text: str, **kw) -> str:
    """把 `**加粗**` 标记解析成多个 run。

    先对**整段**文本做 tight_body，再按 `**` 切分：否则空格正好落在
    加粗段边界时（「增量是 **18 个百分点**」），逐 run 清洗看不到另一侧，
    空格会残留下来。
    """
    text = tight_body(text)
    out = []
    for i, seg in enumerate(text.split("**")):
        if not seg:
            continue
        out.append(run(seg, bold=(i % 2 == 1) or kw.get("bold", False), **{
            k: v for k, v in kw.items() if k != "bold"
        }))
    return "".join(out)


def para(inner: str, *, style=None, jc=None, spacing=None, ind=None,
         shd=None, tabs=None, outline=None, keep_next=False,
         extra_ppr="") -> str:
    """构造一个 w:p。

    shd 单独给，是因为 CT_PPr 的子元素顺序被 schema 强制：
    pStyle → keepNext → … → shd → … → spacing → ind → jc → outlineLvl → rPr。
    制表位（tabs）必须排在 shd 之后、spacing 之前，公式的「居中 + 右对齐编号」
    就靠它实现。
    底纹若跟着 spacing 之后发射，Word 能打开但 XSD 校验会报错。

    outline 直接写 w:outlineLvl，让段落进目录。按 0 起算：
    章标题用模板样式 1（自带 0 级），节用 outline=1，
    目录域配 \\o "1-2" 就只收这两级。
    """
    ppr = []
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    if keep_next:
        ppr.append("<w:keepNext/>")
    if shd:
        ppr.append(shd)
    if tabs:
        ppr.append(tabs)
    if spacing:
        ppr.append(spacing)
    if ind:
        ppr.append(ind)
    if jc:
        ppr.append(f'<w:jc w:val="{jc}"/>')
    if outline is not None:
        ppr.append(f'<w:outlineLvl w:val="{outline}"/>')
    ppr.append(extra_ppr)          # 段的 w:rPr 必须是 pPr 的最后一个子元素
    ppr_xml = f"<w:pPr>{''.join(ppr)}</w:pPr>" if any(ppr) else ""
    return f"<w:p>{ppr_xml}{inner}</w:p>"


# ── 代码清单 ──────────────────────────────────────────────────────────
# 版式完全照用户的 Word 模板（模板.docx）：Consolas + 宋体（eastAsia）、
# 五号（w:sz=21，即 10.5 pt）、白色底纹、段后 0、行距 300 atLeast、基线对齐。
# 缩进靠正文里的真实空格 + xml:space="preserve"——模板用的是不换行空格 U+00A0，
# 那个从文档里复制出去会变成非法 Python 缩进，所以这里刻意用普通空格。
CODE_SZ = 21          # 半磅 → 10.5 pt，与模板一致
CODE_FONT = "Consolas"
CODE_EA_FONT = "宋体"


def code_run(text: str, color: str, bold: bool) -> str:
    """一个带颜色的代码 run。rPr 顺序按 CT_RPr 约束：rFonts → b → color → kern → sz。"""
    rpr = (f'<w:rPr><w:rFonts w:ascii="{CODE_FONT}" w:hAnsi="{CODE_FONT}" '
           f'w:eastAsia="{CODE_EA_FONT}" w:cs="{CODE_EA_FONT}"/>')
    if bold:
        rpr += "<w:b/>"
    rpr += (f'<w:color w:val="{color}"/><w:kern w:val="0"/>'
            f'<w:sz w:val="{CODE_SZ}"/><w:szCs w:val="{CODE_SZ}"/></w:rPr>')
    return f'<w:r>{rpr}<w:t xml:space="preserve">{esc(text)}</w:t></w:r>'


def code_para(runs: list[tuple[str, str, bool]], indent: int = 0) -> str:
    """一行代码 = 一个段落。空行也给一个空格，否则行高会塌掉。

    indent 是该行自身的缩进字符数（空格数）。把它写成悬挂缩进后，
    折行产生的续行会**正好落在这一行的缩进位置上**，而不是顶回行首
    或者掉到一个和它无关的固定位置。
    """
    if runs:
        body = "".join(code_run(t, c, b) for t, c, b in runs)
    else:
        body = code_run(" ", "000000", False)

    # Consolas 10.5 pt 的字符步进是 0.55 em = 5.775 pt = 115.5 缇。
    ind = int(round(min(indent, 16) * 115.5))
    ind_xml = f'<w:ind w:left="{ind}" w:hanging="{ind}"/>' if ind else ""
    ppr = ('<w:pPr><w:widowControl/>'
           '<w:shd w:val="clear" w:color="auto" w:fill="FFFFFF"/>'
           '<w:spacing w:after="0" w:line="300" w:lineRule="atLeast"/>'
           f'{ind_xml}'
           # 必须显式左对齐：文档默认是两端对齐，代码行一旦折行，
           # 上一行会被拉成「词与词之间隔一大块」的样子，比折行还难看。
           '<w:jc w:val="left"/>'
           '<w:textAlignment w:val="baseline"/></w:pPr>')
    return f"<w:p>{ppr}{body}</w:p>"


# 行距：正文 1.3 倍 ≈ line=312 (240 * 1.3)；模板用 360 = 1.5 倍
SP_BODY = '<w:spacing w:line="312" w:lineRule="auto"/>'
SP_H2 = '<w:spacing w:line="360" w:lineRule="auto" w:before="120" w:after="60"/>'
SP_H3 = '<w:spacing w:line="312" w:lineRule="auto" w:before="60" w:after="30"/>'
SP_SINGLE = '<w:spacing w:line="240" w:lineRule="auto" w:before="40" w:after="40"/>'

FONT_HEI = "黑体"
FONT_SONG = "宋体"
FONT_TNR = "Times New Roman"
FONT_MONO = "Consolas"

# 版心宽 8306 缇（A4 11906 − 左右边距 1800×2）。公式用两个制表位：
# 居中位放在正中，右对齐位放在版心右边，编号就正好压住行末。
TAB_RUN = "<w:r><w:tab/></w:r>"
FORMULA_TABS = ('<w:tabs><w:tab w:val="center" w:pos="4153"/>'
                '<w:tab w:val="right" w:pos="8306"/></w:tabs>')

SZ_S4 = 24     # 小四 12pt
SZ_HAO5 = 21   # 五号 10.5pt
SZ_HAO4 = 28   # 四号 14pt
SZ_X3 = 30     # 小三 15pt
SZ_CODE = 18   # 代码清单 9pt —— 再大长行就会折行，版面上很难看


# ── 块渲染 ────────────────────────────────────────────────────────────
class Renderer:
    def __init__(self) -> None:
        self.image_rels: list[tuple[str, str, Path]] = []   # (rId, media_name, src)
        self.link_rels: list[tuple[str, str]] = []          # (rId, url)
        self.next_rid = 900
        self.next_img = 100
        self.svg_used = 0
        self.warnings: list[str] = []
        self.fig_no = {}       # 章号 -> 计数
        self.tab_no = {}
        self.formula_no = {}
        self.code_no = {}

    def resolve(self, path: str) -> Path | None:
        p = Path(path)
        for cand in (HERE / p, PROJECT / p, p):
            if cand.is_file():
                return cand
        return None

    def svg_for(self, png_path: Path) -> Path | None:
        """找与 figs/x.png 同名的 drawio/x.drawio.svg（用户手动导出的矢量图）。

        图源仍是 drawio 目录；figs/ 下的 png 只作为旧版 Word 的回退位图。
        不在脚本里重新导出 SVG——用户会直接在 draw.io 里改图并重新导出。
        """
        name = png_path.stem + ".drawio.svg"
        for base in (png_path.parent.parent, HERE, PROJECT):
            cand = base / "drawio" / name
            if cand.is_file():
                return cand
        return None

    def image_drawing_svg(self, svg_path: Path, fallback_png: Path,
                          width_cm: float) -> str:
        """Word 2016+ 的 SVG 图片：PNG 回退 + asvg:svgBlip 指向矢量图。

        Word 打开/导出 PDF 时优先用 svgBlip；旧版 Word 退回到 PNG。
        SVG 与 PNG 都只被复制进 docx，不修改源文件。
        """
        self.next_rid += 1
        svg_rid = f"rId{self.next_rid}"
        self.next_rid += 1
        png_rid = f"rId{self.next_rid}"
        self.next_img += 1
        base = f"image{self.next_img}"
        svg_media = base + ".svg"
        png_media = base + ".png"
        self.image_rels.append((svg_rid, svg_media, svg_path))
        self.image_rels.append((png_rid, png_media, fallback_png))

        cx = int(width_cm * 360000)
        cy = int(cx * 0.62)
        size = _svg_size(svg_path)
        if size and size[0] > 0:
            cy = int(cx * size[1] / size[0])
        return (
            '<w:r><w:drawing>'
            '<wp:inline distT="0" distB="0" distL="0" distR="0" '
            'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">'
            f'<wp:extent cx="{cx}" cy="{cy}"/>'
            '<wp:docPr id="%d" name="Picture %d"/>'
            '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:nvPicPr><pic:cNvPr id="%d" name="%s"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{png_rid}" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'xmlns:asvg="http://schemas.microsoft.com/office/drawing/2016/SVG/main">'
            '<a:extLst><a:ext uri="{96DAC541-7B7A-43D3-8B79-37D633B846F1}">'
            f'<asvg:svgBlip r:embed="{svg_rid}"/></a:ext></a:extLst>'
            '</a:blip>'
            '<a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>'
        ) % (self.next_img, self.next_img, self.next_img, svg_media)

    def image_drawing(self, img_path: Path, width_cm: float) -> str:
        self.next_rid += 1
        self.next_img += 1
        rid = f"rId{self.next_rid}"
        media = f"image{self.next_img}.png"
        self.image_rels.append((rid, media, img_path))

        # EMU: 1 cm = 360000 EMU
        cx = int(width_cm * 360000)
        # wordprocessingDrawing 的 wp:extent 是**显示尺寸**，Word 不会按图片
        # 原始比例纠正它，所以 cy 必须自己按真实高宽算，否则图会被非等比拉伸。
        cy = int(cx * 0.62)                     # 读不到尺寸时的兜底比例
        size = _png_size(img_path)
        if size:
            cy = int(cx * size[1] / size[0])
        return (
            '<w:r><w:drawing>'
            '<wp:inline distT="0" distB="0" distL="0" distR="0" '
            'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">'
            f'<wp:extent cx="{cx}" cy="{cy}"/>'
            '<wp:docPr id="%d" name="Picture %d"/>'
            '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:nvPicPr><pic:cNvPr id="%d" name="%s"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>'
            '<a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>'
        ) % (self.next_img, self.next_img, self.next_img, media)

    def table(self, blk: dict) -> str:
        header, rows = blk["header"], blk["rows"]
        ncol = len(header)
        total = 8300                      # 可用宽度（缇），页面 11906 - 左右边距 3600

        # 列宽按权重分配：最后一列通常是较长的说明文字，给它双份；
        # 列数很多时（如 8 列的实验结果表）再按比例分反而更挤，改为均分。
        weights = [1] * (ncol - 1) + [2] if ncol <= 5 else [1] * ncol
        unit = total / sum(weights)
        widths = [int(unit * w) for w in weights]
        widths[-1] += total - sum(widths)          # 余数补到最后一列

        def cell(text: str, w: int, bold=False, head=False) -> str:
            # CT_TcPr 的子元素同样有固定顺序：tcW → tcBorders → shd → vAlign。
            # 三线表只在表头下方加一条细线（栏目线），其余横线靠表格级 borders 关掉。
            tcpr = f'<w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>'
            if head:
                tcpr += ('<w:tcBorders><w:bottom w:val="single" w:sz="6" '
                         'w:space="0" w:color="000000"/></w:tcBorders>')
            tcpr += '<w:vAlign w:val="center"/></w:tcPr>'
            return ("<w:tc>" + tcpr
                    + para(runs_from_text(str(text), size=SZ_HAO5,
                                          font=FONT_SONG, bold=bold),
                           jc="center", spacing=SP_SINGLE)
                    + "</w:tc>")

        grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)

        # cantSplit：一行不跨页拆开；tblHeader：表头在续页重复
        head_tr = ('<w:tr><w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>'
                   + "".join(cell(h, w, bold=True, head=True)
                             for h, w in zip(header, widths))
                   + "</w:tr>")
        body_trs = []
        for r in rows:
            body_trs.append('<w:tr><w:trPr><w:cantSplit/></w:trPr>'
                            + "".join(cell(c, w) for c, w in zip(r, widths))
                            + "</w:tr>")

        # 三线表：顶线 + 栏目线（表头下，见 cell 的 tcBorders）+ 底线。
        # 无竖线、无行间横线、无底纹。线宽按国内期刊惯例：
        # 顶/底线 1.5 pt（w:sz 以 1/8 pt 计，故 12），栏目线 0.75 pt（6）。
        tbl = (
            '<w:tbl><w:tblPr><w:tblStyle w:val="a6"/>'
            f'<w:tblW w:w="{total}" w:type="dxa"/><w:jc w:val="center"/>'
            '<w:tblBorders>'
            '<w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
            '<w:left w:val="nil"/>'
            '<w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
            '<w:right w:val="nil"/>'
            '<w:insideH w:val="nil"/>'
            '<w:insideV w:val="nil"/>'
            '</w:tblBorders></w:tblPr>'
            f"<w:tblGrid>{grid}</w:tblGrid>{head_tr}{''.join(body_trs)}</w:tbl>"
        )
        # 表名在表格上方居中：五号宋体、单倍行距（格式规范第 4 条）
        cap = para(run(blk["caption"], size=SZ_HAO5, font=FONT_SONG),
                   jc="center", spacing=SP_SINGLE)
        return cap + tbl + para(run("", size=SZ_HAO5), spacing=SP_SINGLE)

    # ── 代码清单 ──────────────────────────────────────────────────────
    def code_source(self, blk: dict) -> tuple[str, str]:
        """取出清单源码，返回 (源码, 语言)。

        优先用 `file` + `lines` 直接从仓库读：报告里的代码就是仓库里的代码，
        不存在「抄进文档之后又被改了一次」的漂移。`text` 仍保留，
        给需要摘录/拼接的少数场合用。
        """
        if "file" not in blk:
            return blk.get("text", ""), blk.get("lang", "python")

        path = self.resolve(blk["file"])
        if path is None:
            self.warnings.append(f"缺源码：{blk['file']}")
            return "", "python"

        all_lines = path.read_text(encoding="utf-8").splitlines()
        a, b = blk.get("lines", (1, len(all_lines)))
        if not (1 <= a <= b <= len(all_lines)):
            self.warnings.append(
                f"行号越界：{blk['file']} {a}-{b}（文件共 {len(all_lines)} 行）")
            return "", "python"

        lang = blk.get("lang")
        if lang is None:
            lang = "typescript" if path.suffix in (".ts", ".tsx") else "python"
        return "\n".join(all_lines[a - 1:b]), lang

    def code(self, blk: dict, chapter: str) -> str:
        """代码清单：居中题注 + 逐行高亮段落。编号按出现顺序自动生成。"""
        self.code_no[chapter] = self.code_no.get(chapter, 0) + 1
        caption = f"代码 {chapter}-{self.code_no[chapter]} {blk['caption']}"
        src, lang = self.code_source(blk)

        out = [para(run(caption, size=SZ_HAO5, bold=True),
                    jc="center", spacing=SP_SINGLE, keep_next=True)]
        for src_line, runs in zip(src.splitlines(),
                                  pyhighlight.highlight_block(src, lang)):
            indent = len(src_line) - len(src_line.lstrip(" "))
            out.append(code_para(runs, indent))
        out.append(para(run("", size=SZ_HAO5), spacing=SP_SINGLE))
        return "".join(out)

    # ── 正文超链接 ────────────────────────────────────────────────────
    # 正文里用 [[URL|显示文字]] 标记可点击链接：rich_runs 把它换成
    # <w:hyperlink r:id>，并在 document.xml.rels 里登记 TargetMode=External。
    # 不把 URL 直接写成纯文本——纯文本在 Word 里不会自动变成链接。
    LINK_RE = re.compile(r"\[\[([^\[\]|]+)\|([^\[\]]+)\]\]")

    def hyperlink(self, url: str, label: str, *, size=None, font=None) -> str:
        """一个可点击的超链接 run。文字用 Word 默认链接色 + 单下划线。

        rPr 顺序按 CT_RPr 约束：rFonts → color → sz → u。
        label 不做 tight_body：URL 里的 / . : 必须原样保留。
        """
        self.next_rid += 1
        rid = f"rId{self.next_rid}"
        self.link_rels.append((rid, url))

        rpr = []
        if font:
            latin = LATIN_OF.get(font, font)
            east = EAST_OF.get(font, font)
            rpr.append(f'<w:rFonts w:ascii="{latin}" w:hAnsi="{latin}" '
                       f'w:eastAsia="{east}" w:cs="{latin}" w:hint="eastAsia"/>')
        rpr.append('<w:color w:val="0563C1"/>')
        if size:
            rpr.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
        rpr.append('<w:u w:val="single"/>')
        return (f'<w:hyperlink r:id="{rid}" w:history="1">'
                f'<w:r><w:rPr>{"".join(rpr)}</w:rPr>'
                f'<w:t xml:space="preserve">{esc(label)}</w:t></w:r>'
                '</w:hyperlink>')

    def rich_runs(self, text: str, **kw) -> str:
        """解析 `[[URL|文字]]` 并生成超链接，其余部分交给 runs_from_text。

        按链接标记切段后分别清洗：链接前后的中文与 URL 之间不会因此多出空格。
        """
        out = []
        pos = 0
        for m in self.LINK_RE.finditer(text):
            if m.start() > pos:
                out.append(runs_from_text(text[pos:m.start()], **kw))
            link_kw = {k: v for k, v in kw.items() if k in ("size", "font")}
            out.append(self.hyperlink(m.group(1), m.group(2), **link_kw))
            pos = m.end()
        if pos < len(text):
            out.append(runs_from_text(text[pos:], **kw))
        return "".join(out)

    def render(self, blocks: list[dict], chapter: str) -> str:
        out = []
        for blk in blocks:
            t = blk["type"]

            if t == "h2":
                out.append(para(run(blk["text"], bold=True, size=SZ_HAO4,
                                    font=FONT_SONG),
                                jc="left", spacing=SP_H2, outline=1))
            elif t == "h3":
                out.append(para(run(blk["text"], bold=True, size=SZ_S4,
                                    font=FONT_SONG),
                                jc="left", spacing=SP_H3))
            elif t == "p":
                jc = blk.get("align", "both")
                out.append(para(self.rich_runs(blk["text"], size=SZ_S4,
                                               font=FONT_SONG),
                                jc=jc, spacing=SP_BODY,
                                ind='<w:ind w:firstLineChars="200" w:firstLine="480"/>'))
            elif t == "list":
                for i, item in enumerate(blk["items"], 1):
                    out.append(para(self.rich_runs(f"（{i}）{item}", size=SZ_S4,
                                                   font=FONT_SONG),
                                    jc="both", spacing=SP_BODY,
                                    ind='<w:ind w:leftChars="200" w:left="480" '
                                        'w:hangingChars="0" w:hanging="0"/>'))
            elif t == "formula":
                self.formula_no[chapter] = self.formula_no.get(chapter, 0) + 1
                no = f"（{chapter}-{self.formula_no[chapter]}）"
                # 公式另起一行居中，编号放右边行末：居中制表位 + 右对齐制表位。
                # 公式正文走 raw=True，否则 tight() 会吃掉数学式里的空格。
                out.append(para(
                    TAB_RUN
                    + run(blk["text"], size=SZ_S4, font=FONT_TNR, raw=True)
                    + TAB_RUN
                    + run(no, size=SZ_S4, font=FONT_SONG),
                    jc="left", spacing=SP_BODY, tabs=FORMULA_TABS))
            elif t == "code":
                out.append(self.code(blk, chapter))
            elif t == "table":
                out.append(self.table(blk))
            elif t == "figure":
                img = self.resolve(blk["path"])
                if img is None:
                    self.warnings.append(f"缺图：{blk['path']}（{blk['caption']}）")
                    continue
                width = blk.get("width_cm", 14)
                svg = self.svg_for(img) if img.suffix.lower() == ".png" else None
                if svg is not None:
                    # 优先用 drawio/ 下用户手动导出的矢量图；figs/*.png 只做回退
                    self.svg_used += 1
                    drawing = self.image_drawing_svg(svg, img, width)
                else:
                    drawing = self.image_drawing(img, width)
                out.append(para(drawing, jc="center", spacing=SP_SINGLE))
                # 图名放在图下方居中：五号宋体、单倍行距（格式规范第 4 条）
                out.append(para(run(blk["caption"], size=SZ_HAO5, font=FONT_SONG),
                                jc="center", spacing=SP_SINGLE))
            elif t == "ref":
                # 参考文献保留国标格式里的空格（[4] 作者、出版地: 出版社）
                out.append(para(run(blk["text"], size=SZ_HAO5, font=FONT_SONG,
                                    raw=True),
                                jc="left", spacing=SP_SINGLE))
            else:
                self.warnings.append(f"未知块类型：{t}")
        return "".join(out)


# ── 章节与分节符 ──────────────────────────────────────────────────────
def chapter_heading(text: str) -> str:
    """一级标题：小三黑体、居中、2 倍行距（line=480）。"""
    return para(
        run(text, bold=True, size=SZ_X3, font=FONT_HEI),
        style="1", jc="center",
        spacing='<w:spacing w:line="480" w:lineRule="auto" w:before="0" w:after="240"/>')


def sect_para(sectpr: str) -> str:
    """承载 sectPr 的空段落（标记该节结束）。"""
    return f'<w:p><w:pPr><w:pStyle w:val="a3"/>{sectpr}</w:pPr></w:p>'


def toc_field() -> str:
    """插入真正的 Word 目录域；配合 settings.xml 的 updateFields 自动更新。"""
    begin = ('<w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r>')
    instr = ('<w:r><w:instrText xml:space="preserve"> TOC \\o "1-2" \\h \\z \\u '
             '</w:instrText></w:r>')
    sep = '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
    body = run("（目录将在 Word 打开时自动更新；如未更新请按 Ctrl+A 后 F9）",
               size=SZ_S4, font=FONT_SONG)
    end = '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
    return para(begin + instr + sep + body + end, jc="left", spacing=SP_BODY)


# ── 页眉页脚 ──────────────────────────────────────────────────────────
# 格式规范：页眉、页码宋体小五号居中，页眉线为单横线；
# 目录页页眉「目录」、页脚罗马数字；正文奇数页页眉为章标题、
# 偶数页页眉为「河南理工大学课程设计报告」。
W_NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
EVEN_HEADER = "河南理工大学课程设计报告"

_HDR_RPR = ('<w:rPr><w:rFonts w:ascii="宋体" w:hAnsi="宋体" w:eastAsia="宋体" '
            'w:cs="宋体"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>')

_HDR_RULE = ('<w:pBdr><w:bottom w:val="single" w:sz="6" w:space="1" '
             'w:color="auto"/></w:pBdr>')


def header_part(text: str) -> str:
    """页眉部件：宋体小五、居中、单横线。"""
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<w:hdr {W_NS}><w:p><w:pPr><w:pStyle w:val="a5"/>{_HDR_RULE}'
            f'<w:jc w:val="center"/>{_HDR_RPR}</w:pPr>'
            f'<w:r>{_HDR_RPR}<w:t xml:space="preserve">{esc(text)}</w:t></w:r>'
            '</w:p></w:hdr>')


def header_part_pair(odd: str, even: str) -> str:
    """奇数页页眉章标题、偶数页页眉校名——用 IF 域按页号奇偶切换。

    不用 settings.xml 的 evenAndOddHeaders：Word 在「上一节结束于奇数页 +
    本节从下一页开始」时会额外插一张空白页（实测目录前多一页），
    而且目录页码会从 II 起算。IF 域放在同一个 default 页眉里，
    分页行为与普通页眉完全一致。
    """
    rpr = _HDR_RPR

    def fld(kind: str) -> str:
        return f'<w:r>{rpr}<w:fldChar w:fldCharType="{kind}"/></w:r>'

    def instr(text: str) -> str:
        return (f'<w:r>{rpr}<w:instrText xml:space="preserve">{text}'
                '</w:instrText></w:r>')

    def txt(text: str) -> str:
        return f'<w:r>{rpr}<w:t xml:space="preserve">{text}</w:t></w:r>'

    page = (fld("begin") + instr(" PAGE ") + fld("separate")
            + txt("1") + fld("end"))
    mod = (fld("begin") + instr(" =MOD(") + page + instr(",2) ")
           + fld("separate") + txt("1") + fld("end"))
    code = (fld("begin") + instr(" IF ") + mod
            + instr(f' = 0 "{esc(even)}" "{esc(odd)}" ')
            + fld("separate") + txt(odd) + fld("end"))
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<w:hdr {W_NS}><w:p><w:pPr><w:pStyle w:val="a5"/>{_HDR_RULE}'
            f'<w:jc w:val="center"/>{rpr}</w:pPr>{code}</w:p></w:hdr>')


def footer_part(page_number: bool) -> str:
    """页脚部件：宋体小五、居中；page_number=True 时插入 PAGE 域。"""
    if not page_number:
        return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                f'<w:ftr {W_NS}><w:p><w:pPr><w:pStyle w:val="a4"/>'
                f'<w:jc w:val="center"/></w:pPr></w:p></w:ftr>')
    field = ('<w:r>{r}<w:fldChar w:fldCharType="begin"/></w:r>'
             '<w:r>{r}<w:instrText xml:space="preserve"> PAGE  \\* MERGEFORMAT '
             '</w:instrText></w:r>'
             '<w:r>{r}<w:fldChar w:fldCharType="separate"/></w:r>'
             '<w:r>{r}<w:t>1</w:t></w:r>'
             '<w:r>{r}<w:fldChar w:fldCharType="end"/></w:r>').format(r=_HDR_RPR)
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<w:ftr {W_NS}><w:p><w:pPr><w:pStyle w:val="a4"/>'
            f'<w:jc w:val="center"/>{_HDR_RPR}</w:pPr>{field}</w:p></w:ftr>')


def sect_xml(sp: str, *, hdr: str = "", even_hdr: str = "", ftr: str = "",
             even_ftr: str = "", break_type: str = "") -> str:
    """重写一个 sectPr：去掉模板自带的页眉页脚引用，换成新的。

    CT_SectPr 的子元素顺序是固定的：headerReference* → footerReference* →
    type → pgSz → pgMar → … → pgNumType …。因此所有引用都插在 sectPr
    起始标签之后，分节类型紧随其后。
    """
    sp = re.sub(r"<w:(?:header|footer)Reference\b[^>]*/>", "", sp)
    refs = ""
    if hdr:
        refs += f'<w:headerReference w:type="default" r:id="{hdr}"/>'
    if even_hdr:
        refs += f'<w:headerReference w:type="even" r:id="{even_hdr}"/>'
    if ftr:
        refs += f'<w:footerReference w:type="default" r:id="{ftr}"/>'
    if even_ftr:
        refs += f'<w:footerReference w:type="even" r:id="{even_ftr}"/>'
    if break_type:
        refs += f'<w:type w:val="{break_type}"/>'
    return re.sub(r"(<w:sectPr\b[^>]*>)", lambda m: m.group(1) + refs, sp, count=1)


def swap_sect(xml: str, new_sect: str) -> str:
    """把一个块里的 sectPr 换掉（封面/评分表那两个块）。"""
    return re.sub(r"<w:sectPr\b.*?</w:sectPr>", lambda m: new_sect, xml,
                  count=1, flags=re.DOTALL)


def toc_styles() -> str:
    """目录正文样式：小四宋体、1.3 倍行距，数字/英文 Times New Roman。

    目录是域，条目由 Word 生成；不预置这两条样式，Word 会按 Normal（五号）
    临时造一份 TOC1/TOC2，字号就对不上格式规范了。
    """
    def one(sid: str, name: str, indent: int) -> str:
        ind = f'<w:ind w:left="{indent}"/>' if indent else ""
        return (f'<w:style w:type="paragraph" w:styleId="{sid}">'
                f'<w:name w:val="{name}"/><w:basedOn w:val="a"/>'
                '<w:next w:val="a"/><w:qFormat/><w:pPr>'
                '<w:tabs><w:tab w:val="right" w:leader="dot" w:pos="8306"/>'
                '</w:tabs>'
                '<w:spacing w:line="312" w:lineRule="auto"/>'
                f'{ind}<w:jc w:val="left"/></w:pPr>'
                '<w:rPr><w:rFonts w:ascii="Times New Roman" '
                'w:hAnsi="Times New Roman" w:eastAsia="宋体" '
                'w:cs="Times New Roman"/><w:sz w:val="24"/>'
                '<w:szCs w:val="24"/></w:rPr></w:style>')
    return (one("TOC1", "toc 1", 0)
            + one("TOC2", "toc 2", 480))


# ── 顶层块切分（深度感知，模板的评分表是嵌套表格）────────────────────
_TBL_OPEN = re.compile(r"<w:tbl(?=[ >])")


def _is_tag(xml: str, pos: int, name: str) -> bool:
    """`pos` 处是否恰好是 `<w:name>` 或 `<w:name ...>`。

    必须做这个判断：`<w:tbl` 也会匹配上 `<w:tblPr` / `<w:tblGrid` /
    `<w:tblBorders`，直接用前缀判断会把表属性当成表格起始，
    深度计数随之错乱、整篇文档被吞成一块。
    """
    pre = f"<w:{name}"
    if not xml.startswith(pre, pos):
        return False
    nxt = pos + len(pre)
    return nxt < len(xml) and xml[nxt] in " >/"


def split_blocks(xml: str) -> list[str]:
    """把 body 切成顶层 <w:p> / <w:tbl>。

    不能用非贪婪正则：模板里的评分表**是嵌套表格**（外层表里还有一张表），
    非贪婪匹配会在第一个 </w:tbl> 处截断，留下半张表。
    """
    blocks: list[str] = []
    i, n = 0, len(xml)
    while True:
        j = xml.find("<w:", i)
        if j < 0:
            break

        if _is_tag(xml, j, "tbl"):
            depth, k = 0, j
            while True:
                m = _TBL_OPEN.search(xml, k)
                c = xml.find("</w:tbl>", k)
                if c < 0:
                    k = n
                    break
                if m and m.start() < c:
                    depth += 1
                    k = m.end()
                else:
                    depth -= 1
                    k = c + 8
                    if depth == 0:
                        break
            blocks.append(xml[j:k])
            i = k

        elif _is_tag(xml, j, "p"):
            head_end = xml.find(">", j)
            if head_end > 0 and xml[head_end - 1] == "/":     # 自闭合
                blocks.append(xml[j:head_end + 1])
                i = head_end + 1
                continue
            c = xml.find("</w:p>", j)
            if c < 0:
                break
            blocks.append(xml[j:c + 6])
            i = c + 6

        else:
            i = j + 1
    return blocks


# ── 主流程 ────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只自检，不写文件")
    args = ap.parse_args()

    from content_part1 import PART1
    from content_part2 import PART2
    from content_part3 import PART3
    all_chapters = PART1 + PART2 + PART3

    if not TEMPLATE.exists():
        print(f"[错误] 找不到模板：{TEMPLATE}")
        return 1

    # 解压模板到系统临时目录（不放在项目里：Windows 偶尔会锁住解压目录，
    # 放在项目内会导致下次构建删不掉）
    import tempfile
    work = Path(tempfile.mkdtemp(prefix="senti-report-"))
    with zipfile.ZipFile(TEMPLATE) as zf:
        zf.extractall(work)

    doc_path = work / "word" / "document.xml"
    xml = doc_path.read_text(encoding="utf-8")

    # 切块（深度感知）
    blocks = split_blocks(xml)
    print(f"模板顶层块：{len(blocks)}")

    # 提取 sectPr（按出现顺序）
    sectprs = re.findall(r"<w:sectPr\b.*?</w:sectPr>", xml, re.DOTALL)
    print(f"分节符：{len(sectprs)} 个")

    # 正文起点与终点
    body_open = xml[: xml.index(blocks[0])]
    body_close = xml[xml.rindex(blocks[-1]) + len(blocks[-1]):]

    # ── 定位关键块（按内容而非硬编码下标，模板微调后也不会错位）──────
    i_first_sect = next(i for i, b in enumerate(blocks) if "<w:sectPr" in b)
    i_tbl = next(i for i, b in enumerate(blocks) if b.startswith("<w:tbl"))
    i_outline = next(i for i, b in enumerate(blocks) if "编写提纲" in b)
    i_sect1 = next(i for i, b in enumerate(blocks)
                   if i > i_outline and "<w:sectPr" in b)
    print(f"关键块：封面止={i_first_sect} 评分表={i_tbl} "
          f"提纲起={i_outline} 第二节止={i_sect1}")

    # ── 页眉页脚部件 ──────────────────────────────────────────────
    # 模板本身没有页眉部件，这里按格式规范新建：
    # 目录页「目录」、奇偶页页眉、页眉横线、宋体小五居中的页码。
    extra_parts: dict[str, tuple[str, str]] = {}   # 文件名 -> (XML, kind)
    rel_entries: list[tuple[str, str, str]] = []    # (rId, kind, 文件名)
    _seq = [0]

    def add_part(name: str, xml_part: str, kind: str) -> str:
        _seq[0] += 1
        rid = f"rIdX{_seq[0]}"
        extra_parts[name] = (xml_part, kind)
        rel_entries.append((rid, kind, name))
        return rid

    rid_toc_hdr = add_part("header_toc.xml", header_part("目录"), "header")
    rid_blank_ftr = add_part("footer_blank.xml", footer_part(False), "footer")
    rid_page_ftr = add_part("footer_page.xml", footer_part(True), "footer")
    chapter_hdr_rids = [
        add_part(f"header_ch{i}.xml",
                 header_part_pair(tight(ch["heading"]), EVEN_HEADER), "header")
        for i, ch in enumerate(all_chapters, 1)
    ]

    # 封面与评分表合并为一节（都不编页码、不加页眉）。
    # 封面末尾的 sectPr 会被去掉，整节由评分表末尾的 eval_sect 收尾。
    eval_sect = sect_xml(sectprs[1], ftr=rid_blank_ftr)

    # ── 封面 ────────────────────────────────────────────────────────
    # 按用户要求**不自动填写封面信息**：原样保留模板的下划线，
    # 打印前手写或自行填写。把 FILL_COVER 改成 True 就会用下面的 COVER 字典填。
    cover_blocks = list(blocks[: i_first_sect + 1])

    if FILL_COVER:
        keys = list(COVER)
        rows = [
            (16, keys[0], 21, 22),   # 题目：字号小一档，保证 21 字能排进一行
            (17, keys[1], 10, 28),
            (18, keys[2], 12, 28),
            (19, keys[3], 12, 28),
            (20, keys[4], 10, 28),
        ]
        for idx, label, width, size in rows:
            shown = label
            if label == keys[0] and len(label) == 2:      # 「题    目」
                shown = label[0] + "    " + label[1]
            value = COVER[label]
            pad = "　" * max(0, width - len(value))   # 全角空格补齐下划线
            # 标签后留一个全角空格，否则「学生姓名刘航宇」会粘在一起
            inner = (run(shown + "　", size=size, font=FONT_TNR)
                     + run(value + pad, size=size, font=FONT_TNR,
                           underline=True))
            cover_blocks[idx] = para(
                inner, jc="left",
                ind='<w:ind w:firstLineChars="700" w:firstLine="1960"/>')

        if COVER.get("日期"):
            cover_blocks[23] = para(
                run(COVER["日期"], size=SZ_HAO4, font=FONT_TNR),
                jc="center",
                extra_ppr='<w:rPr><w:rFonts w:ascii="Times New Roman" '
                          'w:hAnsi="Times New Roman"/><w:sz w:val="28"/>'
                          '<w:szCs w:val="28"/></w:rPr>')

    # 去掉封面末尾的 sectPr：让封面与评分表并入同一节（页眉页脚都为空）
    cover_blocks[-1] = swap_sect(cover_blocks[-1], "")
    # 打印要求：封皮背面必须为空白。在封面最后一页正文（日期段）末尾
    # 加一个显式分页符，评分表前的空段落再 pageBreakBefore 到第 3 页，
    # 中间第 2 页就是空白。
    cover_blocks[-2] = cover_blocks[-2].replace(
        "</w:p>", '<w:r><w:br w:type="page"/></w:r></w:p>', 1)

    # ── 评分表段：评分表 + 签名 + 该节的 sectPr(#1) ──────────────────
    # 模板中「打印时删除该行」「编写提纲」「格式规范」三部分一并去掉
    # 评分表在模板里是**浮动表格**（tblpPr + tblpYSpec=center）。模板靠后面
    # 一长串空段落给它当锚点；这些段落随「编写提纲/格式规范」一起被删掉后，
    # 浮动表会把唯一的 sectPr 段落顶到下一页，凭空多出一张空白页。
    # 去掉 tblpPr/tblOverlap 改成内联表，表格仍占满版心，且分页稳定。
    eval_tbl = re.sub(r"<w:tblpPr\b[^>]*/>", "", blocks[i_tbl])
    eval_tbl = re.sub(r"<w:tblOverlap\b[^>]*/>", "", eval_tbl)
    eval_end = swap_sect(blocks[i_sect1], eval_sect)
    # 该节末尾的空段落只承载 sectPr。把段落标记压到 1 pt，
    # 否则评分表刚好占满页面时，14 pt 的空行会溢出成一张空白页
    # （实测会在评分表与目录之间多出第 4 页）。
    eval_end = re.sub(r'<w:sz w:val="28"/>', '<w:sz w:val="2"/>',
                      eval_end, count=1)
    eval_end = re.sub(r'<w:szCs w:val="28"/>', '<w:szCs w:val="2"/>',
                      eval_end, count=1)
    eval_blocks = ([
        # 评分表前的空段落：pageBreakBefore 把它推到第 3 页，
        # 第 2 页（封皮背面）保持空白。放 1 pt 空格避免 Word 把
        # 「纯空段落」上的分页优化掉。
        '<w:p><w:pPr><w:pageBreakBefore/>'
        '<w:spacing w:line="240" w:lineRule="auto"/></w:pPr>'
        '<w:r><w:rPr><w:sz w:val="2"/><w:szCs w:val="2"/></w:rPr>'
        '<w:t xml:space="preserve"> </w:t></w:r></w:p>',
        eval_tbl,
    ] + blocks[i_tbl + 1 : i_outline] + [eval_end])

    # ── 目录节 ────────────────────────────────────────────────────
    # 页脚罗马数字、页眉「目录」。目录节本身用 nextPage 起始：
    # `w:type` 描述的是**当前节**如何起始，若在这里写 oddPage，
    # 评分表之后会先插一张空白页再把目录放到奇数页。
    # 第 1 章是否落在奇数页由第 1 章自己的 sectPr 保证。
    toc_base = re.sub(r'<w:pgNumType\b[^>]*/>',
                      '<w:pgNumType w:fmt="upperRoman" w:start="1"/>',
                      sectprs[2], count=1)
    toc_sect = sect_xml(toc_base, hdr=rid_toc_hdr, ftr=rid_page_ftr,
                        break_type="nextPage")
    toc_blocks = [
        para(run("目录", bold=True, size=SZ_X3, font=FONT_HEI),
             jc="center",
             spacing='<w:spacing w:line="480" w:lineRule="auto" w:before="0" w:after="240"/>'),
        toc_field(),
        sect_para(toc_sect),
    ]

    # ── 正文章节 ──────────────────────────────────────────────────
    r = Renderer()
    body_chapters = []
    sect_index = 3

    for i, ch in enumerate(all_chapters):
        heading = ch["heading"]
        is_ref = heading.startswith("参考")
        chap_no = "" if is_ref else heading.split()[0]

        parts = [chapter_heading(heading)]
        parts.append(r.render(ch["blocks"], chap_no))

        # 最后一节（参考文献）不追加 sectPr：文档末尾的 body 级 sectPr 本身就是
        # 它的分节符。再补一个会多出一个空节，末尾多出一页空白。
        if not is_ref:
            # oddPage 是「本节如何起始」：本章从奇数页开始
            sp = sect_xml(sectprs[sect_index],
                          hdr=chapter_hdr_rids[i], ftr=rid_page_ftr,
                          break_type="oddPage")
            parts.append(sect_para(sp))
            sect_index += 1
        body_chapters.append("".join(parts))

    # 文末（参考文献）那一节的 sectPr 在 body_close 里，同样挂上页眉页脚
    ref_sect = sect_xml(sectprs[-1], hdr=chapter_hdr_rids[-1],
                        ftr=rid_page_ftr)
    body_close = body_close.replace(sectprs[-1], ref_sect)

    # ── 拼装 ──────────────────────────────────────────────────────
    new_body = ("".join(tight_xml_texts(b) for b in cover_blocks)
                + "".join(eval_blocks)
                + "".join(toc_blocks) + "".join(body_chapters))
    new_xml = body_open + new_body + body_close

    if args.check:
        print("\n自检通过（未写文件）")
        for w in r.warnings:
            print("  [警告]", w)
        return 0

    doc_path.write_text(new_xml, encoding="utf-8")

    # ── 图片：写入 media、补关系与内容类型 ──────────────────────────
    if r.image_rels:
        rels_path = work / "word" / "_rels" / "document.xml.rels"
        rels = rels_path.read_text(encoding="utf-8")
        extra = []
        for rid, media, src in r.image_rels:
            shutil.copyfile(src, work / "word" / "media" / media)
            extra.append(
                f'<Relationship Id="{rid}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                f'relationships/image" Target="media/{media}"/>')
        rels = rels.replace("</Relationships>",
                            "".join(extra) + "</Relationships>")
        rels_path.write_text(rels, encoding="utf-8")

        ct_path = work / "[Content_Types].xml"
        ct = ct_path.read_text(encoding="utf-8")
        if 'Extension="png"' not in ct:
            ct = ct.replace("<Default ", '<Default Extension="png" '
                            'ContentType="image/png"/><Default ', 1)
        if 'Extension="svg"' not in ct:
            ct = ct.replace("<Default ", '<Default Extension="svg" '
                            'ContentType="image/svg+xml"/><Default ', 1)
        ct_path.write_text(ct, encoding="utf-8")

    # ── 超链接：补 document.xml.rels 的外部关系 ────────────────────
    if r.link_rels:
        rels_path = work / "word" / "_rels" / "document.xml.rels"
        rels = rels_path.read_text(encoding="utf-8")
        extra = [
            f'<Relationship Id="{rid}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/'
            f'relationships/hyperlink" Target="{esc(url)}" TargetMode="External"/>'
            for rid, url in r.link_rels
        ]
        rels = rels.replace("</Relationships>",
                            "".join(extra) + "</Relationships>")
        rels_path.write_text(rels, encoding="utf-8")

    # ── 页眉页脚：写部件、补关系与内容类型 ──────────────────────────
    if extra_parts:
        rels_path = work / "word" / "_rels" / "document.xml.rels"
        rels = rels_path.read_text(encoding="utf-8")
        extra = []
        for rid, kind, name in rel_entries:
            xml_part, _ = extra_parts[name]
            (work / "word" / name).write_text(xml_part, encoding="utf-8")
            extra.append(
                f'<Relationship Id="{rid}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                f'relationships/{kind}" Target="{name}"/>')
        rels = rels.replace("</Relationships>",
                            "".join(extra) + "</Relationships>")
        rels_path.write_text(rels, encoding="utf-8")

        ct_path = work / "[Content_Types].xml"
        ct = ct_path.read_text(encoding="utf-8")
        overrides = "".join(
            f'<Override PartName="/word/{name}" '
            'ContentType="application/vnd.openxmlformats-officedocument.'
            f'wordprocessingml.{kind}+xml"/>'
            for _, kind, name in rel_entries)
        ct = ct.replace("</Types>", overrides + "</Types>")
        ct_path.write_text(ct, encoding="utf-8")

    # ── settings：打开时更新域（让目录页码自动生成）──────────────────
    set_path = work / "word" / "settings.xml"
    st = set_path.read_text(encoding="utf-8")
    # 刻意不写 evenAndOddHeaders：它会让 Word 在特定分节位置多插空白页，
    # 并把目录页码从 II 起算。奇偶页页眉改由 header_part_pair() 的 IF 域实现。
    if "updateFields" not in st:
        # w:settings 的子元素顺序由 schema 严格约束：updateFields 必须紧挨在
        # hdrShapeDefaults 之前。插在开头会被 XSD 判为「元素位置不对」。
        if "<w:hdrShapeDefaults" in st:
            st = st.replace(
                "<w:hdrShapeDefaults",
                '<w:updateFields w:val="true"/><w:hdrShapeDefaults', 1)
        else:
            st = st.replace(
                "</w:settings>",
                '<w:updateFields w:val="true"/></w:settings>', 1)
        set_path.write_text(st, encoding="utf-8")

    # ── styles：关掉「中文与西文/数字之间的自动间距」──────────────────
    # Word 有 autoSpaceDE / autoSpaceDN 两个开关（默认开启），会在渲染时
    # 给中文与西文、数字之间**自动补一段空白**。这意味着文本里一个空格都没有，
    # 「如表2-1所示」在 PDF 里照样显示成「如表 2-1 所示」，pdftotext 也照抄出空格。
    # 光靠删字符解决不了，必须在文档默认段落属性里把这两个开关置 0。
    # 顺带一提，顺序受 schema 约束：autoSpaceDE/DN 必须排在 spacing、ind、jc 之前。
    sty_path = work / "word" / "styles.xml"
    sty = sty_path.read_text(encoding="utf-8")
    if "autoSpaceDE" not in sty:
        flags = '<w:autoSpaceDE w:val="0"/><w:autoSpaceDN w:val="0"/>'
        if "<w:pPrDefault/>" in sty:
            sty = sty.replace(
                "<w:pPrDefault/>",
                f"<w:pPrDefault><w:pPr>{flags}</w:pPr></w:pPrDefault>", 1)
        else:
            sty = re.sub(r"<w:pPrDefault><w:pPr>",
                         lambda m: m.group(0) + flags, sty, count=1)
        stiff = "" if "autoSpaceDE" in sty else "（模板结构不认识，未生效）"
        sty_path.write_text(sty, encoding="utf-8")
        print(f"已关闭中文-西文自动间距{stiff}")

    # 目录正文样式（Word 更新目录域时会套用这两条样式）
    if "TOC1" not in sty:
        sty = sty.replace("</w:styles>", toc_styles() + "</w:styles>", 1)
        sty_path.write_text(sty, encoding="utf-8")
        print("已写入目录样式 TOC1/TOC2（小四宋体、1.3 倍行距）")

    # ── 重新打包 ──────────────────────────────────────────────────
    if OUTPUT.exists():
        OUTPUT.unlink()
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(work.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(work).as_posix())

    print(f"\n已生成：{OUTPUT}  ({OUTPUT.stat().st_size:,} bytes)")
    print(f"图片关系 {len(r.image_rels)} 条（其中 SVG 主图 {r.svg_used} 张），"
          f"超链接 {len(r.link_rels)} 条，章节 {len(all_chapters)} 个")
    for w in r.warnings:
        print("  [警告]", w)

    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
