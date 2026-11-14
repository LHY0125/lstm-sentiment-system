# -*- coding: utf-8 -*-
"""极简语法高亮：把源码切成 [(文字, 颜色), ...]，供 Word 代码块使用。

配色取自用户给定的 Word 模板（VS 风格），实测抓到的对应关系：

    关键字/赋值号   006699  加粗
    类型/内置函数   FF1493
    字面量           808080   （None / True / False / 数字）
    字符串           0000FF
    注释/文档字符串  008200
    普通标识符/标点  000000
    空白             333333

不做语法分析，只做词法扫描：注释、字符串、数字、标识符按优先级匹配，
其余字符原样输出。够用，且不会像 ast 那样在代码片段不完整时抛错。
"""
from __future__ import annotations

import re

# ── 配色 ──────────────────────────────────────────────────────────────
C_KEYWORD = "006699"
C_TYPE = "FF1493"
C_LITERAL = "808080"
C_STRING = "0000FF"
C_COMMENT = "008200"
C_PLAIN = "000000"
C_WS = "333333"

# Python 关键字。赋值号与箭头在模板里与关键字同色，一并归入。
PY_KEYWORDS = frozenset("""
    False None True and as assert async await break class continue def del elif
    else except finally for from global if import in is lambda nonlocal not or
    pass raise return try while with yield
""".split())

# 模板里 "=" 与 "-" 走关键字色，故把这些符号也当关键字处理。
PY_SYMBOL_KEYWORDS = frozenset({"->", "="})

PY_BUILTINS = frozenset("""
    abs all any bool bytes callable chr dict dir divmod enumerate eval filter
    float format frozenset getattr hasattr hash id int isinstance issubclass
    iter len list map max min next object open ord pow print property range repr
    reversed round set setattr slice sorted staticmethod str sum super tuple
    type vars zip
""".split())

PY_LITERALS = frozenset({"None", "True", "False"})

TS_KEYWORDS = frozenset("""
    as async await break case catch class const continue default delete do else
    enum export extends finally for from function if implements import in
    instanceof interface let namespace new of private protected public readonly
    return satisfies static super switch throw try type typeof var void while
    yield declare abstract
""".split())

TS_SYMBOL_KEYWORDS = frozenset({"->", "=", "=>"})

TS_BUILTINS = frozenset("""
    string number boolean object any unknown never void symbol bigint Array
    Promise Record Partial Readonly Required Map Set Date Error JSON Object
    String Number Boolean console Math
""".split())

TS_LITERALS = frozenset({"null", "undefined", "true", "false"})


def _profile(lang: str) -> tuple[frozenset, frozenset, frozenset, frozenset]:
    if lang in ("ts", "typescript", "js", "javascript"):
        return (TS_KEYWORDS, TS_SYMBOL_KEYWORDS, TS_LITERALS, TS_BUILTINS)
    return (PY_KEYWORDS, PY_SYMBOL_KEYWORDS, PY_LITERALS, PY_BUILTINS)


def _scanner(lang: str) -> re.Pattern:
    ts = lang in ("ts", "typescript", "js", "javascript")
    block_comment = r"/\*[\s\S]*?\*/" if ts else r"(?!x)x"   # 非 TS 时永不匹配
    line_comment = r"//[^\n]*" if ts else r"\#[^\n]*"
    tpl = r"`(?:\\.|[^`\\])*`" if ts else r"(?!x)x"
    return re.compile(
        rf"""
          (?P<com>   {line_comment} | {block_comment} )
        | (?P<tri>   \"\"\"[\s\S]*?\"\"\" | '''[\s\S]*?''' )
        | (?P<str>   "(?:\\.|[^"\\\n])*"
                   | '(?:\\.|[^'\\\n])*'
                   | {tpl} )
        | (?P<num>   \b(?:0[xX][0-9a-fA-F_]+ | 0[bB][01_]+ | 0[oO][0-7_]+
                        | \d[\d_]*(?:\.[\d_]*)?(?:[eE][+-]?\d+)?) \b )
        | (?P<word>  [^\W\d]\w* )
        | (?P<op>    ->|=>|= )
        | (?P<ws>    [ \t\xa0]+ )
        | (?P<other> [\s\S] )
        """,
        re.VERBOSE,
    )


def scan(src: str, lang: str = "python") -> list[tuple[str, str, bool]]:
    """返回 [(文字, 颜色, 是否加粗), ...]，含换行符本身。"""
    kws, syms, lits, builtins = _profile(lang)
    rx = _scanner(lang)
    out: list[tuple[str, str, bool]] = []

    for m in rx.finditer(src):
        kind = m.lastgroup
        text = m.group()

        if kind == "com" or kind == "tri":
            out.append((text, C_COMMENT, False))
        elif kind == "str":
            out.append((text, C_STRING, False))
        elif kind == "num":
            out.append((text, C_LITERAL, False))
        elif kind == "ws":
            out.append((text, C_WS, False))
        elif kind == "op":
            out.append((text, C_KEYWORD, True))
        elif kind == "word":
            # 字面量先判：None/True/False 同时在关键字表里，
            # 模板把 None/False 涂成灰色而不是关键字蓝，故以字面量为准。
            if text in lits:
                out.append((text, C_LITERAL, False))
            elif text in kws or text in syms:
                out.append((text, C_KEYWORD, True))
            elif text in builtins:
                out.append((text, C_TYPE, False))
            else:
                out.append((text, C_PLAIN, False))
        else:
            out.append((text, C_PLAIN, False))

    return out


def highlight_block(src: str, lang: str = "python") -> list[list[tuple[str, str, bool]]]:
    """整段源码 → 逐行的 [(文字, 颜色, 加粗), ...]。

    必须整段一次扫描而不是逐行：三引号字符串会跨行，
    逐行扫会把第二行起的内容当成普通代码。
    """
    lines: list[list[tuple[str, str, bool]]] = [[]]
    for text, color, bold in scan(src, lang):
        if "\n" not in text:
            lines[-1].append((text, color, bold))
            continue
        for i, part in enumerate(text.split("\n")):
            if i:
                lines.append([])
            if part:
                lines[-1].append((part, color, bold))
    return lines
