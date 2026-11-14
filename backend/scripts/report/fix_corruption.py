# -*- coding: utf-8 -*-
"""一次性修复 content_part2.py 中的 U+FFFD 损坏字符。

按行号定位，把该行第一个连续的 U+FFFD 序列替换成正确汉字（用转义写，
避免脚本自身的中文字面量再被同样的编码问题破坏）。
幂等：已无 U+FFFD 的行会被跳过。
"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# (行号, 正确字符的 unicode 转义)
FIXES = [
    (13,  "语"),   # 语
    (23,  "统"),   # 统
    (98,  "集"),   # 集
    (128, "计"),   # 计
    (145, "相"),   # 相
    (289, "极"),   # 极
    (390, "的"),   # 的
]

path = Path(__file__).with_name("content_part2.py")
lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

changed = 0
for lineno, repl in FIXES:
    idx = lineno - 1
    if idx >= len(lines):
        print(f"line {lineno}: OUT OF RANGE")
        continue
    if "�" not in lines[idx]:
        print(f"line {lineno}: already clean, skip")
        continue
    new = re.sub("�+", repl, lines[idx], count=1)
    if new != lines[idx]:
        lines[idx] = new
        changed += 1
        print(f"line {lineno}: fixed")

path.write_text("".join(lines), encoding="utf-8")
print(f"\nwritten: {path} ({changed} line(s) changed)")

remaining = [i + 1 for i, l in enumerate(lines) if "�" in l]
print(f"remaining U+FFFD lines: {remaining}")
