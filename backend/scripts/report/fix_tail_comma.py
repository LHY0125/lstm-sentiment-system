# -*- coding: utf-8 -*-
"""把 content_part2.py 第 390 行末尾的「完成的」改为「完成，」。

用 chr(0xFF0C) 构造全角逗号，脚本本身不含任何中文字面量，
从根本上排除"修脚本时把脚本改坏"的风险。幂等。
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

FULLWIDTH_COMMA = chr(0xFF0C)   # ，
TARGET_LINE = 390

path = Path(__file__).with_name("content_part2.py")
lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

idx = TARGET_LINE - 1
line = lines[idx]           # 以 '的"\n' 结尾

if line[-3] == FULLWIDTH_COMMA:
    print("already fixed, skip")
else:
    # 用码点比较，不依赖任何中文字面量
    old_cp = hex(ord(line[-3]))
    lines[idx] = line[:-3] + FULLWIDTH_COMMA + line[-2:]
    print(f"line {TARGET_LINE}: replaced U+{old_cp[2:].upper()} -> U+FF0C")
    path.write_text("".join(lines), encoding="utf-8")
    print("written")

# 复核：把该行末尾三个字符的码点打出来
new_line = path.read_text(encoding="utf-8").splitlines(keepends=True)[idx]
print("tail codepoints:", [hex(ord(c)) for c in new_line[-3:]])
