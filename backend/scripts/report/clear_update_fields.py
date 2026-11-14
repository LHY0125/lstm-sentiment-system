# -*- coding: utf-8 -*-
"""清掉 docx 的 updateFields 与 lastRenderedPageBreak。

`build_report.py` 会写入 `updateFields=true`，让第一次打开时把目录域算出来。
但 `render_pdf.ps1` 在 TOC.Update() 之后把目录字体压平成了小四宋体/TNR；
如果这个开关还留着，用户下次打开 Word 会再次更新 TOC，把标题的直接格式
（黑体、加粗）重新复制进目录条目，压平就白做了。

`lastRenderedPageBreak` 是 Word 保存时写入的「上次分页位置」提示。
再次打开时 Word 会把它当成一次分页，叠加在分节符上就会凭空多出空白页
（目录前那一页就是这么来的）。导出与保存后都清掉，保证下次打开干净。

用法：
    python clear_update_fields.py <docx>
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("用法: python clear_update_fields.py <docx>")
        return 2

    target = Path(sys.argv[1]).resolve()
    tmp = target.with_name(target.name + ".tmp")
    with zipfile.ZipFile(target) as zin, \
            zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/settings.xml":
                text = data.decode("utf-8")
                text = re.sub(r"<w:updateFields\b[^>]*/>", "", text)
                data = text.encode("utf-8")
            elif item.filename == "word/document.xml":
                text = data.decode("utf-8")
                text = text.replace("<w:lastRenderedPageBreak/>", "")
                data = text.encode("utf-8")
            zout.writestr(item, data)
    tmp.replace(target)
    print(f"已关闭 updateFields：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
