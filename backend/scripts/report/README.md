# 课程设计报告生成工具

把 `docs/课程设计报告模板 - 神经网络与深度学习.docx` 的副本填成完整报告，
产物是 `docs/《神经网络与深度学习》课程设计报告.docx` 与同名 PDF。

文字内容全部在 `content_part*.py` 里，**改文字不需要碰生成器**。

---

## 文件说明

| 文件 | 作用 |
|------|------|
| `content_part1.py` | 第 1–2 章（选题背景、技术方案） |
| `content_part2.py` | 第 3 章（详细设计） |
| `content_part3.py` | 第 4–6 章与参考文献（编码实现、结果分析、总结体会） |
| `build_report.py` | 生成器：读模板 → 填内容 → 输出 docx |
| `pyhighlight.py` | 代码清单的词法着色（配色照 `模板.docx`） |
| `drawio/` | **示意图的权威源**（6 个 `.drawio`，可用 draw.io 直接编辑） |
| `export_drawio.ps1` | 用 draw.io 命令行把 `drawio/*.drawio` 导出成 `figs/*.png` |
| `make_figs.py` | matplotlib 版旧图，**已退役**；有 `drawio/` 时拒绝运行，除非加 `--force` |
| `crop_shots.py` | 裁掉界面截图四周空白 |
| `render_pdf.ps1` | 用本机 Word 更新目录页码并导出 PDF |
| `fix_corruption.py` / `fix_tail_comma.py` | 一次性修复脚本，历史用途，可保留备查 |
| `clear_update_fields.py` | 导出后关闭 `updateFields`，防止 Word 重新生成目录时覆盖目录字体 |
| `figs/` | 报告插图 |

内容块支持的类型见 `content_part1.py` 顶部注释。

---

## 用法

```powershell
cd backend\scripts\report

# 1. 导出示意图（改了 .drawio 才需要跑；需要本机装 draw.io）
pwsh -ExecutionPolicy Bypass -File export_drawio.ps1
pwsh -ExecutionPolicy Bypass -File export_drawio.ps1 -Name fig_2_1_roadmap   # 只导一张

# 2. 只自检，不写文件（检查图片是否齐、语法是否通）
..\..\..\.venv\Scripts\python.exe build_report.py --check

# 3. 生成 docx
..\..\..\.venv\Scripts\python.exe build_report.py

# 4. 更新目录页码并导出 PDF（需要本机装 Microsoft Word）
pwsh -ExecutionPolicy Bypass -File render_pdf.ps1
```

生成后可做一次 XSD 校验（可选）：

```powershell
# 注意必须带 PYTHONUTF8=1：校验脚本在中文 Windows 上会按 GBK 读 UTF-8 文件，
# 不加这个环境变量会报一堆假的「codec can't decode byte」错误
$env:PYTHONUTF8=1
python "C:\Users\33644\.claude\skills\docx\scripts\office\validate.py" `
  "docs\《神经网络与深度学习》课程设计报告.docx" `
  --original "docs\课程设计报告模板 - 神经网络与深度学习.docx"
```

界面截图（图 4-1 ~ 4-3）是抓线上站点得到的，不随构建自动生成；
换机器时请自行重新截图，或删掉 `content_part3.py` 里对应的 `figure` 块。

---

## 几个关键设计

**正文、样式、页眉页脚分别处理。** 模板的封面、评分表、样式表、分节符原样保留；
正文块只替换 `word/document.xml`，页眉页脚则新建独立部件（模板原本没有页眉），
再通过 `document.xml.rels` / `[Content_Types].xml` 挂接。
奇偶页页眉不靠 `evenAndOddHeaders`（它会多插空白页、把目录页码推到 II），
而是每章一个页眉部件，里面用 `IF { =MOD({ PAGE },2) } = 0` 域切换校名与章标题。

**分节符必须保留。** 模板用 10 个 `sectPr` 把文档切成封面 / 评分表 / 目录 /
6 章 / 参考文献。报告把封面与评分表合并成一节（共 9 节），
好让「封皮背面空白」只需页内分页，不触发额外的奇数页空白。

**评分表是嵌套表格。** 切块用的是深度感知的 `split_blocks()`，
不能用非贪婪正则——那会在第一个 `</w:tbl>` 处截断，留下半张表。

**目录用域而非静态文字。** `toc_field()` 插入真正的 `TOC` 域，
配合 `settings.xml` 里的 `updateFields`，页码由 Word 计算。
所以**必须跑 `render_pdf.ps1`**，否则目录里没有页码。
导出完成后 `clear_update_fields.py` 会把 `updateFields` 清掉，
防止下次打开 Word 重新生成目录、覆盖掉压平后的目录字体。

目录收两级（`\o "1-2"`）。章标题用模板自带的 `styleId="1"`
（`heading 1`，`outlineLvl=0`），而**模板里没有 heading 2 样式**，
所以节标题（`h2` 块）是靠 `para(outline=1)` 直接写 `w:outlineLvl` 进目录的。
想再收到三级，把域改成 `\o "1-3"`、给 `h3` 块加 `outline=2` 即可。

**页码。** 每节都显式挂同一份居中页码页脚（封面与评分表挂空白页脚）；
正文第 1 章的分节写 `pgNumType start="1"`，后续节继续编号，
目录节用 `upperRoman`。`render_pdf.ps1` 负责逐 story 刷新域并导出两遍。

**封面默认不填。** `build_report.py` 顶部有 `FILL_COVER = False`，
保留模板的空下划线。改成 `True` 就会用 `COVER` 字典填。

**插图的显示高度按真实宽高算。** `wordprocessingDrawing` 的 `wp:extent`
是**显示尺寸**，Word 不会替你纠正比例；早先写死 `cy = cx * 0.62`
会把扁的图纵向拉伸、把高的图压扁。现在 `_png_size()` 直接读 PNG 的 IHDR
拿真实宽高（不依赖 Pillow），原始比例是多少就按多少显示。

**表格是三线表。** 表格级 `tblBorders` 只留上下两条 1.5 pt 的线，
左右、`insideH`、`insideV` 全部 `nil`；栏目线（表头下那条 0.75 pt）
挂在表头单元格的 `tcBorders` 上；表头不加底纹。注意 `CT_TcPr` 的顺序仍然是
`tcW → tcBorders → shd → vAlign`，写错位置 XSD 会报错。

**中文与英文/数字之间没有空格，这需要改两处。**
① 文本层面：`tight()` 在 `run()` 里统一清洗 CJK↔ASCII 边界，
代码块走 `raw=True` 跳过（否则和仓库源码对不上）。
② **渲染层面**：Word 的 `autoSpaceDE` / `autoSpaceDN`（「自动调整中文与西文/数字
之间的间距」）默认是开的，会在排版时**自动补一段空白**——哪怕 XML 里一个空格都没有，
`如表2-1所示` 在 PDF 里照样显示成 `如表 2-1 所示`。所以还要在 `styles.xml` 的
`pPrDefault` 里把这两个开关置 0，否则改文本是白改。

**封面没有做间距清洗。** 模板封面把日期、学年拆成了多个 run
（`2026年` 和 ` 1月` 不在同一个 `w:t` 里），正则跨不过 run 边界；
逐段落重排又会破坏下划线与居中。要改的话需要按段落合并文本再回填。

---

## 格式规范落实情况

模板里写明的格式要求，逐条对应到实现：

| 规范 | 实现位置 |
|------|----------|
| 正文小四宋体、1.3 倍行距，数字/英文 TNR | `SP_BODY`；`run(font=宋体)` 把 ascii 映射到 Times New Roman |
| 目录标题小三黑体 2 倍；目录正文小四宋体 1.3 倍 | 目录标题段落 + `toc_styles()` 的 TOC1/TOC2 |
| 一级标题小三黑体 2 倍；二级四号宋体加粗 1.5 倍；三级小四宋体 1.3 倍 | `chapter_heading()`、`SP_H2`/`SP_H3` |
| 图/表/注/参考文献五号宋体、单倍行距 | `SP_SINGLE`；题注统一 `SZ_HAO5` |
| 图名在图下、表名在表上、序号在左 | `Renderer.render()` 的 figure / table 分支 |
| 公式居中、编号在右行末 | `FORMULA_TABS` 的居中制表位 + 右对齐制表位 |
| 页眉页码宋体小五居中、页眉单横线 | `header_part()` / `footer_part()` |
| 目录页眉「目录」、页脚罗马数字 | `header_toc.xml` + 目录节 `upperRoman` |
| 每章从奇数页开始 | 各章 `sectPr` 写 `<w:type w:val="oddPage"/>` |
| 奇偶页页眉不同 | `header_part_pair()` 用 `IF` 域按 `PAGE` 奇偶切换，不用 `evenAndOddHeaders` |

目录条目由 Word 生成时会复制标题的直接格式（一级标题的黑体、二级标题的加粗），
所以 `render_pdf.ps1` 在 `TOC.Update()` 之后把整段目录压回小四宋体/TNR 并取消加粗。
只改样式表不够，必须在这一步覆盖直接格式。

---

## 代码清单的版式（照 `D:\下载\OneDrive\Desktop\办公\模板.docx`）

版式以用户给出的模板为准，逐项对齐：

| 项目 | 取值 |
|------|------|
| 字体 | `Consolas` + 东亚 `宋体`（`w:cs` 也是宋体） |
| 字号 | `w:sz=21`，即 **10.5 pt**（五号） |
| 底纹 | 白色（`fill=FFFFFF`，即无灰底） |
| 段后 | 0 |
| 行距 | `line=300 lineRule=atLeast` |
| 对齐 | 基线（`w:textAlignment=baseline`） |
| 边距 | 无边框 |

配色同样照模板抓下来的（VS 风格）：关键字与赋值号 `006699` 加粗、类型与内置函数
`FF1493`、`None/True/False/数字` `808080`、字符串 `0000FF`、注释与文档字符串
`008200`、普通标识符 `000000`、空白 `333333`。实现在 `pyhighlight.py`。

**清单内容从仓库直接读**，不手抄：

```python
{"type": "code", "caption": "下载、校验与原子改名（senti/data.py）",
 "file": "backend/senti/data.py", "lines": (111, 149)},
```

`file` 相对项目根，`lines` 是 1 起的闭区间；省略 `lines` 表示整份文件。
语言按扩展名推断（`.ts` → TypeScript），也可以显式给 `"lang"`。
题注不必写「代码 4-N」，编号由 `Renderer.code()` 按出现顺序自动生成——
这样在中间插一段清单，后面全部自动重排，不会出现「代码 4-8」后面跟着「代码 4-10」。

三个容易踩的点：

1. **代码段必须显式 `w:jc="left"`**。文档默认是两端对齐，代码行一旦折行，
   上一行会被拉成词间留大片空白的样子。
2. **折行靠「按行缩进的悬挂缩进」兜住**：每行按自身的前导空格数设置
   `w:ind left=N hanging=N`（`N = 空格数 × 115.5` 缇），首行回到 0、续行落在该行的
   缩进位置上。否则续行会顶平到页面左边，和上一条语句分不出来。
3. **不要用不换行空格（U+00A0）做缩进**。模板里是 U+00A0（Word 粘贴时的自动转换），
   但那样从文档里复制出去的代码会变成非法 Python 缩进。这里用普通空格 +
   `xml:space="preserve"`，渲染效果相同而复制可用。

---

## 示意图为什么改用 draw.io

第 2、3 章的 6 张图早先是用 matplotlib 画图元拼的。两个问题促成了迁移：

1. **字号折算下来太小。** 图的最终显示宽度由 `content_part*.py` 里的
   `width_cm` 决定，字体的实际磅值 = 字号 / 画布宽度 × `width_cm`。
   比如图 3-1 原来 7 步横排、`width_cm=14`，折算只有约 5 pt，打印基本看不清。
   改成 draw.io 后按目标磅值反推字号，现在正文标签在 8–10 pt。
2. **布局维护成本。** 手工摆坐标改一处要重算一片；`.drawio` 可以直接在
   draw.io 里拖，版式由程序负责导出。

几个使用上的约定：

- `drawio/page="0"` 关掉固定页面，导出时加 `--crop`，图的大小随内容走。
- 字体统一 `fontFamily=Microsoft YaHei`，与报告里其他图一致。
- 标签里的 HTML 是**转义**的（`&lt;br&gt;` 而非 `<br>`）——XML 属性值里
  不允许裸 `<`，draw.io 自己保存时也是这个形式。
- 导出命令在 `export_drawio.ps1` 里，**该脚本刻意只写 ASCII**：
  Windows PowerShell 5.1 会把无 BOM 的 UTF-8 当 GBK 读，中文注释能把整段脚本
  解析失败（就是用 pwsh 7 跑也是这个道理，干脆不依赖编码）。

---

## 已知偏差

- `oddPage` 分节会在「上一章正好结束在奇数页」时插入空白页（本报告实际有 3 页）。
  这是 Word 对「每章从奇数页开始」的固有行为：空白页不带页眉页脚，Word 也不为它
  生成页码，因此正文里会跳过那个偶数页码。
