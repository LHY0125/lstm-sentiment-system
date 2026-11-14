# 用本机 Microsoft Word 打开报告，更新所有域（含目录与页脚页码）并导出 PDF。
#
# 为什么不用 LibreOffice：本机没装，且 docx 技能自带的 soffice.py 依赖
# AF_UNIX，在 Windows 上直接报错。
#
# 为什么按「体积最大的 docx」选文件：docs 目录下只有模板与报告两份 .docx，
# 报告含 10 张图约 1 MB、模板约 86 KB。这样脚本里不必出现中文字面量，
# 避免脚本本身被编码问题破坏。
#
# 页码为什么容易错：模板的页脚里 PAGE 域带着**缓存值**（字面「1」）。
# 只有把每个 story（正文 / 页眉 / 页脚）里的域都真正更新一遍，
# 页码才会正确；只更新主文档或用 $f.Exists 过滤都会漏掉继承来的页脚。

$ErrorActionPreference = 'Stop'

$docs = Join-Path $PSScriptRoot '..\..\..\docs'
$docs = (Resolve-Path -LiteralPath $docs).Path

$docx = Get-ChildItem -LiteralPath $docs -Filter *.docx |
        Sort-Object Length -Descending | Select-Object -First 1
if (-not $docx) { throw 'docs 目录下找不到 .docx' }
"目标文档：$($docx.FullName)  ($([math]::Round($docx.Length/1MB,2)) MB)"

$pdf = [System.IO.Path]::ChangeExtension($docx.FullName, '.pdf')

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$doc = $null

function Update-AllFields($document) {
    # 1) 正文 + 所有页眉页脚 story（NextStoryRange 链式遍历，才有覆盖全部）
    $story = $document.StoryRanges.Item(1)
    while ($null -ne $story) {
        $story.Fields.Update() | Out-Null
        $story = $story.NextStoryRange
    }
    # 2) 逐节再兜一遍（链接到上一节的页脚也要刷）
    foreach ($sec in $document.Sections) {
        for ($i = 1; $i -le 3; $i++) {
            try { $sec.Footers.Item($i).Range.Fields.Update() | Out-Null } catch {}
            try { $sec.Headers.Item($i).Range.Fields.Update() | Out-Null } catch {}
        }
    }
    # 3) 目录单独刷新（TOC 是域，但需要 Update() 才会重排页码）
    #    更新后 Word 会把标题的直接格式带进条目（一级标题的黑体、二级标题的加粗），
    #    而格式规范要求目录正文统一为小四宋体 + Times New Roman 的 1.3 倍行距，
    #    所以刷新完再把整段目录的字体压平。取消加粗但保留字号与行距。
    foreach ($toc in $document.TablesOfContents) {
        $toc.Update() | Out-Null
        $range = $toc.Range
        $range.Font.Name = 'Times New Roman'
        $range.Font.NameFarEast = '宋体'
        $range.Font.Size = 12
        $range.Font.Bold = 0
    }
}

$py = Join-Path $PSScriptRoot '..\..\..\.venv\Scripts\python.exe'
$cleaner = Join-Path $PSScriptRoot 'clear_update_fields.py'
if ((Test-Path -LiteralPath $py) -and (Test-Path -LiteralPath $cleaner)) {
    # Word 上次保存写入的 lastRenderedPageBreak 会被当成一次分页，
    # 叠加在分节符上就是那张多出来的空白页；打开前先清掉。
    & $py $cleaner $docx.FullName
}

try {
    $doc = $word.Documents.Open($docx.FullName, $false, $false)

    # 章节 2 及其后统一「续前节」编号（StartingPageNumber = 0 表示继续）。
    # 不设置的话，Word 会按每节自己的起始页重排，页码出现 1、1、2 这种跳变。
    # 第 1 节=封面+评分表 2=目录 3=第1章 4=第2章，故从第 4 节开始处理。
    for ($i = 4; $i -le $doc.Sections.Count; $i++) {
        try { $doc.Sections.Item($i).PageSetup.StartingPageNumber = 0 } catch {}
    }

    # 目录节：罗马数字且从 I 开始。XML 里的 pgNumType 在部分 Word 版本上
    # 会从 II 起算（偶数页起始时节号被多加 1），这里用 COM 再钉一次。
    try {
        $doc.Sections.Item(2).PageSetup.StartingPageNumber = 1
        $doc.Sections.Item(2).PageSetup.PageNumberStyle = 1   # 大写罗马数字
    } catch {}

    Update-AllFields $doc
    $doc.Repaginate()

    $pages = $doc.ComputeStatistics(2)   # wdStatisticPages
    "页数：$pages   节数：$($doc.Sections.Count)"

    # 页脚是我们自己新建的 PAGE 域，Update-AllFields 已经逐 story 刷过，
    # 不需要模板时代那套「导出两次预热」；反而不预热才能避免
    # lastRenderedPageBreak 在第二次排版时被当成真实分页。
    $doc.ExportAsFixedFormat($pdf, 17)
    $doc.Save()

    "PDF：$pdf"
}
finally {
    if ($doc) { $doc.Close($false) }
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    [GC]::Collect()
}

# Word 保存时会把 updateFields 与 lastRenderedPageBreak 写回文档。
# 前者会让下次打开重新生成目录、覆盖压平的字体；后者会制造空白页。
# 必须放在 finally 之后：Word 没退出前文件还被占用。
if ((Test-Path -LiteralPath $py) -and (Test-Path -LiteralPath $cleaner)) {
    & $py $cleaner $docx.FullName
}
