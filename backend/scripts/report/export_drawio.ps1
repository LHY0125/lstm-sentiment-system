# Export every .drawio under drawio/ to figs/<name>.png
#
# ASCII-only on purpose: this script must parse identically under
# Windows PowerShell 5.1 (which decodes BOM-less files as GBK) and pwsh 7.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File export_drawio.ps1
#   powershell -ExecutionPolicy Bypass -File export_drawio.ps1 -Name fig_2_1_roadmap
#
# Requires draw.io desktop (scoop install "draw.io"); the CLI flags below are
# stable across v20+ and let us crop to content, which matters because the
# diagrams are not laid out on a fixed page.

param(
    [string]$Name,
    [int]$Scale = 3,
    [int]$Border = 12
)

$ErrorActionPreference = 'Stop'

$here   = $PSScriptRoot
$src    = Join-Path $here 'drawio'
$out    = Join-Path $here 'figs'
$drawio = 'D:\settings\settings\Scoop\shims\draw.io.exe'

if (-not (Test-Path -LiteralPath $drawio)) {
    throw "draw.io not found at $drawio"
}
if (-not (Test-Path -LiteralPath $out)) {
    New-Item -ItemType Directory -Path $out | Out-Null
}

$files = Get-ChildItem -LiteralPath $src -Filter *.drawio
if ($Name) { $files = $files | Where-Object { $_.BaseName -eq $Name } }
if (-not $files) { throw "no .drawio matched under $src" }

foreach ($f in $files) {
    $png = Join-Path $out ($f.BaseName + '.png')
    Write-Host ("export {0} -> {1}.png" -f $f.Name, $f.BaseName)
    & $drawio --export --format png --scale $Scale --border $Border --crop `
        --output $png $f.FullName | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "draw.io exited $LASTEXITCODE on $($f.Name)" }
    if (-not (Test-Path -LiteralPath $png)) { throw "no output for $($f.Name)" }
    $size = (Get-Item -LiteralPath $png).Length
    Write-Host ("  ok {0:N0} bytes" -f $size)
}
