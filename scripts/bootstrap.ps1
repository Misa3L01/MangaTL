# MangaTL - arranque inicial: uv + Python 3.12 + dependencias, todo dentro de R:.
# Uso (desde la raiz del proyecto):   .\scripts\bootstrap.ps1
# Es idempotente: si algo ya esta instalado, lo reutiliza.

param(
    [string]$UvVersion = '0.12.19',
    [string]$PythonVersion = '3.12',
    [switch]$SkipSync
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $root 'env.ps1')

$uvExe = Join-Path $root '.local\bin\uv.exe'

# 1. uv (binario portable, sin instalador: no toca el PATH permanente)
if (-not (Test-Path $uvExe)) {
    $zip = Join-Path $root "tmp\uv-$UvVersion.zip"
    $url = "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
    Write-Host "Descargando uv $UvVersion (~17 MB)..."
    curl.exe -L --fail --silent --show-error -o $zip $url
    if ($LASTEXITCODE -ne 0) { throw "No se pudo descargar uv desde $url" }
    Expand-Archive -Path $zip -DestinationPath (Join-Path $root '.local\bin') -Force
    Remove-Item $zip
}
Write-Host ("uv: " + (& $uvExe --version))

# 2. Python gestionado por uv, instalado en .local\python
& $uvExe python install $PythonVersion
if ($LASTEXITCODE -ne 0) { throw "Fallo la instalacion de Python $PythonVersion" }

# 3. Entorno virtual .venv y dependencias (PyTorch con CUDA desde el indice cu130)
if (-not $SkipSync) {
    & $uvExe sync
    if ($LASTEXITCODE -ne 0) { throw "Fallo 'uv sync'" }
}

Write-Host "Listo. Siguiente paso:  uv run mangatl setup" -ForegroundColor Green
