# MangaTL - compila la interfaz web del editor (web/ -> web/dist).
# Uso (desde la raiz del proyecto):   .\scripts\build-web.ps1
# Instala Node.js portable en .local\node si falta (sin instalador ni cambios globales) y deja
# la cache de npm en .local\npm-cache. Solo hace falta volver a ejecutarlo si cambia web/.

param([string]$NodeVersion = 'v24.21.0')

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $root 'env.ps1')

$nodeDir = Join-Path $root '.local\node'
$nodeExe = Join-Path $nodeDir 'node.exe'
if (-not (Test-Path $nodeExe)) {
    $name = "node-$NodeVersion-win-x64"
    $zip = Join-Path $root "tmp\$name.zip"
    Write-Host "Descargando Node.js $NodeVersion portable (~36 MB)..."
    curl.exe -L --fail --silent --show-error -o $zip "https://nodejs.org/dist/$NodeVersion/$name.zip"
    if ($LASTEXITCODE -ne 0) { throw "No se pudo descargar Node.js" }
    Expand-Archive -Path $zip -DestinationPath (Join-Path $root 'tmp') -Force
    if (Test-Path $nodeDir) { Remove-Item -Recurse -Force $nodeDir }
    Move-Item (Join-Path $root "tmp\$name") $nodeDir
    Remove-Item $zip
}
$env:PATH = "$nodeDir;$env:PATH"
$env:npm_config_cache = Join-Path $root '.local\npm-cache'
$env:npm_config_update_notifier = 'false'
$env:npm_config_fund = 'false'
Write-Host ("Node " + (& $nodeExe --version))

Push-Location (Join-Path $root 'web')
# npm/vite write notices to stderr: judge success by the exit code only.
$ErrorActionPreference = 'Continue'
try {
    if (Test-Path 'package-lock.json') { npm ci --no-audit } else { npm install --no-audit }
    if ($LASTEXITCODE -ne 0) { throw "Fallo la instalacion de dependencias de npm" }
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "Fallo la compilacion de la interfaz" }
} finally {
    Pop-Location
}
Write-Host "Listo. Abre el editor con:  uv run mangatl ui" -ForegroundColor Green
