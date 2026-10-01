# MangaTL - variables de entorno de la sesion.
# Uso (desde la raiz del proyecto):   . .\env.ps1
# Solo afecta a la sesion actual de PowerShell: no cambia la configuracion global de Windows.
# Objetivo: que todo lo que se instale, descargue o cachee quede en R:, nunca en C:.

$MangaTLRoot = $PSScriptRoot
$local = Join-Path $MangaTLRoot '.local'
$models = Join-Path $MangaTLRoot 'models'

$env:MANGATL_ROOT = $MangaTLRoot

# uv: binario, Python gestionado, cache y herramientas
$env:UV_INSTALL_DIR = "$local\bin"
$env:UV_NO_MODIFY_PATH = '1'
$env:UV_CACHE_DIR = "$local\uv-cache"
$env:UV_PYTHON_INSTALL_DIR = "$local\python"
$env:UV_PYTHON_BIN_DIR = "$local\python-bin"
$env:UV_PYTHON_INSTALL_BIN = '0'        # sin accesos directos en ~\.local\bin
$env:UV_PYTHON_INSTALL_REGISTRY = '0'   # sin entradas en el registro de Windows
$env:UV_MANAGED_PYTHON = '1'            # ignorar los Python instalados en C:
$env:UV_TOOL_DIR = "$local\uv-tools"
$env:UV_TOOL_BIN_DIR = "$local\uv-tools\bin"

# pip (solo por si alguna herramienta lo invoca)
$env:PIP_CACHE_DIR = "$local\pip-cache"

# Pesos de modelos y caches de ML
$env:HF_HOME = "$models\hf"
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TORCH_HOME = "$models\torch"
$env:XDG_CACHE_HOME = "$local\xdg-cache"
$env:CUDA_CACHE_PATH = "$local\nv-compute-cache"   # cache JIT de NVIDIA (por defecto en %APPDATA%)
$env:CUDA_VISIBLE_DEVICES = '0'                    # solo la RTX 4050

# Ollama portable (lo arranca mangatl automaticamente cuando hace falta)
$env:OLLAMA_MODELS = "$models\ollama"
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_FLASH_ATTENTION = '1'
$env:OLLAMA_KV_CACHE_TYPE = 'q8_0'
$env:OLLAMA_CONTEXT_LENGTH = '16384'   # mismo valor que translator.ollama.num_ctx

# Archivos temporales
$env:TEMP = "$MangaTLRoot\tmp"
$env:TMP = "$MangaTLRoot\tmp"

# npm (Fase 3)
$env:npm_config_cache = "$local\npm-cache"

# GitHub CLI portable: configuracion en .local (el token queda en el Administrador de credenciales)
$env:GH_CONFIG_DIR = "$local\gh\config"
$env:GH_NO_UPDATE_NOTIFIER = '1'
$env:GH_TELEMETRY = 'false'

# PATH de la sesion
$env:PATH = "$local\bin;$local\ollama;$local\gh\bin;$env:PATH"

foreach ($d in "$local\bin", "$local\uv-cache", "$local\python", "$models\hf", "$models\torch", "$models\ollama", "$MangaTLRoot\tmp", "$MangaTLRoot\logs") {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

Write-Host "MangaTL: entorno cargado (raiz: $MangaTLRoot)" -ForegroundColor Green
