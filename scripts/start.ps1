$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "$root\vendor"
Set-Location $root
python run.py
