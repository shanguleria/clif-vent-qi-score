# Pool member sites (stage 7c) and fit the outcome model on the stack (stage 8, pooled). Windows.
#
#   .\run_pooled.ps1 [-Members a,b,c] [-PoolOnly]
#
# -Members a,b,c  write sites\pooled.json with these members (order sets Site A, B, ...);
#                 without it, the existing sites\pooled.json is used
# -PoolOnly       stop after stage 7c
#
# Every member must already have run stages 0-7 (.\run_scores.ps1 -Site <member>).
param(
  [string[]]$Members = @(),
  [switch]$PoolOnly
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Py = ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
  Write-Error "No virtualenv at .venv. Create it with:`n  py -3.12 -m venv .venv; .venv\Scripts\python.exe -m pip install -r requirements.txt"
}

$PooledProfile = "sites\pooled.json"
# Accept both -Members a,b,c (an array) and -Members "a,b,c" (one string).
$list = @($Members | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($list.Count -gt 0) {
  if ($list.Count -lt 2) { Write-Error "-Members needs at least two sites" }
  foreach ($m in $list) { if ($m -notmatch '^[A-Za-z0-9_-]+$') { Write-Error "bad site name in -Members: '$m'" } }
  @{ members = $list } | ConvertTo-Json | Set-Content -Encoding ascii $PooledProfile
  Write-Host "wrote $PooledProfile`: $($list -join ', ') (Site A = $($list[0]))"
} elseif (-not (Test-Path $PooledProfile)) {
  Write-Error "No $PooledProfile. Pass the members, e.g.: .\run_pooled.ps1 -Members site_one,site_two,site_three"
}

if (-not $PoolOnly) {
  if (-not (Get-Command Rscript -ErrorAction SilentlyContinue)) {
    Write-Error "Rscript not found. Install R (the version renv.lock pins), then: Rscript -e `"renv::restore(prompt = FALSE)`""
  }
  # Checked before the pooling step so a missing package fails in seconds, not after it.
  & Rscript -e "suppressPackageStartupMessages({library(arrow); library(lme4); library(ggplot2)})" *> $null
  if ($LASTEXITCODE -ne 0) {
    Write-Error "R packages are not installed for this project. Run once (10-30 min):`n  Rscript -e `"renv::restore(prompt = FALSE)`""
  }
}

$sw = [Diagnostics.Stopwatch]::StartNew()
function Format-Hms([int]$d) {
  return "{0}h{1:d2}m{2:d2}s" -f [int][math]::Floor($d/3600), [int][math]::Floor(($d%3600)/60), ($d%60)
}

Write-Host ""; Write-Host "========== 07c_pool_sites =========="
& $Py code\07c_pool_sites.py
if ($LASTEXITCODE -ne 0) {
  throw "07c refused or failed; nothing was written. Its log: .\show_results.ps1 -Site pooled"
}

if ($PoolOnly) {
  Write-Host ""; Write-Host "=== pooled in $(Format-Hms $sw.Elapsed.TotalSeconds) - log: .\show_results.ps1 -Site pooled ==="
  exit 0
}

Write-Host ""; Write-Host "========== 08_outcome_model (pooled) =========="
$env:CLIF_SITE = "pooled"
& Rscript code\08_outcome_model.R
if ($LASTEXITCODE -ne 0) { throw "stage 8 failed with exit code $LASTEXITCODE" }

Write-Host ""
Write-Host "=== done in $(Format-Hms $sw.Elapsed.TotalSeconds) - results: output\pooled\final_no_phi\08_outcome_model\ ==="
Write-Host "    summaries to share: .\show_results.ps1 -Site pooled"
