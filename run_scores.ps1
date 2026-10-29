# Build the ventilator QI score panel for one site (Windows).
#
#   .\run_scores.ps1 [-Site <id>] [-Rebuild] [-RequireStage] [-SkipVerify]
#
# -Rebuild       ignore any stage cache and rebuild the cohort from raw CLIF (slow: re-runs
#                encounter stitching and the clifpy respiratory-support waterfall)
# -RequireStage  fail rather than fall back to a cold build (for CI)
param(
  [string]$Site = $(if ($env:CLIF_SITE) { $env:CLIF_SITE } else { "uchicago" }),
  [switch]$Rebuild,
  [switch]$RequireStage,
  [switch]$SkipVerify
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:CLIF_SITE = $Site

$Py = ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
  Write-Error "No virtualenv at .venv. Create it with:`n  py -3.12 -m venv .venv; .venv\Scripts\python.exe -m pip install -r requirements.txt"
}
if (-not (Test-Path "sites\$Site.json")) {
  Write-Error "No site profile at sites\$Site.json. Create it with:`n  Copy-Item sites\uchicago.example.json sites\$Site.json   # then edit data_path"
}

$Pass = @()
if ($Rebuild)      { $Pass += "--rebuild" }
if ($RequireStage) { $Pass += "--require-stage" }

$Out = "output\$Site"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Csv = "$Out\run_timings.csv"
if (-not (Test-Path $Csv)) { "run_started_utc,site,stage,seconds,hms,git_sha" | Set-Content $Csv }
$StartUtc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$Sha = (& git rev-parse --short HEAD 2>$null); if (-not $Sha) { $Sha = "NA" }

function Invoke-Stage {
  param([string]$Script, [string[]]$Extra = @())
  $name = [IO.Path]::GetFileNameWithoutExtension($Script)
  Write-Host ""; Write-Host "========== $name =========="
  $sw = [Diagnostics.Stopwatch]::StartNew()
  & $Py $Script --site $Site @Extra
  if ($LASTEXITCODE -ne 0) { throw "$name failed with exit code $LASTEXITCODE" }
  $sw.Stop()
  $d = [int]$sw.Elapsed.TotalSeconds
  $hms = "{0}h{1:d2}m{2:d2}s" -f [int]($d/3600), [int](($d%3600)/60), ($d%60)
  Add-Content $Csv "$StartUtc,$Site,$name,$d,$hms,$Sha"
  Write-Host ">>> $name`: $hms"
}

$total = [Diagnostics.Stopwatch]::StartNew()
Invoke-Stage "code\00_probe_units.py"        $Pass
Invoke-Stage "code\01_build_cohort.py"       $Pass
Invoke-Stage "code\02_metric_records.py"     $Pass
Invoke-Stage "code\03_build_panel.py"
Invoke-Stage "code\04_exposure_assignment.py"
if (-not $SkipVerify) { Invoke-Stage "code\05_verify_equivalence.py" }
$total.Stop()

$d = [int]$total.Elapsed.TotalSeconds
$hms = "{0}h{1:d2}m{2:d2}s" -f [int]($d/3600), [int](($d%3600)/60), ($d%60)
Add-Content $Csv "$StartUtc,$Site,TOTAL,$d,$hms,$Sha"
Write-Host ""
Write-Host "=== done in $hms - panel: $Out\final_no_phi\unit_period_panel.csv ==="
