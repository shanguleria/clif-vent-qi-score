# Build the ventilator QI score panel and patient rows for one site (Windows).
#
#   .\run_scores.ps1 [-Site <id>] [-Rebuild] [-Fresh] [-From <stage>] [-Probe]
#                    [-RequireStage] [-SkipVerify]
#
# -Rebuild       ignore any stage cache and rebuild the cohort from raw CLIF (slow: re-runs
#                encounter stitching and the clifpy respiratory-support waterfall)
# -Fresh         move output\<site> aside to a timestamped backup first; implies -Rebuild
# -From <stage>  start at 00, 01, 02, 03, 04 or 07, reusing the earlier stages' outputs
#                (e.g. -From 07 after a stage-7 fix); -Rebuild and -Fresh need -From 00
# -Probe         run the 07b covariate probe after stage 7
# -RequireStage  fail rather than fall back to a cold build (for CI)
param(
  [string]$Site = $(if ($env:CLIF_SITE) { $env:CLIF_SITE } else { "uchicago" }),
  [switch]$Rebuild,
  [switch]$Fresh,
  [ValidateSet("00", "01", "02", "03", "04", "07")][string]$From = "00",
  [switch]$Probe,
  [switch]$RequireStage,
  [switch]$SkipVerify
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:CLIF_SITE = $Site

if ($Fresh) { $Rebuild = $true }
if ($From -ne "00" -and $Rebuild) {
  Write-Error "-Rebuild and -Fresh rebuild the cohort in stage 0, so they need -From 00"
}

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
if ($Fresh -and (Test-Path $Out)) {
  $Backup = "output\${Site}_backup_$(Get-Date -Format yyyyMMdd_HHmmss)"
  Move-Item $Out $Backup
  Write-Host "moved $Out to $Backup (delete it once this run succeeds)"
}
if ($From -ne "00" -and -not (Test-Path "$Out\intermediate_phi\cohort\source.json")) {
  Write-Error "No earlier run for '$Site' to resume from ($Out\intermediate_phi\cohort\source.json).`nRun from stage 00 first: .\run_scores.ps1 -Site $Site"
}

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
  $hms = "{0}h{1:d2}m{2:d2}s" -f [int][math]::Floor($d/3600), [int][math]::Floor(($d%3600)/60), ($d%60)
  Add-Content $Csv "$StartUtc,$Site,$name,$d,$hms,$Sha"
  Write-Host ">>> $name`: $hms"
}
function Test-FromHere([string]$Stage) { return [int]$Stage -ge [int]$From }

$total = [Diagnostics.Stopwatch]::StartNew()
# Stage 0 resolves the cohort source (building it cold when needed) and records the result;
# every later stage reuses that record. -Rebuild therefore goes ONLY to stage 0 - handing it
# further down would re-run the upstream waterfall build once per stage.
if (Test-FromHere "00") { Invoke-Stage "code\00_probe_units.py" $Pass }
if (Test-FromHere "01") { Invoke-Stage "code\01_build_cohort.py" }
if (Test-FromHere "02") { Invoke-Stage "code\02_metric_records.py" }
if (Test-FromHere "03") { Invoke-Stage "code\03_build_panel.py" }
if (Test-FromHere "04") { Invoke-Stage "code\04_exposure_assignment.py" }
Invoke-Stage "code\07_analysis_dataset.py"
if ($Probe) { Invoke-Stage "code\07b_covariate_probe.py" }
if (-not $SkipVerify) { Invoke-Stage "code\05_verify_equivalence.py" }
$total.Stop()

$d = [int]$total.Elapsed.TotalSeconds
$hms = "{0}h{1:d2}m{2:d2}s" -f [int][math]::Floor($d/3600), [int][math]::Floor(($d%3600)/60), ($d%60)
Add-Content $Csv "$StartUtc,$Site,TOTAL,$d,$hms,$Sha"
Write-Host ""
Write-Host "=== done in $hms - panel: $Out\final_no_phi\unit_period_panel.csv ==="
if ($Probe) { Write-Host "    probe log: .\show_results.ps1 -Site $Site" }
