# Print the aggregate, PHI-free outputs for one site or for the pooled run, ready to share.
# Windows.
#
#   .\show_results.ps1 -Site <id>      the site's covariate probe log (and its stage-8
#                                      summaries, if stage 8 has run for that site)
#   .\show_results.ps1 -Site pooled    the pooling log and the pooled model summaries
#
# Reads only the files named below, all under final_no_phi\. Per-unit tables
# (unit_summary.csv, unit_effects.csv) are left out: they carry per-unit counts that can fall
# below the small-cell threshold.
param(
  [string]$Site = $env:CLIF_SITE
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not $Site -or $Site -notmatch '^[A-Za-z0-9_-]+$') {
  Write-Error "usage: .\show_results.ps1 -Site <id>|pooled"
}

$Share = "output\$Site\final_no_phi"
$Model = "$Share\08_outcome_model"
$Summaries = @("icc.csv", "covariate_blocks.csv", "sensitivity_analyses.csv", "model_coefficients.csv")

$Files = @()
if ($Site -eq "pooled") {
  $Files += "$Share\logs\07c_pool_sites.log"
  $Files += $Summaries | ForEach-Object { "$Model\$_" }
} else {
  $Files += "$Share\logs\07b_covariate_probe.log"
  if (Test-Path $Model) { $Files += $Summaries | ForEach-Object { "$Model\$_" } }
}

foreach ($f in $Files) {
  Write-Host ""
  if (Test-Path $f) {
    Write-Host "===== $f"
    Get-Content $f
  } else {
    Write-Host "===== $f - not found (has that step run for '$Site'?)"
  }
}
