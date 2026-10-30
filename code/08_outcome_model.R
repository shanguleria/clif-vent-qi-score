# ==============================================================================
# 08_outcome_model.R  --  unit ventilator-QI score vs in-hospital mortality
#
# Purpose : The composite ventilator quality score of the ICU a patient spent
#           most of their hospitalisation in (qclif), against in-hospital death,
#           by mixed-effects logistic regression with a unit random intercept.
# Author  : Shan Guleria
# Created : 2026-10-29
# Inputs  : output/<site>/intermediate_phi/analysis/analysis_dataset.parquet
#           output/<site>/final_no_phi/unit_panel.csv          (the exposure)
#           output/<site>/final_no_phi/panel_provenance.json   (how it was built)
#           output/<site>/intermediate_phi/analysis/manifest.json
# Outputs : output/<site>/final_no_phi/08_outcome_model/ : the files in OWNED
#
# SKELETON. Sections marked TODO are stubs. Nothing here is a result yet.
# ==============================================================================

# Run this in a FRESH R session (RStudio: Cmd+Shift+F10).

# ---- 1. Packages -------------------------------------------------------------

pkgs <- c("here", "jsonlite", "arrow", "lme4", "broom.mixed", "ggplot2")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, repos = "https://cloud.r-project.org")
  }
  library(p, character.only = TRUE)
}


# ---- 2. Config ---------------------------------------------------------------
# This repo's configuration is the Python pipeline's: the site profile carries the
# access and grain settings, definitions/composite.json carries the score spec.
# Read them rather than restating any of it here.

SITE <- Sys.getenv("CLIF_SITE", "uchicago")
site_profile <- fromJSON(here("sites", paste0(SITE, ".json")), simplifyVector = FALSE)
composite    <- fromJSON(here("definitions", "composite.json"), simplifyVector = FALSE)

OUT_ROOT  <- here("output", SITE)
PHI_DIR   <- file.path(OUT_ROOT, "intermediate_phi")
SHARE_DIR <- file.path(OUT_ROOT, "final_no_phi")

# The exposure. score_core is the primary; score_full (which adds proning) is the
# prespecified sensitivity. See definitions/composite.json _comment_scores.
EXPOSURE      <- "score_core"
EXPOSURE_SENS <- "score_full"
COVARIATES    <- c("age_at_admission", "cci", "sofa")
OUTCOME       <- "died"
CLUSTER       <- "unit_name"

# Reporting scale: the model term is per SD of the score, because a one-point
# change in a 0-100 composite is not an interpretable unit.
EXPOSURE_SCALE <- "sd"

stopifnot(
  "definitions/composite.json must declare the primary exposure" =
    !is.null(composite$scores[[EXPOSURE]]),
  "score_core must be the primary_exposure" =
    identical(composite$scores[[EXPOSURE]]$role, "primary_exposure"),
  "score_full must be the sensitivity" =
    identical(composite$scores[[EXPOSURE_SENS]]$role, "sensitivity"))


# ---- 3. Paths and provenance -------------------------------------------------

dirs <- list(
  phase       = file.path(SHARE_DIR, "08_outcome_model"),
  diagnostics = file.path(SHARE_DIR, "08_outcome_model", "diagnostics"),
  logs        = here("logs"))
for (d in dirs) dir.create(d, recursive = TRUE, showWarnings = FALSE)

panel_prov <- fromJSON(file.path(SHARE_DIR, "panel_provenance.json"),
                       simplifyVector = FALSE)
analysis_manifest <- fromJSON(file.path(PHI_DIR, "analysis", "manifest.json"),
                              simplifyVector = FALSE)

OWNED <- list(
  # Reported directly.
  phase = c("table1.csv", "exposure_by_unit.csv",
            "model_coefficients.csv", "model_covariance.csv",
            "strobe.csv", "provenance.json", "captions.md",
            "qclif_by_unit.png", "adjusted_prediction.png"),
  # Method checks: evidence about whether the model is admissible, not results.
  diagnostics = c("cluster_summary.csv", "icc.csv", "collinearity.csv",
                  "sofa_component_missingness.csv", "singularity.csv"))


# ---- 4. Guards ---------------------------------------------------------------

message(sprintf("  site %s | pipeline code %s | definitions %s @ %s",
                SITE,
                panel_prov$provenance$code_version,
                panel_prov$provenance$definition_version,
                panel_prov$provenance$upstream_sha))

stopifnot(
  "the panel must be at unit grain for a unit-level exposure" =
    identical(panel_prov$grains$unit$role, "primary"),
  "the analysis dataset must be one row per patient" =
    grepl("one row per patient", analysis_manifest$grain))

# ---- 5. Read -----------------------------------------------------------------

d_raw <- read_parquet(file.path(PHI_DIR, "analysis", "analysis_dataset.parquet"))
panel <- read.csv(file.path(SHARE_DIR, "unit_panel.csv"), stringsAsFactors = FALSE)

message(sprintf("  %s patients, %s units",
                format(nrow(d_raw), big.mark = ","),
                length(unique(d_raw[[CLUSTER]]))))

stopifnot("the exposure must be present in the panel" = EXPOSURE %in% names(panel))


# ---- 6. The analysis set -----------------------------------------------------
# Complete-case on the model covariates. At UChicago this drops ~4 patients, but
# state it rather than assume it: SOFA sums a MISSING subscore as 0, so a SOFA of
# 0 can mean "well" or "nothing was charted". diagnostics/ records which.

d <- d_raw
d$.complete <- stats::complete.cases(d[, c(OUTCOME, "qclif", COVARIATES)])

strobe <- data.frame(
  step = c("patients in the analysis dataset",
           "excluded: incomplete on a model covariate",
           "analysed"),
  n = c(nrow(d), sum(!d$.complete), sum(d$.complete)))

d <- d[d$.complete, , drop = FALSE]
d[[CLUSTER]] <- factor(d[[CLUSTER]])

# Standardise so every term is an odds ratio per SD and the exposure is on the
# same footing as the covariates.
z <- function(x) as.numeric(scale(x))
d$qclif_z <- z(d$qclif)
for (v in COVARIATES) d[[paste0(v, "_z")]] <- z(d[[v]])

N_CLUSTERS <- nlevels(d[[CLUSTER]])


# ---- 7. Descriptives ---------------------------------------------------------
# TODO: Table 1 by outcome, and the exposure-by-unit table. The house pattern is
# a gtsummary-shaped frame rendered through the dashboard template; here a plain
# CSV is enough until the manuscript needs it.

cluster_summary <- do.call(rbind, lapply(levels(d[[CLUSTER]]), function(u) {
  s <- d[d[[CLUSTER]] == u, ]
  data.frame(unit = u, n = nrow(s), deaths = sum(s[[OUTCOME]]),
             mortality = mean(s[[OUTCOME]]), qclif = s$qclif[1],
             age = median(s$age_at_admission), cci = median(s$cci),
             sofa = median(s$sofa))
}))

# Case mix tracks the exposure here (corr(qclif, unit mean SOFA) ~ +0.80 at
# UChicago), which is why the crude and adjusted estimates point opposite ways.
# Record it rather than discover it in review.
collinearity <- data.frame(
  pair = c("qclif ~ unit mean sofa", "qclif ~ unit mortality"),
  pearson = c(cor(cluster_summary$qclif, cluster_summary$sofa),
              cor(cluster_summary$qclif, cluster_summary$mortality)))


# ---- 8. The models -----------------------------------------------------------

f_crude    <- as.formula(sprintf("%s ~ qclif_z + (1 | %s)", OUTCOME, CLUSTER))
f_adjusted <- as.formula(sprintf("%s ~ qclif_z + %s + (1 | %s)", OUTCOME,
                                 paste0(COVARIATES, "_z", collapse = " + "),
                                 CLUSTER))

fit_one <- function(form, label) {
  m <- glmer(form, data = d, family = binomial(link = "logit"),
             control = glmerControl(optimizer = "bobyqa",
                                    optCtrl = list(maxfun = 2e5)))
  tid <- broom.mixed::tidy(m, effects = "fixed", conf.int = TRUE, exponentiate = TRUE)
  tid$model <- label
  tid$singular <- isSingular(m)
  list(fit = m, tidy = tid)
}

models <- list(
  crude    = fit_one(f_crude, "crude"),
  adjusted = fit_one(f_adjusted, "adjusted"))

# TODO: sensitivity analyses, each a row in model_coefficients.csv
#   (a) EXPOSURE_SENS (score_full, incl. proning) in place of score_core
#   (b) restricted to units above their min_denominator floor (score_core_low)
#   (c) excluding patients whose hospitalisation spanned >1 ICU (n_icus > 1)
#   (d) SOFA without the liver subscore, the most unit-differential component

coefs <- do.call(rbind, lapply(models, `[[`, "tidy"))

# ICC: how much of the outcome variance sits between units at all. With few
# clusters this is a descriptive quantity, not an inferential one.
icc_of <- function(m) {
  v <- as.numeric(VarCorr(m)[[CLUSTER]])
  v / (v + pi^2 / 3)                       # latent-scale ICC, logistic link
}
icc <- data.frame(model = names(models),
                  between_unit_var = sapply(models, function(x) as.numeric(VarCorr(x$fit)[[CLUSTER]])),
                  icc = sapply(models, function(x) icc_of(x$fit)))

singularity <- data.frame(
  model = names(models),
  singular = sapply(models, function(x) x$tidy$singular[1]),
  note = "a singular fit means the between-unit variance collapsed to ~0; with few clusters this is expected and makes the random intercept uninformative")


# ---- 9. Figures --------------------------------------------------------------
# House rule: no title, no subtitle. Axis titles, tick labels and the legend stay;
# the n, the denominator and any methodological note belong in captions.md.
# Legend at the foot, in a box. Type near-black, never grey.
# TODO: qclif_by_unit.png (unit mortality vs qclif, point size = n)
# TODO: adjusted_prediction.png (predicted mortality across the observed qclif
#       range at covariate means, with the CI)


# ---- 10. Write ---------------------------------------------------------------

write.csv(strobe,          file.path(dirs$phase, "strobe.csv"), row.names = FALSE)
write.csv(cluster_summary, file.path(dirs$phase, "exposure_by_unit.csv"), row.names = FALSE)
write.csv(coefs,           file.path(dirs$phase, "model_coefficients.csv"), row.names = FALSE)
write.csv(cluster_summary, file.path(dirs$diagnostics, "cluster_summary.csv"), row.names = FALSE)
write.csv(icc,             file.path(dirs$diagnostics, "icc.csv"), row.names = FALSE)
write.csv(collinearity,    file.path(dirs$diagnostics, "collinearity.csv"), row.names = FALSE)
write.csv(singularity,     file.path(dirs$diagnostics, "singularity.csv"), row.names = FALSE)

# PHI gate: this phase writes into final_no_phi/, so nothing here may carry a
# row-level identifier. Mirrors phi_gate() in code/03_build_panel.py.
forbidden <- c("patient_id", "hospitalization_id", "encounter_block", "episode_id")
for (f in list.files(dirs$phase, pattern = "[.]csv$", full.names = TRUE)) {
  hdr <- names(read.csv(f, nrows = 1))
  bad <- intersect(tolower(hdr), forbidden)
  if (length(bad)) stop(sprintf("[phi-gate] %s carries %s", basename(f),
                                paste(bad, collapse = ", ")))
}


# ---- 11. Provenance ----------------------------------------------------------

write_json(list(
  site = SITE,
  script = "code/08_outcome_model.R",
  generated = format(Sys.time(), tz = site_profile$timezone, usetz = TRUE),
  exposure = EXPOSURE,
  exposure_scale = EXPOSURE_SCALE,
  covariates = COVARIATES,
  outcome = OUTCOME,
  cluster = CLUSTER,
  n_analysed = nrow(d),
  n_clusters = N_CLUSTERS,
  pipeline_provenance = panel_prov$provenance,
  sofa = analysis_manifest$sofa,
  outcome_definition = analysis_manifest$outcome
), file.path(dirs$phase, "provenance.json"), auto_unbox = TRUE, pretty = TRUE)

writeLines(
  c(paste("Run at:", format(Sys.time(), tz = site_profile$timezone, usetz = TRUE)),
    paste("Script :", "code/08_outcome_model.R"),
    "",
    capture.output(sessionInfo())),
  file.path(dirs$logs, "08_outcome_model_sessioninfo.txt"))

cat(sprintf("\nDone. %s patients analysed across %s units.\n",
            format(nrow(d), big.mark = ","), N_CLUSTERS))
cat("THIS IS AN ASSOCIATION. The exposure is a unit-level process measure and\n")
cat("case mix tracks it closely -- the higher-scoring ICUs here are the medical\n")
cat("ICUs, which take the sickest patients. Age, Charlson and SOFA are the only\n")
cat("adjustment, and documentation quality sits upstream of BOTH the quality\n")
cat("score and the severity measurement. Say 'associated with', never 'improves'.\n")
