# 08_outcome_model.R -- unit ventilator-QI score vs in-hospital mortality
#
# Input:  output/<site>/intermediate_phi/analysis/analysis_dataset_extended.parquet
#         output/<site>/final_no_phi/unit_panel.parquet
#
# Output: output/<site>/final_no_phi/08_outcome_model/, or 08_outcome_model_<cohort>/ when
#         CLIF_COHORT names a restricted cohort (see COHORTS)

library(arrow)
library(lme4)
library(ggplot2)
source(file.path("code", "caterpillar.R"))
# Print each warning as it occurs, beside the [time] line of its section, instead of a deferred
# "There were 50 or more warnings" with the text lost.
options(warn = 1)

site <- Sys.getenv("CLIF_SITE", "uchicago")
cohort <- Sys.getenv("CLIF_COHORT", "all")
out <- model_out_dir(site, cohort)
dir.create(out, recursive = TRUE, showWarnings = FALSE)

TIMES <- data.frame(section = character(), seconds = numeric())
lap <- local({
  last <- proc.time()[["elapsed"]]
  function(section) {
    now <- proc.time()[["elapsed"]]
    TIMES[nrow(TIMES) + 1, ] <<- list(section, round(now - last, 1))
    cat(sprintf("[time] %-32s %8.1f s\n", section, now - last))
    last <<- now
  }
})

df <- as.data.frame(read_parquet(
  file.path("output", site, "intermediate_phi", "analysis",
            "analysis_dataset_extended.parquet")))
df <- df[complete.cases(df[, c("died", "qclif", "sofa", "cci", "age_at_admission")]), ]

# Restrictions of the outcome cohort. The exposure is untouched: each unit's score still comes
# from all of its ventilated care. Location is the ADT location at the first ventilator record,
# which for patients intubated in the OR is usually the ICU they arrive in, so "non-procedural"
# also needs the procedural-stay flag. Hypoxaemia uses the worst P/F and S/F in the SOFA window
# (24 h before to 4 h after intubation; S/F only from SpO2 <= 97%), Rice 2007 equivalents.
EARLY_LOCATIONS <- c("ed", "ward_other", "icu")
EARLY_HOURS <- c("<= 0 h", "0-24 h")
early_nonop <- function(d) d$location_at_intubation %in% EARLY_LOCATIONS &
  d$hours_to_intubation %in% EARLY_HOURS & d$procedural_within_24h == 0
hypoxaemic <- function(d) (!is.na(d$pf) & d$pf < 300) | (!is.na(d$sf) & d$sf < 315)
COHORTS <- list(
  all                   = list(label = "all patients",
                               keep = function(d) rep(TRUE, nrow(d))),
  early_nonop           = list(label = paste("first IMV in the ED, ward or ICU within 24 h of",
                                             "admission, no procedural stay in the prior 24 h"),
                               keep = early_nonop),
  early_nonop_hypoxemia = list(label = paste("the early non-procedural cohort with P/F < 300 or",
                                             "S/F < 315 within 24 h before to 4 h after intubation"),
                               keep = function(d) early_nonop(d) & hypoxaemic(d)))
if (!cohort %in% names(COHORTS))
  stop(sprintf("CLIF_COHORT='%s' is not one of: %s", cohort, paste(names(COHORTS), collapse = ", ")))
n_before <- nrow(df)
df <- df[COHORTS[[cohort]]$keep(df), ]
cat(sprintf("cohort '%s': %d of %d patients\n", cohort, nrow(df), n_before))
stopifnot(nrow(df) > 0)
COHORT_LABEL <- COHORTS[[cohort]]$label
# The sensitivity analyses and covariate blocks are fit for the primary cohort only; the others
# get the m1-m3 chain, its variance tests, the per-unit effects, Table 1 and the figures.
PRIMARY_COHORT <- "early_nonop"
stopifnot(PRIMARY_COHORT %in% names(COHORTS))
FULL <- cohort == PRIMARY_COHORT
PRIMARY_ONLY_FILES <- c("sensitivity_analyses.csv", "covariate_blocks.csv")
if (!FULL) unlink(file.path(out, PRIMARY_ONLY_FILES))   # a stale copy from an earlier run would be shown

# Physiology at intubation, the primary severity adjustment in place of SOFA. Each value is the
# worst in the SOFA window (24 h before to 4 h after intubation) except GCS, the worst in the 24 h
# BEFORE it: after the airway, GCS mostly measures sedation. An unmeasured value is set to a fixed
# normal one and flagged, so the flag carries the difference between "not measured" and "normal".
# NEE has no flag: no vasopressor is a true zero. Oxygenation is P/F, else S/F as its Rice 2007
# P/F equivalent, with its source (P/F, S/F, none) as the flag. Continuous terms are natural
# splines with knots at the tertiles of the MEASURED values, so a pile of fills cannot collapse
# two knots onto one value; bases are built once per cohort and reused by every refit.
PHYS_NORMAL <- c(pf_equiv = 400, map = 85, gcs_pre = 15, platelet_count = 250,
                 bilirubin_total = 0.7, creatinine = 0.9)
SPLINE_KNOTS <- c(1, 2) / 3
RICE_SF_INTERCEPT <- 64     # S/F = 64 + 0.84 x P/F; common/oxygenation.py SF_FOR_PF uses the same
RICE_SF_SLOPE <- 0.84
# An integer score such as GCS can put both tertiles on one value, or on the boundary; the knots
# kept are the distinct interior ones, so such a term gets fewer degrees of freedom, reported below.
phys_ns <- function(x, measured, name) {
  stopifnot(!anyNA(x), sum(measured) >= 10)
  k <- unique(quantile(x[measured], SPLINE_KNOTS, names = FALSE))
  k <- k[k > min(x) & k < max(x)]
  b <- splines::ns(x, knots = k, Boundary.knots = range(x))
  setNames(as.data.frame(matrix(b, nrow(b))), paste0(name, "_ns", seq_len(ncol(b))))
}
stopifnot(!anyNA(df$nee))
df$pf_equiv <- ifelse(!is.na(df$pf), df$pf, (df$sf - RICE_SF_INTERCEPT) / RICE_SF_SLOPE)
oxy <- ifelse(!is.na(df$pf), "P/F", ifelse(!is.na(df$sf), "S/F", "none"))
df$oxygenation_source <- factor(oxy, levels = intersect(c("P/F", "S/F", "none"), oxy))
PHYS_GROUPS <- list()
for (v in names(PHYS_NORMAL)) {
  measured <- !is.na(df[[v]])
  b <- phys_ns(ifelse(measured, df[[v]], PHYS_NORMAL[[v]]), measured, v)
  df[names(b)] <- b
  flag <- if (v == "pf_equiv") "oxygenation_source" else paste0(v, "_unmeasured")
  if (v != "pf_equiv") df[[flag]] <- as.integer(!measured)
  PHYS_GROUPS[[v]] <- c(names(b), flag)
}
df$nee_any <- as.integer(df$nee > 0)
b <- phys_ns(log1p(df$nee), df$nee > 0, "nee")
df[names(b)] <- b
PHYS_GROUPS$nee <- c("nee_any", names(b))
PHYS_TERMS <- unlist(PHYS_GROUPS, use.names = FALSE)
for (v in names(PHYS_GROUPS))
  cat(sprintf("physiology %-16s %-15s %5.1f%%  spline df %d\n", v,
              if (v == "nee") "on a vasopressor" else "measured",
              100 * if (v == "nee") mean(df$nee > 0) else mean(!is.na(df[[v]])),
              sum(grepl("_ns[0-9]+$", PHYS_GROUPS[[v]]))))

# The primary adjustment. Entries name model terms, so age enters as age_z.
ADJUST <- c(PHYS_TERMS, "cci_poa", "age_z", "hours_to_intubation", "location_at_intubation",
            "procedural_within_24h", "limited_code_status", "admission_type")
ADJUST_LABEL <- paste("age, present-on-admission Charlson comorbidity index, physiology at",
                      "intubation (oxygenation, mean arterial pressure, vasopressor dose,",
                      "pre-intubation GCS, platelets, bilirubin and creatinine),",
                      "admission-to-intubation time, location at intubation, a procedural",
                      "stay in the 24 h before intubation, limited code status and admission type")
REF <- c(hours_to_intubation = "0-24 h", location_at_intubation = "icu", admission_type = "ed")
HOURS_LEVELS <- c("<= 0 h", "0-24 h", "24-72 h", "72 h-7 d", "> 7 d")
# Sensitivity only: the primary cohort has no minimum IMV duration.
MIN_IMV_HOURS <- 24
stopifnot("first_episode_hours" %in% names(df))
SOFA_COMPONENTS <- c(sofa_resp = "Respiratory", sofa_coag = "Coagulation",
                     sofa_liver = "Liver", sofa_cv_97 = "Cardiovascular",
                     sofa_cns = "Central nervous system", sofa_renal = "Renal")
raw_cols <- function(terms) sub("^age_z$", "age_at_admission", terms)
stopifnot(!anyNA(df[, raw_cols(ADJUST)]))
# `other` is sparse and matches `direct` on crude mortality, so it joins it. `osh` and `facility`
# stay separate: where both are used, facility admissions are a distinctly sicker group, even
# though one site codes most of its transfers as `facility`. Stage 7 keeps the raw categories.
ADMISSION_RECODE <- c(other = "direct")
adm <- as.character(df$admission_type)
df$admission_type <- ifelse(adm %in% names(ADMISSION_RECODE), unname(ADMISSION_RECODE[adm]), adm)
for (v in names(REF)) df[[v]] <- relevel(factor(df[[v]]), ref = REF[[v]])

# Pooled data (code/07c_pool_sites.py) carries a `site` column. Site enters every specification as
# a fixed effect, so the score's coefficient is estimated from differences between units within a
# site; site-wide differences in charting or case mix go to the site term.
POOLED <- "site" %in% names(df)
if (POOLED) {
  df$site <- relevel(factor(df$site), ref = names(which.max(table(df$site))))
  ADJUST <- c(ADJUST, "site")
  ADJUST_LABEL <- paste0(ADJUST_LABEL, ", with a fixed effect for site")
}
SITE_LABEL <- if (POOLED) sprintf("%d pooled sites", nlevels(df$site)) else site

# A restricted cohort can make a term constant (procedural_within_24h is 0 throughout the early
# non-procedural cohort). It is dropped from every specification and the drop is reported.
CONSTANT <- Filter(function(v) length(unique(df[[raw_cols(v)]])) < 2, ADJUST)
if (length(CONSTANT)) {
  cat(sprintf("constant in cohort '%s', dropped from every model: %s\n",
              cohort, paste(CONSTANT, collapse = ", ")))
  ADJUST <- setdiff(ADJUST, CONSTANT)
  PHYS_TERMS <- setdiff(PHYS_TERMS, CONSTANT)
  PHYS_GROUPS <- lapply(PHYS_GROUPS, setdiff, CONSTANT)
  ADJUST_LABEL <- paste0(ADJUST_LABEL, " (", paste(CONSTANT, collapse = ", "),
                         " constant in this cohort, so omitted)")
}

panel <- as.data.frame(read_parquet(
  file.path("output", site, "final_no_phi", "unit_panel.parquet")))
prow <- match(df$unit_name, panel$unit_name)
stopifnot(!anyNA(prow))
# The exposure must BE the panel's score, not a copy that has drifted from it.
stopifnot(isTRUE(all.equal(df$qclif, panel$score_core[prow])))

# Pooled unit_name is "<letter>::<site's own unit name>"; everything written uses the anonymized
# unit_id ("Site A · Unit 3") instead.
disp <- function(u) if (POOLED) panel$unit_id[match(u, panel$unit_name)] else u
FIG_W <- max(7.5, 0.55 * nrow(panel))
lap("load and prepare")

# Rescale for interpretability and convergence
df$age_z <- as.numeric(scale(df$age_at_admission))
unit_q <- df$qclif[!duplicated(df$unit_name)]
df$qclif_centered <- df$qclif - mean(unit_q)   # per 1-point increase

# lme4's default maxfun is below bobyqa's recommended 10 * npar^2 for the ~45-parameter primary
# model, which then stops on the evaluation limit rather than converging.
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
# Cached on the exact rows and terms, so the full-cohort null and the primary specification are fit
# once and the sensitivity rows and covariate blocks that repeat them reuse m1-m3. Derived columns
# (age_z, qclif_centered) are functions of the rows, so the same rows give the same model.
FITS <- list()
FIT_KEYS <- character()
fit <- function(terms, d = df) {
  key <- paste(c(sort(terms), "|", rownames(d)), collapse = " ")
  hit <- match(key, FIT_KEYS)
  if (!is.na(hit)) return(FITS[[hit]])
  m <- glmer(reformulate(c(terms, "(1 | unit_name)"), response = "died"),
             data = d, family = binomial, control = ctrl)
  FIT_KEYS <<- c(FIT_KEYS, key)
  FITS[[length(FITS) + 1]] <<- m
  m
}
m1 <- fit("1"); lap("fit m1")
m2 <- fit(ADJUST); lap("fit m2")
m3 <- fit(c(ADJUST, "qclif_centered")); lap("fit m3")

anova(m1, m2, m3)
summary(m3)


# ---- Export ------------------------------------------------------------------

# Odds ratios, one row per term per model
coefs <- do.call(rbind, lapply(names(list(m2 = m2, m3 = m3)), function(nm) {
  m  <- get(nm)
  ci <- confint(m, method = "Wald", parm = "beta_")
  data.frame(model = nm, term = names(fixef(m)), estimate = unname(fixef(m)),
             OR = exp(unname(fixef(m))), lo = exp(ci[, 1]), hi = exp(ci[, 2]),
             p = coef(summary(m))[, "Pr(>|z|)"], row.names = NULL)
}))
write.csv(coefs, file.path(out, "model_coefficients.csv"), row.names = FALSE)

# Nested model comparison
cmp <- as.data.frame(anova(m1, m2, m3))
cmp$model <- rownames(cmp)
write.csv(cmp[, c("model", setdiff(names(cmp), "model"))],
          file.path(out, "model_comparison.csv"), row.names = FALSE)

# Between-unit variation, in four ways. The raw variance is on the log-odds
# scale and is hard to read; MOR is the one to quote.
#   icc  the share of LATENT variance that is between-unit. Its denominator
#        carries the logistic distribution's fixed individual-level variance
#        (pi^2/3 = 3.29), a floor that no amount of patient homogeneity lowers,
#        so a substantial unit effect still gives a single-digit ICC. Hospital
#        mortality ICCs in health-services research typically run 1-3%.
#   mor  median odds ratio: take two patients with identical covariates from two
#        randomly chosen units; this is the median ratio of their odds of death,
#        higher unit over lower. Directly comparable to a fixed-effect OR.
vc <- function(m) as.numeric(VarCorr(m)$unit_name)
icc <- data.frame(model = c("m1", "m2", "m3"),
                  adjustment = c("none", "primary adjustment",
                                 "primary adjustment + qclif"),
                  between_unit_var = sapply(list(m1, m2, m3), vc))
icc$sd_logodds <- sqrt(icc$between_unit_var)
icc$or_per_sd_unit <- exp(icc$sd_logodds)
icc$icc <- icc$between_unit_var / (icc$between_unit_var + pi^2 / 3)
mor_of_sd <- function(s) exp(sqrt(2) * s * qnorm(0.75))
icc$mor <- mor_of_sd(sqrt(icc$between_unit_var))
icc$var_explained_vs_m1 <- 1 - icc$between_unit_var / icc$between_unit_var[1]

# Is there between-unit variation at all? Each model against its twin with the same fixed effects
# and no random intercept. The null puts the variance on the boundary of its space, so the
# reference is a 50:50 mixture of chi-square(0) and chi-square(1) (Self & Liang 1987).
lrt <- function(m, terms) {
  g <- glm(reformulate(terms, response = "died"), data = df, family = binomial)
  max(0, 2 * (as.numeric(logLik(m)) - as.numeric(logLik(g))))
}
icc$lrt_chisq <- c(lrt(m1, "1"), lrt(m2, ADJUST), lrt(m3, c(ADJUST, "qclif_centered")))
icc$lrt_p <- 0.5 * pchisq(icc$lrt_chisq, df = 1, lower.tail = FALSE)
lap("coefficients and variance LRT")

writeLines(c(
  "between_unit_var  variance of the unit random intercept, LOG-ODDS scale",
  "sd_logodds        its square root; the typical distance of a unit from average",
  "or_per_sd_unit    odds ratio for a unit one SD above average vs the average unit",
  "icc               share of latent variance between units; denominator includes the",
  "                  logistic fixed individual-level variance pi^2/3 = 3.29, so this",
  "                  reads low by construction (1-3% is typical for hospital mortality)",
  "mor               median odds ratio: median ratio of the odds of death for two",
  "                  identical patients in two randomly chosen units. Quote this one.",
  "var_explained_vs_m1 fraction of m1's between-unit variance removed by adjustment",
  "lrt_chisq, lrt_p    likelihood-ratio test of the unit variance against the same fixed",
  "                  effects without a random intercept; p from the 50:50 chi-square(0)/(1)",
  "                  mixture, because the null value 0 is on the boundary",
  "between_unit_var_lo/hi, mor_lo/hi  95% profile-likelihood interval for the unit SD,",
  "                  carried to the variance and the MOR; m2 in the primary cohort only,",
  "                  NA for every other model and cohort"
), file.path(out, "icc_readme.txt"))

# Unadjusted, per unit
unit_tab <- aggregate(cbind(n = died, deaths = died) ~ unit_name, df,
                      function(x) c(length(x), sum(x)))
unit_tab <- data.frame(unit_name = unit_tab$unit_name,
                       n = unit_tab$n[, 1], deaths = unit_tab$deaths[, 2])
unit_tab$mortality <- unit_tab$deaths / unit_tab$n
unit_tab$qclif <- df$qclif[match(unit_tab$unit_name, df$unit_name)]
unit_tab <- unit_tab[order(unit_tab$mortality), ]
write.csv(transform(unit_tab, unit_name = disp(unit_name)), file.path(out, "unit_summary.csv"),
          row.names = FALSE)


# Per-unit effects left after adjusting for covariates but not qCLIF
re  <- ranef(m2, condVar = TRUE)$unit_name
u   <- re[, 1]
se  <- sqrt(as.numeric(attr(re, "postVar")))
if (vc(m2) > 0 && all(u == 0))
  stop("unit effects are all zero although the unit variance is not; m2 has been mutated")

# Adjusted predicted in-hospital survival for a standard patient using cohort-mean covariate profile
# plus each unit's random intercept.
eta_fixed <- sum(colMeans(model.matrix(m2)) * fixef(m2))
surv <- function(eta) 1 / (1 + exp(eta))       # P(survive) = 1 - plogis(eta_death)

cat_tab <- data.frame(
  unit_name = rownames(re),
  logodds_death = u, se = se,
  or_survival    = exp(-u),
  or_lo          = exp(-(u + 1.96 * se)),
  or_hi          = exp(-(u - 1.96 * se)),
  # higher u is more death, so the upper survival bound comes from u - 1.96 se
  survival       = 100 * surv(eta_fixed + u),
  lo             = 100 * surv(eta_fixed + u + 1.96 * se),
  hi             = 100 * surv(eta_fixed + u - 1.96 * se))
SURV_AVG <- 100 * surv(eta_fixed)              # the average unit, random effect 0
cat_tab$n <- unit_tab$n[match(cat_tab$unit_name, unit_tab$unit_name)]
cat_tab$qclif <- unit_tab$qclif[match(cat_tab$unit_name, unit_tab$unit_name)]
# Parker 2019 (JAMA 322:1789) Figure 2 classifies each centre against the MEAN
# centre at p < .05 and colours it low / average / high. Here the reference is
# the average unit, i.e. a random intercept of 0, so the test is whether the
# 95% interval excludes OR = 1.
cat_tab$group <- ifelse(cat_tab$or_hi < 1, "Below average",
                 ifelse(cat_tab$or_lo > 1, "Above average", "Not distinguishable"))
cat_tab <- cat_tab[order(cat_tab$survival), ]
cat_tab$icu <- icu_label(nrow(cat_tab))
write.csv(transform(cat_tab, unit_name = disp(unit_name)), file.path(out, "unit_effects.csv"),
          row.names = FALSE)


# ---- Table 1 -----------------------------------------------------------------

# Strata are tertiles of the UNITS ranked by score, not of the patient distribution. The score
# is lumpy enough across this few units that a patient-weighted cut can leave a stratum holding
# a single ICU, which would publish that ICU's case mix in a column of its own.
TERTILES <- c("Low qCLIF", "Middle qCLIF", "High qCLIF")
uq <- sort(unique(df$qclif))
stopifnot(length(uq) >= length(TERTILES))
df$tertile <- factor(
  TERTILES[cut(seq_along(uq), breaks = length(TERTILES), labels = FALSE)][match(df$qclif, uq)],
  levels = TERTILES)
stopifnot(all(table(df$tertile) > 0))

idx <- c(list(Overall = rep(TRUE, nrow(df))),
         setNames(lapply(TERTILES, function(t) df$tertile == t), TERTILES))
hdr <- vapply(names(idx), function(nm)
  sprintf("%s  N = %s", nm, format(sum(idx[[nm]]), big.mark = ",")), "")

fmt_iqr <- function(x, dec) {
  x <- x[!is.na(x)]
  if (!length(x)) return("NA")
  q <- quantile(x, c(0.25, 0.5, 0.75), names = FALSE)
  f <- paste0("%.", dec, "f")
  sprintf(paste0(f, " (", f, ", ", f, ")"), q[2], q[1], q[3])
}
fmt_npct <- function(k, n) if (n > 0) sprintf("%s (%.0f%%)", format(k, big.mark = ","),
                                              100 * k / n) else "NA"
fmt_p <- function(p) {
  if (is.na(p)) return("NA")
  if (p < 0.001) return("<0.001")
  if (p < 0.01) sprintf("%.3f", p) else sprintf("%.2f", p)
}
p_cont <- function(v) {
  ok <- !is.na(v)
  g <- droplevels(df$tertile[ok])
  if (nlevels(g) < 2) return(NA_real_)
  kruskal.test(v[ok], g)$p.value
}
p_cat <- function(f) {
  tb <- table(f, df$tertile)
  tb <- tb[rowSums(tb) > 0, , drop = FALSE]
  if (nrow(tb) < 2 || ncol(tb) < 2) return(NA_real_)
  suppressWarnings(chisq.test(tb)$p.value)
}
# An absent category is reported, never folded into "Other" -- a silent fold would make a
# charting gap indistinguishable from a real level.
as_level <- function(x) {
  lv <- if (is.factor(x)) levels(x) else NULL
  x <- as.character(x)
  x[trimws(x) == ""] <- NA
  f <- if (is.null(lv)) factor(x) else factor(x, levels = lv)
  if (anyNA(x)) {
    f <- addNA(f)
    levels(f)[is.na(levels(f))] <- "Unknown"
  }
  f
}

t1 <- list()
emit <- function(label, cells, p) t1[[length(t1) + 1]] <<- c(label, unname(cells), p)
bold <- function(s) sprintf("__%s__", s)
blank <- rep("", length(idx))

row_cont <- function(label, v, dec = 0, p = p_cont(v), sub = FALSE)
  emit(if (sub) label else bold(label),
       vapply(idx, function(i) fmt_iqr(v[i], dec), ""), fmt_p(p))

row_bin <- function(label, b, sub = FALSE)
  emit(if (sub) label else bold(label), vapply(idx, function(i) fmt_npct(sum(b[i]), sum(i)), ""),
       fmt_p(p_cat(factor(b))))

row_cat <- function(label, x) {
  f <- as_level(x)
  emit(bold(label), blank, fmt_p(p_cat(f)))
  for (lv in levels(f))
    emit(lv, vapply(idx, function(i) fmt_npct(sum(f[i] == lv), sum(i)), ""), "")
}

if (POOLED) row_cat("Site", df$site)
row_cont("Age at admission, years", df$age_at_admission)
row_cat("Sex", df$sex_category)
row_cat("Race", df$race_category)
row_cat("Ethnicity", df$ethnicity_category)
row_cont("Charlson comorbidity index", df$cci)
row_cont("Charlson comorbidity index, present on admission", df$cci_poa)
row_cont("SOFA at intubation", df$sofa)
emit(bold("SOFA components"), blank, "")
for (col in names(SOFA_COMPONENTS)) row_cont(SOFA_COMPONENTS[[col]], df[[col]], sub = TRUE)
# Measured values only; the fills that enter the model are not physiology.
PHYS_LABELS <- c(pf_equiv = "P/F ratio, or S/F as its P/F equivalent",
                 map = "Mean arterial pressure, mmHg",
                 gcs_pre = "GCS, worst in the 24 h before intubation",
                 platelet_count = "Platelets, x10^3/uL",
                 bilirubin_total = "Bilirubin, mg/dL", creatinine = "Creatinine, mg/dL")
PHYS_DEC <- c(pf_equiv = 0, map = 0, gcs_pre = 0, platelet_count = 0, bilirubin_total = 1,
              creatinine = 1)
stopifnot(setequal(names(PHYS_LABELS), names(PHYS_NORMAL)))
emit(bold("Physiology at intubation, worst value where measured"), blank, "")
for (v in names(PHYS_LABELS)) row_cont(PHYS_LABELS[[v]], df[[v]], dec = PHYS_DEC[[v]], sub = TRUE)
row_bin("Any vasopressor", df$nee > 0, sub = TRUE)
row_cont("Norepinephrine equivalent, mcg/kg/min, if any",
         ifelse(df$nee > 0, df$nee, NA_real_), dec = 2, sub = TRUE)
emit(bold("Not measured in the window"), blank, "")
for (v in names(PHYS_LABELS)) row_bin(PHYS_LABELS[[v]], is.na(df[[v]]), sub = TRUE)
row_cat("Admission type", df$admission_type)
row_cat("Admission to intubation", factor(df$hours_to_intubation, levels = HOURS_LEVELS))
row_cat("Location at intubation", df$location_at_intubation)
row_bin("Procedural stay in the 24 h before intubation", df$procedural_within_24h == 1)
row_bin("Limited code status before intubation", df$limited_code_status == 1)
row_cont("Hours in the attributed ICU", df$icu_hours_in_unit, dec = 1)
row_bin("Cared for in more than one ICU", !is.na(df$n_icus) & df$n_icus > 1)
# Tautological against strata defined by the score itself, so no p-value is reported.
row_cont("Ventilator QI score (score_core)", df$qclif, dec = 1, p = NA_real_)
row_bin("Died in hospital", df$died == 1)
row_cat("Discharge category", df$discharge_category)

t1df <- as.data.frame(do.call(rbind, t1), stringsAsFactors = FALSE)
names(t1df) <- c("Characteristic", hdr, "p-value")
write.csv(t1df, file.path(out, "table1.csv"), row.names = FALSE)

ter <- unique(df[, c("unit_id", "qclif", "tertile")])
ter$n <- as.integer(table(df$unit_id)[ter$unit_id])
ter <- ter[order(ter$qclif), ]
write.csv(ter, file.path(out, "table1_tertiles.csv"), row.names = FALSE)
lap("unit effects and Table 1")


# ---- Sensitivity analyses ----------------------------------------------------

# Each variant has its OWN null model on its own rows, so var_explained compares like with like; a
# subset that loses a whole unit changes the baseline it should be judged against. The *_adjusted
# columns are from the covariates-only model, the rest from the model with the score added.
fit_one <- function(id, desc, d, adjust = ADJUST, score = TRUE) {
  d <- droplevels(d[complete.cases(d[, c("died", "qclif", raw_cols(adjust))]), ])
  d$age_z <- as.numeric(scale(d$age_at_admission))
  d$qclif_centered <- d$qclif - mean(d$qclif[!duplicated(d$unit_name)])
  n0 <- fit("1", d)
  na <- fit(adjust, d)
  row <- data.frame(
    analysis = id, description = desc,
    n_patients = nrow(d), n_deaths = sum(d$died), n_units = length(unique(d$unit_name)),
    between_unit_var_adjusted = vc(na),
    mor_adjusted = mor_of_sd(sqrt(vc(na))),
    var_explained_by_covariates = 1 - vc(na) / vc(n0),
    aic_adjusted = AIC(na),
    row.names = NULL)
  if (score) {
    nf <- fit(c(adjust, "qclif_centered"), d)
    ci <- confint(nf, method = "Wald", parm = "beta_")
    v <- vc(nf)
    row <- cbind(row,
      qclif_or_per_point = exp(unname(fixef(nf)["qclif_centered"])),
      lo = exp(ci["qclif_centered", 1]), hi = exp(ci["qclif_centered", 2]),
      p = coef(summary(nf))["qclif_centered", "Pr(>|z|)"],
      between_unit_var = v,
      mor = mor_of_sd(sqrt(v)),
      var_explained_vs_null = 1 - v / vc(n0))
  }
  lap(paste("fit:", id))
  row
}

NON_PHYS <- setdiff(ADJUST, PHYS_TERMS)
# Block descriptions name the physiology set once rather than its spline columns.
describe <- function(terms)
  paste(c(if (any(terms %in% PHYS_TERMS)) "physiology", setdiff(terms, PHYS_TERMS)),
        collapse = " + ")

if (FULL) {
  # A missing subscore enters as 0, the convention the SOFA total already applies.
  for (col in names(SOFA_COMPONENTS)) df[[paste0(col, "_0")]] <- ifelse(is.na(df[[col]]), 0, df[[col]])
  sens <- rbind(
    fit_one("primary", "primary adjustment, score_core, all complete cases", df),
    fit_one("sofa_total", "SOFA total in place of the physiology terms",
            df, adjust = c("sofa", NON_PHYS)),
    fit_one("sofa_components", "the six SOFA subscores in place of the physiology terms, absent = 0",
            df, adjust = c(paste0(names(SOFA_COMPONENTS), "_0"), NON_PHYS)),
    fit_one("single_icu", "patients cared for in exactly one ICU",
            df[!is.na(df$n_icus) & df$n_icus == 1, ]),
    fit_one("imv_24h", "first IMV episode lasting at least 24 h",
            df[which(df$first_episode_hours >= MIN_IMV_HOURS), ]))
  if (POOLED) sens <- rbind(sens, do.call(rbind, lapply(levels(df$site), function(s)
    fit_one(paste0("without_", gsub(" ", "_", tolower(s))), paste("all sites except", s),
            df[df$site != s, ]))))
  # Stated outright, so a restriction that happens to exclude nobody on this site's data reads as
  # vacuous rather than as applied.
  sens$n_excluded <- sens$n_patients[1] - sens$n_patients
  SENS_COLS <- c("analysis", "description", "n_patients", "n_excluded", "n_deaths", "n_units",
                 "qclif_or_per_point", "lo", "hi", "p", "between_unit_var_adjusted",
                 "mor_adjusted", "var_explained_by_covariates", "between_unit_var", "mor",
                 "var_explained_vs_null", "aic_adjusted")
  write.csv(sens[, SENS_COLS], file.path(out, "sensitivity_analyses.csv"), row.names = FALSE)


  # ---- Covariate blocks --------------------------------------------------------

  # Cumulative, in a fixed prespecified order, ending at the primary adjustment; covariates only,
  # no score. Read for how each step moves the between-unit variance; with this few units the
  # order is a presentation choice, not a test. Site comes first when pooled, so its row is the
  # share of the unit variance that is between sites.
  STEPS <- list(
    site                = if (POOLED) "site",
    patient             = c("age_z", "cci_poa"),
    physiology          = PHYS_TERMS,
    intubation_setting  = c("hours_to_intubation", "location_at_intubation",
                            "procedural_within_24h"),
    limited_code_status = "limited_code_status",
    admission_type      = "admission_type")
  STEPS <- Filter(length, lapply(STEPS, setdiff, CONSTANT))
  BLOCKS <- setNames(Reduce(c, STEPS, accumulate = TRUE), names(STEPS))
  # Every primary term is claimed by exactly one step, so the last block IS the primary model.
  stopifnot(!anyDuplicated(unlist(STEPS)), setequal(BLOCKS[[length(BLOCKS)]], ADJUST))
  blocks <- do.call(rbind, lapply(names(BLOCKS), function(b)
    fit_one(b, describe(BLOCKS[[b]]), df, adjust = BLOCKS[[b]], score = FALSE)))
  names(blocks)[names(blocks) == "analysis"] <- "block"
  write.csv(blocks, file.path(out, "covariate_blocks.csv"), row.names = FALSE)
}


# ---- Figures -----------------------------------------------------------------

INK <- "#0b0b0b"; GRIDLINE <- "#e1e0d9"; PAPER <- "#fcfcfb"; DIED <- "#8c2f18"
BAND <- "#dcdbd4"

# Ordered by unadjusted mortality
unit_tab$label <- sprintf("%s\nn = %s", disp(unit_tab$unit_name),
                          format(unit_tab$n, big.mark = ",", trim = TRUE))
unit_tab$label <- factor(unit_tab$label, levels = unit_tab$label)

p <- ggplot(unit_tab, aes(x = label, y = deaths)) +
  geom_col(fill = DIED, width = 0.7) +
  geom_text(aes(label = sprintf("%.1f%%", 100 * mortality)),
            vjust = -0.5, colour = INK, size = 3.4) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(x = "ICU (ordered by unadjusted in-hospital mortality)",
       y = "Patients who died in hospital") +
  theme_minimal(base_size = 12) +
  theme(axis.title = element_text(colour = INK),
        axis.text = element_text(colour = INK),
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(colour = GRIDLINE, linewidth = 0.4),
        plot.background = element_rect(fill = PAPER, colour = NA))

ggsave(file.path(out, "deaths_by_unit.png"), p, width = FIG_W, height = 4.5, dpi = 300)


plot_caterpillar(cat_tab, SURV_AVG, file.path(out, "unit_caterpillar.png"))


# Exposure-response: what the model says the score itself buys, as opposed to the
# caterpillar's per-unit intercepts. Covariates held at the cohort mean, random effect 0.
qs <- seq(min(unit_q), max(unit_q), length.out = 200)
xbar <- colMeans(model.matrix(m3))
newX <- matrix(xbar, nrow = length(qs), ncol = length(xbar), byrow = TRUE,
               dimnames = list(NULL, names(xbar)))
newX[, "qclif_centered"] <- qs - mean(unit_q)
eta <- as.numeric(newX %*% fixef(m3))
# Fixed-effect uncertainty only: a level-2 slope estimated from as many points as there are
# units, so this band is a lower bound on the true uncertainty.
se_eta <- sqrt(rowSums((newX %*% as.matrix(vcov(m3))) * newX))
curve <- data.frame(qclif = qs, fit = 100 * plogis(eta),
                    lo = 100 * plogis(eta - 1.96 * se_eta),
                    hi = 100 * plogis(eta + 1.96 * se_eta))

# Units are placed with m2's conditional modes, i.e. the model that has NOT seen the score,
# so the points are not fitted to the line they are plotted against.
pts <- data.frame(qclif = cat_tab$qclif, fit = 100 - cat_tab$survival, n = cat_tab$n)

er <- ggplot(curve, aes(x = qclif, y = fit)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), fill = BAND) +
  geom_line(colour = INK, linewidth = 0.7) +
  geom_point(data = pts, aes(size = n), colour = DIED, alpha = 0.85) +
  scale_size_area(max_size = 8, name = "Patients per ICU",
                  labels = function(x) format(x, big.mark = ",", trim = TRUE)) +
  scale_y_continuous(labels = function(x) sprintf("%.0f%%", x)) +
  labs(x = "Ventilator QI score (score_core, 0-100)",
       y = "Adjusted in-hospital mortality, standard patient") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom",
        legend.background = element_rect(fill = PAPER, colour = INK, linewidth = 0.3),
        legend.margin = margin(t = 5, r = 8, b = 5, l = 8),
        legend.box.margin = margin(t = 4),
        legend.text = element_text(colour = INK, size = 9),
        legend.title = element_text(colour = INK, size = 9),
        axis.title = element_text(colour = INK),
        axis.text = element_text(colour = INK),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(colour = GRIDLINE, linewidth = 0.4),
        plot.background = element_rect(fill = PAPER, colour = NA))

ggsave(file.path(out, "exposure_response.png"), er, width = 7.5, height = 4.5, dpi = 300)
lap("figures")

writeLines(c(
  "# Figure captions",
  "",
  sprintf("Cohort for every figure and table here: %s (n = %s patients).", COHORT_LABEL,
          format(nrow(df), big.mark = ",")),
  "",
  "**deaths_by_unit.png** — In-hospital deaths by ICU, among adults receiving",
  sprintf("invasive mechanical ventilation at %s (n = %s patients, first qualifying",
          SITE_LABEL, format(nrow(df), big.mark = ",")),
  sprintf("IMV hospitalisation per patient; %s deaths, %.1f%% overall). Units are ordered",
          format(sum(df$died), big.mark = ","), 100 * mean(df$died)),
  "by unadjusted in-hospital mortality. Bars are counts and therefore track unit volume",
  "as well as unit mortality;",
  "the denominator is printed under each unit and the unadjusted",
  "mortality above each bar. Death is defined as a discharge category of Expired or",
  "Hospice, or a recorded death within one day of discharge.",
  "",
  "**unit_caterpillar.png** — Caterpillar plot after Parker et al, JAMA 2019;322:1789,",
  "Figure 2. Conditional modes of the ICU random intercept from the adjusted model",
  sprintf("(%s), expressed as an odds ratio for survival", ADJUST_LABEL),
  "relative to the average ICU, with 95% intervals from the conditional standard",
  "errors, converted to the predicted probability of surviving to discharge for a",
  sprintf("patient at the cohort-mean covariate profile (reference line, the average unit, %.1f%%).",
          SURV_AVG),
  "Units are coloured by a separate two-sided Wald test for each unit of whether its random",
  "intercept differs from zero (|conditional mode / conditional SE| > 1.96): red, survival",
  "below the average unit; blue, above; black, not distinguishable. This compares each unit",
  "with the average unit, not units with one another, and is not corrected for multiple",
  "comparisons: if no unit truly differed, about 1 in 20 would still be coloured. The",
  "conditional standard errors treat the fixed effects and the between-unit variance as known,",
  "which narrows the intervals; shrinkage pulls small units toward the average, which widens",
  "the bar for flagging them.",
  if (POOLED) paste("The model includes a fixed effect for site, so each unit is compared with",
                    "the average unit after site-wide differences are removed.") else NULL,
  "Units are ordered by point estimate and labelled ICU 1 (lowest) onward, with no site or",
  sprintf("unit name; the icu column of unit_effects.csv is the key. n per unit ranges %s to %s.",
          format(min(cat_tab$n), big.mark = ","), format(max(cat_tab$n), big.mark = ",")),
  "",
  "**exposure_response.png** — Adjusted probability of in-hospital death as a function of",
  sprintf("the unit ventilator QI score (score_core), from the full model (%s,", ADJUST_LABEL),
  "and the score, with a random ICU intercept). The line holds",
  "every covariate at its cohort mean and the random effect at zero; the band is the 95%",
  "interval from the fixed-effect covariance alone. Points are the individual ICUs, placed",
  "at their own score and at the adjusted mortality implied by the conditional mode from the",
  "model WITHOUT the score, so they are not fitted to the line; point area is proportional to",
  sprintf("the number of patients (range %s to %s).",
          format(min(pts$n), big.mark = ","), format(max(pts$n), big.mark = ",")),
  sprintf("The score is a unit-level (level-2) exposure taking %d distinct values, so the slope is",
          length(unit_q)),
  "identified by that many points however many patients contribute; the band reflects",
  "fixed-effect uncertainty only and should be read as a lower bound.",
  "",
  "**table1.csv** — Patient characteristics at the first qualifying IMV hospitalisation,",
  "overall and by tertile of the unit ventilator QI score. Tertiles are cuts of the ICUs",
  "ranked by score rather than of the patient distribution, so no stratum can consist of a",
  "single ICU; strata therefore differ in size. The composition of each tertile, with its",
  "score range and patient count, is given in table1_tertiles.csv using anonymised unit",
  "labels. Continuous",
  "variables are median (Q1, Q3), categorical are n (%), and an absent value is reported as",
  "its own Unknown level rather than folded into another category. p-values are",
  "Kruskal-Wallis across the three tertiles for continuous variables and chi-square for",
  "categorical; they treat patients as independent and therefore IGNORE clustering within",
  "ICUs, so they are descriptive only. No p-value is reported for the score itself, which",
  "defines the strata. Cells are NOT small-cell suppressed: some categorical levels hold very",
  "few patients, so the table needs a suppression pass before it leaves the site.",
  if (FULL) c("",
  "**covariate_blocks.csv** — Cumulative covariate blocks ending at the primary adjustment, in a",
  "prespecified order:",
  if (POOLED) "site; then" else NULL,
  "age and the present-on-admission Charlson comorbidity index; then physiology at intubation;",
  "then admission-to-intubation time, location at intubation and a procedural stay in the 24 h",
  "before intubation; then limited code status (a DNR-type order at or before intubation; Full",
  "and no recorded order are one level, because recording varies by year and unit); then",
  "admission type. Each row gives the between-unit variance and median odds ratio with those",
  "covariates and no score, and the share of the unadjusted between-unit variance they remove.",
  "With this few units the order is a presentation choice, not a test.")
), file.path(out, "captions.md"))

# 95% profile-likelihood interval for the unit SD, carried to the variance and the MOR, both
# monotone in it. Last, because profiling leaves each model's reference-class state (@pp, @resp)
# at the final SD it tried, so ranef() and vcov() read afterwards are wrong. It refits the model
# many times, over an hour per model, so it runs for m2 in the primary cohort only: m2 carries the
# adjusted MOR, and m3's adds nothing its point estimate does not. Elsewhere the columns are NA.
# A failure is reported, not fatal.
sd_ci <- function(m) tryCatch(
  unname(confint(m, parm = "theta_", method = "profile", oldNames = FALSE)[1, ]),
  error = function(e) {
    message("profile interval for the unit SD failed: ", conditionMessage(e))
    c(NA_real_, NA_real_)
  })
ci_sd <- matrix(NA_real_, nrow = 3, ncol = 2)
if (FULL) { ci_sd[icc$model == "m2", ] <- sd_ci(m2); lap("profile CI m2") }
icc$between_unit_var_lo <- ci_sd[, 1]^2
icc$between_unit_var_hi <- ci_sd[, 2]^2
icc$mor_lo <- mor_of_sd(ci_sd[, 1])
icc$mor_hi <- mor_of_sd(ci_sd[, 2])
write.csv(icc, file.path(out, "icc.csv"), row.names = FALSE)

TIMES[nrow(TIMES) + 1, ] <- list("total", sum(TIMES$seconds))
write.csv(TIMES, file.path(out, "timings.csv"), row.names = FALSE)

cat(sprintf("\nWrote %s\n", out))
