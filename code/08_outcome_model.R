# 08_outcome_model.R -- unit ventilator-QI score vs in-hospital mortality
#
# Input: output/<site>/intermediate_phi/analysis/analysis_dataset.parquet
#        (one row per patient; swap in analysis_dataset_extended.parquet for the
#         SOFA subscores, raw components, n_icus and unit_id)

library(arrow)
library(lme4)

site <- Sys.getenv("CLIF_SITE", "uchicago")
df <- read_parquet(file.path("output", site, "intermediate_phi", "analysis",
                             "analysis_dataset.parquet"))
df <- df[complete.cases(df[, c("died", "qclif", "sofa", "cci", "age_at_admission")]), ]

# Rescale for interpretability and convergence
df$age_z     <- as.numeric(scale(df$age_at_admission))
df$quality10 <- (df$qclif - mean(unique(df$qclif))) / 10  # per 10-point increase

ctrl <- glmerControl(optimizer = "bobyqa")
m1 <- glmer(died ~ 1 + (1 | unit_name), data = df, family = binomial, control = ctrl)
m2 <- update(m1, . ~ . + sofa + cci + age_z)
m3 <- update(m2, . ~ . + quality10)

anova(m1, m2, m3)
summary(m3)

# Odds ratios
exp(cbind(OR = fixef(m3), confint(m3, method = "Wald", parm = "beta_")))
