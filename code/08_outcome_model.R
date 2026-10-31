# 08_outcome_model.R -- unit ventilator-QI score vs in-hospital mortality
#
# Input:  output/<site>/intermediate_phi/analysis/analysis_dataset.parquet
#         (one row per patient; swap in analysis_dataset_extended.parquet for the
#          SOFA subscores, raw components, n_icus and unit_id)
# Output: output/<site>/final_no_phi/08_outcome_model/

library(arrow)
library(lme4)
library(ggplot2)

site <- Sys.getenv("CLIF_SITE", "uchicago")
out  <- file.path("output", site, "final_no_phi", "08_outcome_model")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

df <- read_parquet(file.path("output", site, "intermediate_phi", "analysis",
                             "analysis_dataset.parquet"))
df <- df[complete.cases(df[, c("died", "qclif", "sofa", "cci", "age_at_admission")]), ]

# Rescale for interpretability and convergence
df$age_z <- as.numeric(scale(df$age_at_admission))
unit_q <- df$qclif[!duplicated(df$unit_name)]
df$qclif_centered <- df$qclif - mean(unit_q)   # per 1-point increase

ctrl <- glmerControl(optimizer = "bobyqa")
m1 <- glmer(died ~ 1 + (1 | unit_name), data = df, family = binomial, control = ctrl)
m2 <- update(m1, . ~ . + sofa + cci + age_z)
m3 <- update(m2, . ~ . + qclif_centered)

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

# Between-unit variance and ICC (latent scale, logistic link)
icc <- data.frame(model = c("m1", "m2", "m3"),
                  between_unit_var = sapply(list(m1, m2, m3),
                                            function(m) as.numeric(VarCorr(m)$unit_name)))
icc$icc <- icc$between_unit_var / (icc$between_unit_var + pi^2 / 3)
write.csv(icc, file.path(out, "icc.csv"), row.names = FALSE)

# Unadjusted, per unit
unit_tab <- aggregate(cbind(n = died, deaths = died) ~ unit_name, df,
                      function(x) c(length(x), sum(x)))
unit_tab <- data.frame(unit_name = unit_tab$unit_name,
                       n = unit_tab$n[, 1], deaths = unit_tab$deaths[, 2])
unit_tab$mortality <- unit_tab$deaths / unit_tab$n
unit_tab$qclif <- df$qclif[match(unit_tab$unit_name, df$unit_name)]
unit_tab <- unit_tab[order(unit_tab$mortality), ]
write.csv(unit_tab, file.path(out, "unit_summary.csv"), row.names = FALSE)


# ---- Figure ------------------------------------------------------------------
# House style: no title, no subtitle -- those belong in captions.md. Type reads
# near-black, gridlines horizontal only, paper ground.

INK <- "#0b0b0b"; GRIDLINE <- "#e1e0d9"; PAPER <- "#fcfcfb"; DIED <- "#8c2f18"

# Ordered by unadjusted mortality; n under each category, because a raw death
# count tracks unit VOLUME as much as unit mortality (N04E has the most deaths
# on the fourth-highest rate).
unit_tab$label <- sprintf("%s\nn = %s", unit_tab$unit_name,
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

ggsave(file.path(out, "deaths_by_unit.png"), p, width = 7.5, height = 4.5, dpi = 300)

writeLines(c(
  "# Figure captions",
  "",
  "**deaths_by_unit.png** — In-hospital deaths by ICU, among adults receiving",
  sprintf("invasive mechanical ventilation at %s (n = %s patients, first qualifying",
          site, format(nrow(df), big.mark = ",")),
  sprintf("IMV hospitalisation per patient; %s deaths, %.1f%% overall). Units are ordered",
          format(sum(df$died), big.mark = ","), 100 * mean(df$died)),
  "by unadjusted in-hospital mortality. Bars are counts and therefore track unit volume",
  "as well as unit mortality;",
  "the denominator is printed under each unit and the unadjusted",
  "mortality above each bar. Death is defined as a discharge category of Expired or",
  "Hospice, or a recorded death within one day of discharge."
), file.path(out, "captions.md"))

cat(sprintf("\nWrote %s\n", out))
