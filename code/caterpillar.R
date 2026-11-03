# Caterpillar plot of the ICU random intercepts, one row per ICU.
#
# Sourced by 08_outcome_model.R. Run on its own to redraw from a finished stage 8 without
# refitting:  CLIF_SITE=pooled Rscript code/caterpillar.R
# Reads and writes only output/<site>/final_no_phi/08_outcome_model/.

library(ggplot2)

CAT_INK <- "#0b0b0b"; CAT_GRIDLINE <- "#e1e0d9"; CAT_PAPER <- "#fcfcfb"
CAT_COLOURS <- c("Below average"       = "#93331a",
                 "Not distinguishable" = "#0b0b0b",
                 "Above average"       = "#35649f")

# Labels carry no site or unit name; unit_effects.csv's `icu` column is the key.
# Expects rows already ordered by survival, ascending.
icu_label <- function(n) sprintf("ICU %d", seq_len(n))

plot_caterpillar <- function(cat_tab, surv_avg, path) {
  stopifnot(!is.unsorted(cat_tab$survival))
  cat_tab$label <- factor(icu_label(nrow(cat_tab)), levels = rev(icu_label(nrow(cat_tab))))
  cat_tab$group <- factor(cat_tab$group, levels = names(CAT_COLOURS))
  cp <- ggplot(cat_tab, aes(y = label, x = survival, colour = group)) +
    geom_vline(xintercept = surv_avg, colour = CAT_INK, linewidth = 0.4) +
    geom_linerange(aes(xmin = lo, xmax = hi), linewidth = 0.6) +
    geom_point(size = 1.8) +
    scale_x_continuous(labels = function(x) sprintf("%.0f%%", x)) +
    scale_y_discrete(expand = expansion(add = 0.7)) +
    scale_colour_manual(values = CAT_COLOURS, drop = FALSE, name = NULL) +
    labs(y = NULL, x = "Adjusted in-hospital survival, standard patient") +
    theme_minimal(base_size = 11) +
    theme(legend.position = "bottom",
          legend.background = element_rect(fill = CAT_PAPER, colour = CAT_INK, linewidth = 0.3),
          legend.margin = margin(t = 4, r = 8, b = 4, l = 8),
          legend.text = element_text(colour = CAT_INK, size = 9),
          axis.title = element_text(colour = CAT_INK),
          axis.text = element_text(colour = CAT_INK),
          axis.text.y = element_text(colour = CAT_INK, size = 8, hjust = 1),
          panel.grid.minor = element_blank(),
          panel.grid.major.y = element_blank(),
          panel.grid.major.x = element_line(colour = CAT_GRIDLINE, linewidth = 0.4),
          plot.background = element_rect(fill = CAT_PAPER, colour = NA))
  ggsave(path, cp, width = 6.5, height = 1.2 + 0.14 * nrow(cat_tab), dpi = 300)
}

if (sys.nframe() == 0L) {
  out <- file.path("output", Sys.getenv("CLIF_SITE", "uchicago"), "final_no_phi", "08_outcome_model")
  cat_tab <- read.csv(file.path(out, "unit_effects.csv"))
  cat_tab <- cat_tab[order(cat_tab$survival), ]
  # survival = 100 * plogis(-(eta_fixed + u)), so every row recovers the same eta_fixed.
  eta_fixed <- qlogis(1 - cat_tab$survival / 100) - cat_tab$logodds_death
  stopifnot(diff(range(eta_fixed)) < 1e-6)
  surv_avg <- 100 * plogis(-eta_fixed[1])
  plot_caterpillar(cat_tab, surv_avg, file.path(out, "unit_caterpillar.png"))
  cat(sprintf("wrote %s (%d ICUs, reference %.1f%%)\n",
              file.path(out, "unit_caterpillar.png"), nrow(cat_tab), surv_avg))
}
