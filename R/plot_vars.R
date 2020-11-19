library(tidyverse)
library(tidymodels)
library(sjPlot)
library(ggplot2)
library(scales)
library(here)
library(broom)

df <- read.csv(here(
    "data/2020-10-21_logistic-regression-lasso-selected-features.csv"
    )) %>%
  mutate(y = as.factor(y))

logreg = glm("y ~ .", data = df, family = "binomial")

# fit <- logistic_reg(mode = "classification") %>%
#   set_engine("glm", family = binomial(link = "logit")) %>%
#   fit(formula(y ~ .), data=df)

saveRDS(logreg, file = "models/2020-10-21_logistic-regression-model.rds")

logreg = readRDS(file = here("models/2020-10-21_logistic-regression-model.rds"))

high_p <- names(which(summary(logreg)$coefficients[-1, 4] >= 0.05))

p <- plot_model(
  logreg,
  sort.est = TRUE,
  title = "",
  rm.terms = high_p,
  vline.color = "grey",
  colors = c("darkgreen", "firebrick"),
  p.shape = TRUE
) + theme_bw()

p + scale_y_continuous(
  trans = "log2", 
  # limits = c(0.01, 1000),
  breaks=c(2^-5, 2^-3, 2^-1, 2^2, 2^4, 2^6),
  label=c(2^-5, 2^-3, 2^-1, 2^2, 2^4, 2^6),
  # labels=c(0.01, 0.1, 10, 1000)
  ) + aes(shape = group) + theme(legend.position = "none")

ggsave("plots/all_coefplot.png", dpi = 300)
