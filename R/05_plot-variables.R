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
mid_or <- names(which(
  summary(logreg)$coefficients[-1, 1] >= 2**-9
  & summary(logreg)$coefficients[-1, 1] <= 2**9
  ))
high_p
mid_or
nrow(summary(logreg)$coefficients)

plot_data <- tidy(logreg)
plot_data

rm_terms <- tidy(logreg) %>%
  mutate(or = exp(estimate)) %>%
  filter(or < 2**5 | or > 2**-5) %>% 
  select(term)

p <- plot_model(
  logreg,
  sort.est = TRUE,
  title = "",
  rm.terms = c(high_p, "C5700A", "C17440T", "C11563T", "X", "G28881A", "C3373A","C22432T", "G26233T", "C18877T"),
  vline.color = "grey",
  colors = c("darkgreen", "firebrick"),
  p.shape = TRUE
) + theme_bw()

p + scale_y_continuous(
  trans = "log2", 
  limits = c(0.001, 1),
  breaks=c(2^-7, 2^-5, 2^-3, 2^-1),
  label=c(2^-7, 2^-5, 2^-3, 2^-1),
  # labels=c(0.01, 0.1, 10, 1000)
  ) + aes(shape = group) + theme(legend.position = "none")

ggsave("plots/all_coefplot.png", dpi = 300)
