library(tidyverse)
library(tidymodels)
library(sjPlot)
library(ggplot2)

df <- read.csv("data/top_btm_df.csv") %>% select(-pid )

logreg = glm("y ~ .", data = df, family = "binomial")

df2 <- df %>%
  select(
    which(summary(logreg)$coefficients[-1, 4] < .05),
    TGCACCTCATGGTCATGTTATGGTTGAGCTGGTA499T,
    C15324T,
    C28253T
  )  %>% rename(
  Asia=cat_region_1,
  Europe=cat_region_2,
  "North America"=cat_region_3,
  "South America"=cat_region_5,
  Age=covv_patient_age,
  )

logreg2 = glm("y ~ .", data = df2, family = "binomial")

plot_model(
  logreg,
  sort.est = TRUE,
  title = "",
  rm.terms = c(to_drop, "Asia", "North America", "South America", "Age", "Europe"),
  vline.color = "grey",
  axis.lim=c(0.01, 1200)
) + theme_bw()

ggsave("plots/coefplot.png", dpi = 300)