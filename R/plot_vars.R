library(tidyverse)
library(tidymodels)
library(sjPlot)
library(ggplot2)
library(scales)

df <- read.csv("data/top_btm_df.csv") %>% select(-pid )

logreg = glm("y ~ .", data = df, family = "binomial")

df2 <- df %>%
  select(
    which(summary(logreg)$coefficients[-1, 4] < .05),
    TGCACCTCATGGTCATGTTATGGTTGAGCTGGTA499T,
    C15324T,
    C28253T,
    y
  )  %>% rename(
  Asia=cat_region_1,
  Europe=cat_region_2,
  "North America"=cat_region_3,
  "South America"=cat_region_5,
  Age=covv_patient_age,
  )

logreg2 = glm("y ~ .", data = df2, family = "binomial")

p <- plot_model(
  logreg2,
  sort.est = TRUE,
  title = "",
  rm.terms = c(
    "Asia",
    # "North America",
    # "South America",
    # "Age",
    # "Europe",
    "C28253T",
    "G29711T"
    ),
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
