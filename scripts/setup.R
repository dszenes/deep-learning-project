# load the required packages
packages <- c(
  "here", # for the project's organization
  "tidyverse", # for wrangling
  "keras", "cloudml", #DL R packages
  "knitr", "kableExtra", "bookdown", "rmarkdown", "DT", # for the report
  "ggthemes" #for visualization
)
purrr::walk(packages, library, character.only = TRUE)


# general options
options(
  digits = 3,
  str = strOptions(strict.width = "cut"),
  width = 69,
  tibble.width = 69,
  cli.unicode = FALSE
)


# knitr options
opts_chunk$set(
  comment = "#>",
  collapse = TRUE,
  # cache = TRUE,
  fig.retina = 0.8, # figures are either vectors or 300 dpi diagrams
  dpi = 300,
  out.width = "70%",
  fig.align = "center",
  fig.width = 6,
  fig.asp = 0.618,
  fig.show = "hold",
  message = FALSE,
  echo = FALSE
)
