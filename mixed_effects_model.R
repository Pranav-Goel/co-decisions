# Package names
packages <- c("lme4", "tidyverse")

# Install packages not yet installed
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages], repos = "https://cloud.r-project.org")
}

# Packages loading
invisible(lapply(packages, library, character.only = TRUE))

# library(lme4)
# library(tidyverse)

args = commandArgs(trailingOnly=TRUE)

decisions_data = read.csv(args[1])
pairwise_data = read.csv(args[2])

v <- rep(ncol(decisions_data) - 1, nrow(pairwise_data)) #setting weights = number of decisions

colnames <- colnames(pairwise_data)
X <- colnames[grepl("Same_|_Sim",colnames)]
f <- as.formula(paste("CoDecision_Agreement_Rate ~ ",
                      paste(X, sep = ' + ', collapse = ' + '),
                      " + (1|Actor_1) + (1|Actor_2)"))

lme = glmer(f, data=pairwise_data, family = binomial(link = "logit"), weights = v)

sink(args[3])
print(summary(lme))
sink()