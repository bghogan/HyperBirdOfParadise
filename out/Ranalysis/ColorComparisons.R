#####################################
## name: ColorComparisons.R
## author: Benedict G. Hogan <bhogan@princeton.edu>
##
## Compare the average color for each patch between the hybrid, king, and magnificent bird-of-paradise
## statistical distance for dS, dL, as well as bootstrapping JND/dist to see if the distances are likely to be perceptible
## using https://github.com/rmaia/msdichromatism/blob/master/Rmd/lizardexample.md
## and https://academic.oup.com/beheco/article/29/3/649/4964869
## Maia & White, Comparing colors using visual models, Behavioral Ecology, Volume 29, Issue 3, May/June 2018, Pages 649–659,
####################################

library(pavo)
library(ggplot2)
library(broom)
library(tidyverse)

set.seed(123) # set seed to get repeatable results

# convenience function to grab a distance array from a pavo coldist array
getDistMat <- function(coldist, which = 'dS'){
  out <- coldist2mat(coldist)[[which]]
  return(dist(out))
}

# convenience collect bootcoldist output sensibly
gatherBootOut <- function(bootds, patch, weight_unweight = NA){
  bout <- bootds %>%
    as.data.frame() %>%
    rownames_to_column('contrast') %>% 
    pivot_longer(cols = -contrast, 
                 names_pattern = "(\\w+)\\.(\\w+)", names_to = c(".value", "meanupdown")) %>%
    pivot_longer(c(dS, dL)) %>% 
    pivot_wider(values_from = value, names_from = meanupdown) %>%
    rename(distance_type = name) %>%
    mutate(patch=patch) %>%
    mutate(distance_type=distance_type) %>%
    mutate(weight_unweight = weight_unweight)
  return(bout)
}

##### read in processed spectra data #####

nams <- c(# 'Normal_hyperspectral_images_Whole', # if averaging, makes no sense to look at whole
          'Tilted_hyperspectral_images_Breast',
          'Normal_hyperspectral_images_Belly',
          'Normal_hyperspectral_images_Shoulder',
          'Normal_hyperspectral_images_BackBottom')

ReadSpec = function(x){
  specs <- read.csv(paste0(x, '_spec.csv'))
  specs <- as.rspec(specs, lim = c(300, 700))
  return(specs)
}

list_patches <- list(# 'Whole', 
                     'Breast', 
                     'Belly', 
                     'Shoulder', 
                     'BackBottom')

# generate a list of all samples for each patch
spec_list <- lapply(nams, ReadSpec)
spec_list <- setNames(spec_list, list_patches)

par(mfrow = c(2, 2))
for(i in 1:length(spec_list)){
  plot(spec_list[[i]])
  title(names(spec_list)[i])
}

##### Apply visual models

# some visual parameters for later calulating JNDs
usen<-c(1,1,1,2) # columbia livia (As reported in Vorobyev and Osorio 1998), or c(1,2,2,4) pavo default (Leiothrix lutea value)
useweber<-0.1 # default is 0.1, 0.05 is more conservative (and see Garcia, Rohr, & Dyer 2021)
useweber.achro<-0.1 # default is .1, see above

# apply visual model, here getting abs cone catches
models <- lapply(spec_list, pavo::vismodel,
                 visual = 'avg.v', 
                 relative = F, # this for weighted distances
                 achromatic = 'ch.dc')

# write the sensitivities for the avg_v bird to a file for use in python scripts
mod <- pavo::sensdata(visual = 'avg.v')
write.csv(mod, '../../dat/Sensi/Avg_v.csv', row.names = F)

# get colorspace 
spaces <- lapply(models, colspace)

# get dS and dL - weighted color contrast and luminance contrast
# deltaS <- lapply(models, coldist, achro = TRUE, n=usen, weber=useweber, weber.achro=useweber.achro, qcatch='Qi', noise = "neural")

########## See if perceptual differences are large
# get groups for each model
groups <- lapply(models, function(x) str_extract(rownames(x), '[:alpha:]*(?=_)'))

# get bootstrapped color contrast per group
bootds <- lapply(list_patches, function(x) bootcoldist(models[[x]], groups[[x]], n=usen, weber=useweber, weber.achro=useweber.achro, qcatch='Qi', noise = "neural", boot.n=100))
bootds <- setNames(bootds, list_patches)

# concatenate those results
bootres <- lapply(list_patches, function(x) gatherBootOut(bootds[[x]], x, 'weight'))
bootres <- do.call(rbind, bootres)

# plot those bootstrapped color distances between groups
plo <- bootres %>%
  mutate(patch = if_else(patch == 'Belly', 'Belly/vent', patch)) %>% 
  mutate(patch = if_else(patch == 'BackBottom', 'Back', patch)) %>%
  mutate(patch = factor(patch, levels = c('Breast', 'Belly/vent', 'Shoulder', 'Back'))) %>% 
  mutate(distance_type = factor(distance_type, levels = c('dS', 'dL'))) %>%
  ggplot(aes(x = contrast, y = mean)) +
  geom_point(show.legend = F) +
  geom_errorbar(aes(ymin = lwr, ymax = upr, width = 0), show.legend = F) +
  ylim(c(0,31)) +
  facet_grid(rows = vars(distance_type), cols = vars(patch), switch="y") + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ylab('Color (dS) and brightness (dL) contrast') + xlab('Group comparison') +
  geom_hline(aes(yintercept=3))

plo

ggsave(paste0('../../out/Ranalysis/weighted_color_contrasts', '_weber', useweber, '_aweber', useweber.achro, '_n', paste0(usen, collapse = ''), '.png'), plot = plo, width = 8, height = 4, scale = 1.2)












