# 04_daily_fire_pattern_statistics.R
# Analysis of daily fire patterns and active burning hours (ABH) statistics
# This script analyzes basic statistics of active burning hours and regional distributions

# Libraries --------------------------------------------------------------
library(rgdal)
library(raster)
library(sp)   
library(dplyr)
library(tidyr)
library(ggplot2)
library(reshape2)
library(ggsci)
library(lutz)
library(suncalc)
library(gtools)
library(foreach)
library(doParallel)
library(parallel)
library(tcltk)
library(doSNOW)
library(caret)
library(sf)

# Load data --------------------------------------------------------------
daily_fp <- read.csv('023_outputs/NAfires_GOES_to daily_combine/GOES_combined_daily_fire_pattern.csv')
daily_fp_active <- daily_fp %>% filter(all.frp.mean.mean > 0)
daily_fp_active <- daily_fp_active %>%
  mutate(howmanyhr = ifelse(howmanyhr == 25, 24, howmanyhr))

# Basic statistics -------------------------------------------------------
# Calculate total active hours
hmh = sum(daily_fp_active$howmanyhr)
print(paste("Total active hours:", hmh))

# Visualize distribution of active hours
ggplot(data = daily_fp_active, aes(x = howmanyhr)) + 
  geom_histogram() +
  labs(x = "Active Burning Hours", y = "Count") +
  theme_minimal()

# Regional analysis ------------------------------------------------------
# Group boreal biomes together
daily_fp_active <- daily_fp_active %>%
  mutate(
    biome_grouped = case_when(
      grepl("boreal", biome, ignore.case = TRUE) ~ "boreal", 
      TRUE ~ biome 
    )
  )

# Calculate sum of active hours by biome
fp_biome = daily_fp_active %>% 
  group_by(biome_grouped) %>% 
  summarise(activehrsum = sum(howmanyhr))

# Visualize active hours by biome
ggplot(data = fp_biome, aes(x = biome_grouped, y = activehrsum)) + 
  geom_col() +
  labs(x = "Biome", y = "Total Active Hours") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Calculate cumulative percentages for key biomes
daily_fp_active_cumsum <- daily_fp_active %>%
  filter(biome_grouped %in% c("Temperate mountain system west", "boreal", "Subtropical mountain system")) %>%
  group_by(biome_grouped) %>%
  arrange(howmanyhr) %>%
  mutate(cumulative_hr = cumsum(howmanyhr), 
         cumulative_percentage = cumulative_hr / sum(howmanyhr) * 100) %>%
  ungroup()

# Calculate total hours and FRP by biome
total_hours_by_biome <- daily_fp_active_cumsum %>%
  group_by(biome_grouped) %>%
  summarise(total_hours = sum(howmanyhr),
            frp.mean.totalperday = sum(all.frp.mean.mean*howmanyhr),
            frp.max.totalperday = sum(all.frp.max.mean*howmanyhr)) %>%
  ungroup()

print(total_hours_by_biome)

# Analyze distribution of active hours across time periods ---------------
daily_fp_active_grouped <- daily_fp_active_cumsum %>%
  mutate(hour_group = case_when(
    howmanyhr >= 1 & howmanyhr <= 6 ~ "1-6h",
    howmanyhr >= 7 & howmanyhr <= 12 ~ "7-12h",
    howmanyhr >= 13 & howmanyhr <= 18 ~ "13-18h",
    howmanyhr >= 19 & howmanyhr <= 23 ~ "19-23h",
    howmanyhr == 24 ~ "24h"
  )) %>%
  mutate(hour_group = factor(hour_group, 
                             levels = c("24h", "19-23h", "13-18h", "7-12h", "1-6h")))

# Calculate proportions within each hour group by biome
biome_hour_proportions <- daily_fp_active_grouped %>%
  group_by(biome_grouped, hour_group) %>%
  summarise(count = n()) %>%
  group_by(biome_grouped) %>%
  mutate(proportion = count/sum(count) * 100) %>%
  ungroup()

# Visualize proportions of active hour groups by biome
ggplot(biome_hour_proportions, aes(x = hour_group, y = proportion, fill = biome_grouped)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(
    x = "Active Hour Group",
    y = "Percentage (%)",
    fill = "Biome") +
  theme_minimal() +
  theme(text = element_text(size = 7))

# Export results
# write.table(biome_hour_proportions, file = '026_plots/hour_proportion.csv', sep = ',', row.names = FALSE)

# Nighttime burning analysis ---------------------------------------------
burn2night = daily_fp_active %>% 
  filter(night.frp.max.max > 0) %>% 
  mutate(max_frp_day_or_night = night.frp.max.max - day.frp.max.max)

howmanymax_at_night = burn2night %>% filter(max_frp_day_or_night > 0)
print(paste("Percentage of fires with maximum FRP at night:", 
            round(nrow(howmanymax_at_night) / nrow(burn2night) * 100, 2), "%"))