# 05_burning_duration_intensity_analysis.R
# Analysis of the relationship between burning duration and fire intensity
# This script examines how active burning hours relate to FRP and fire progression

# Libraries --------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(sf)
library(scales)

# Load data --------------------------------------------------------------
daily_fp <- read.csv('023_outputs/NAfires_GOES_to daily_combine/GOES_combined_daily_fire_pattern.csv')
daily_fp_active <- daily_fp %>% filter(all.frp.mean.mean > 0)
daily_fp_active <- daily_fp_active %>%
  mutate(howmanyhr = ifelse(howmanyhr == 25, 24, howmanyhr))

# Analysis of burning duration vs. intensity -----------------------------
# Calculate statistics for hours 23 vs 24 to examine effect of complete burning cycle
stats_comparison <- daily_fp_active %>%
  filter(howmanyhr %in% c(23, 24)) %>%
  group_by(howmanyhr) %>%
  summarise(
    mean_frp = mean(all.frp.max.max, na.rm = TRUE),
    median_frp = median(all.frp.max.max, na.rm = TRUE),
    n = n()
  )

# Calculate percentage increase in FRP from 23h to 24h
increase_percentage <- (stats_comparison$mean_frp[2] - stats_comparison$mean_frp[1]) / 
  stats_comparison$mean_frp[1] * 100
print(paste("Increase in mean maximum FRP from 23h to 24h:", 
            round(increase_percentage, 1), "%"))

# Create plot of burning hours vs. maximum FRP
duration_intensity_plot <- ggplot(daily_fp_active, 
                                  aes(x = as.numeric(as.character(howmanyhr)), 
                                      y = all.frp.max.max)) +
  # Add jittered points for all observations
  geom_jitter(width = 0.3, alpha = 0.1, color = "black", size = 0.5) +
  
  # Add boxplots without outliers
  geom_boxplot(aes(group = howmanyhr), outlier.shape = NA, fill = NA, size = 0.25,
               color = "#DAA520", alpha = 0.6) +
  
  # Add mean line and points
  stat_summary(fun = mean, geom = "line", color = "#8B0000", linewidth = 0.25) +
  stat_summary(fun = mean, geom = "point", color = "#8B0000", size = 1) +
  
  # Configure axis scales
  scale_x_continuous(breaks = seq(2, 24, 2)) +
  scale_y_continuous(
    labels = comma,
    breaks = c(0, 2000, 4000, 6000, 8000),
    limits = c(0, 8000)
  ) +
  
  # Labels
  labs(
    x = "Daily active burning hours",
    y = "Daily Maximum FRP (MW)"
  ) +
  
  # Theme settings
  theme_minimal() +
  theme(
    text = element_text(size = 7),
    legend.key.size = unit(0.15, 'inches'),
    plot.margin = margin(1, 1, 1, 1, "cm"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.line = element_line(color = "black", size = 0.25),
    axis.ticks = element_line(color = "black", size = 0.25)
  )

# Save the plot
# ggsave(plot = duration_intensity_plot, "024_plots/fig2_duration_intensity.pdf", 
#        width = 3.5, height = 2.8, units = "in")

# Fire event level analysis ----------------------------------------------
# Summarize data at the fire event level
firelevel <- daily_fp_active %>% 
  group_by(year, lat, long, POLY_HA) %>% 
  summarise(
    firesize = POLY_HA[1],
    country = country[1],
    activedays = n(),
    activehours = sum(as.numeric(howmanyhr)),
    activedayhrs = sum(dorn_0_count),
    activenighthrs = sum(dorn_1_count),
    maxhrfrp_headfire = max(all.frp.max.max),
    diff_hours_maxmax_first = diff_hours_maxmax_first[1],
    
    # Timing metrics
    first_active_day = min(day),
    max_frp_day = day[which.max(all.frp.max.max)],
    first_above_90_percent_max = day[which(all.frp.max.max > 0.9 * max(all.frp.max.max))[1]],
    days_max_frp_to_first_active = as.numeric(max_frp_day - first_active_day),
    days_first_above_90_to_first_active = as.numeric(first_above_90_percent_max - first_active_day)
  ) %>%
  # Add categorical variables for analysis
  mutate(
    firesize_category = factor(case_when(
      log(firesize) < 7 ~ "<7",
      log(firesize) >= 7 & log(firesize) < 9 ~ "7-9",
      log(firesize) >= 9 & log(firesize) < 11 ~ "9-11",
      log(firesize) >= 11 ~ ">11"
    ), levels = c("<7", "7-9", "9-11", ">11")),
    activehours_category = factor(case_when(
      activehours < 100 ~ "0-100",
      activehours >= 100 & activehours < 300 ~ "100-300",
      activehours >= 300 ~ "