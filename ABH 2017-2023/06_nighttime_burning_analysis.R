# 09_nighttime_burning_analysis.R
# Analysis of nighttime burning patterns and fire intensity timing
# This script examines nighttime burning characteristics and temporal fire development

# Libraries --------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(scales)
library(sf)
library(zoo)  # For rolling averages if needed

# Load data --------------------------------------------------------------
daily_fp <- read.csv('023_outputs/NAfires_GOES_to daily_combine/GOES_combined_daily_fire_pattern.csv')
daily_fp_active <- daily_fp %>% filter(all.frp.mean.mean > 0)
daily_fp_active <- daily_fp_active %>%
  mutate(howmanyhr = ifelse(howmanyhr == 25, 24, howmanyhr))

# Analyze nighttime burning patterns -------------------------------------
# Identify fires that burn into the night
burn2night <- daily_fp_active %>% 
  filter(night.frp.max.max > 0) %>% 
  mutate(max_frp_day_or_night = night.frp.max.max - day.frp.max.max)

# Count fires where maximum FRP occurs at night
howmanymax_at_night <- burn2night %>% filter(max_frp_day_or_night > 0)

# Calculate proportion of fires with maximum FRP at night
night_max_proportion <- nrow(howmanymax_at_night) / nrow(burn2night) * 100
print(paste("Percentage of nighttime-burning fires with maximum FRP at night:", 
            round(night_max_proportion, 1), "%"))

# Analyze intensity of day vs. night burning -----------------------------
# Get summary statistics for day vs. night FRP values
night_day_stats <- burn2night %>%
  summarize(
    mean_day_frp = mean(day.frp.max.max, na.rm = TRUE),
    median_day_frp = median(day.frp.max.max, na.rm = TRUE),
    sd_day_frp = sd(day.frp.max.max, na.rm = TRUE),
    
    mean_night_frp = mean(night.frp.max.max, na.rm = TRUE),
    median_night_frp = median(night.frp.max.max, na.rm = TRUE),
    sd_night_frp = sd(night.frp.max.max, na.rm = TRUE),
    
    day_night_ratio = mean_day_frp / mean_night_frp,
    median_ratio = median_day_frp / median_night_frp
  )

print(night_day_stats)

# Visualize day vs. night FRP distributions
day_night_plot <- ggplot(burn2night, aes(x = factor(1))) +
  geom_boxplot(aes(y = day.frp.max.max, fill = "Day"), 
               alpha = 0.7, width = 0.4, position = position_nudge(x = -0.2)) +
  geom_boxplot(aes(y = night.frp.max.max, fill = "Night"), 
               alpha = 0.7, width = 0.4, position = position_nudge(x = 0.2)) +
  scale_fill_manual(values = c("Day" = "#FF614E", "Night" = "#2E2E2E"), 
                    name = "Time of Day") +
  scale_y_continuous(labels = comma_format()) +
  labs(
    x = "",
    y = "Maximum FRP (MW)",
    title = "Comparison of Day vs. Night Maximum FRP"
  ) +
  theme_minimal() +
  theme(
    text = element_text(size = 7),
    legend.position = "top",
    panel.grid.major.x = element_blank(),
    plot.title = element_text(hjust = 0.5, size = 8),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  )

# Save the plot
# ggsave(plot = day_night_plot, "024_plots/day_night_frp_comparison.pdf", 
#        width = 80, height = 100, units = "mm")

# Analyze timing of maximum FRP within fire events -----------------------
# Create a fire event level dataset
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
    days_max_frp_to_first_active = as.numeric(max_frp_day - first_active_day)
  )

# Convert to hours for better temporal resolution
firelevel_hrs <- firelevel %>%
  mutate(hours_to_max_frp = diff_hours_maxmax_first)

# Calculate proportion of fires reaching maximum intensity within different timeframes
within_24hr <- sum(firelevel_hrs$hours_to_max_frp <= 24, na.rm = TRUE)
within_48hr <- sum(firelevel_hrs$hours_to_max_frp <= 48, na.rm = TRUE)
total_fires <- sum(!is.na(firelevel_hrs$hours_to_max_frp))

pct_24hr <- within_24hr / total_fires * 100
pct_48hr <- within_48hr / total_fires * 100

print(paste("Proportion of fires reaching maximum FRP within 24 hours:", 
            round(pct_24hr, 1), "%"))
print(paste("Proportion of fires reaching maximum FRP within 48 hours:", 
            round(pct_48hr, 1), "%"))

# Create histogram of time to maximum FRP
max_frp_timing_plot <- ggplot(firelevel_hrs, aes(x = hours_to_max_frp)) +
  geom_histogram(
    breaks = seq(0, 240, 24),
    fill = "black", 
    color = "white", 
    alpha = 0.8,
    width = 20
  ) +
  scale_x_continuous(
    breaks = seq(0, 240, 48),
    labels = seq(0, 240, 48),
    limits = c(0, 240)
  ) +
  labs(
    x = "Hours between maximum FRP and first active fire detection",
    y = "Frequency",
    title = "Time to reach maximum FRP"
  ) +
  theme_minimal() +
  theme(
    text = element_text(size = 7),
    legend.key.size = unit(0.15, 'inches'),
    plot.margin = margin(1, 1, 1, 1, "cm"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.line = element_line(color = "black", size = 0.25),
    axis.ticks = element_line(color = "black", size = 0.25),
    plot.title = element_text(hjust = 0.5, size = 8)
  )

# Save the plot
# ggsave(plot = max_frp_timing_plot, "024_plots/hours_to_max_frp.pdf", 
#        width = 2.6, height = 2.8, units = "in")

# Create time categories for fires
firelevel_categorized <- firelevel_hrs %>%
  mutate(
    time_category = case_when(
      hours_to_max_frp <= 24 ~ "0-24h",
      hours_to_max_frp <= 48 ~ "24-48h",
      hours_to_max_frp <= 72 ~ "48-72h",
      hours_to_max_frp <= 96 ~ "72-96h",
      TRUE ~ ">96h"
    ),
    time_category = factor(time_category, 
                           levels = c("0-24h", "24-48h", "48-72h", "72-96h", ">96h"))
  )

# Calculate proportions by time category
time_category_proportions <- firelevel_categorized %>%
  group_by(time_category) %>%
  summarise(
    count = n(),
    percentage = count / sum(count) * 100
  )

print(time_category_proportions)

# Create plot showing distribution by time category
time_category_plot <- ggplot(time_category_proportions, 
                             aes(x = time_category, y = percentage, fill = time_category)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), 
            position = position_stack(vjust = 0.5),
            color = "white", 
            size = 2.5) +
  scale_fill_viridis_d(option = "D", direction = -1) +
  labs(
    x = "Time to Maximum FRP",
    y = "Percentage of Fires",
    title = "Distribution of Fires by Time to Maximum FRP"
  ) +
  theme_minimal() +
  theme(
    text = element_text(size = 7),
    legend.position = "none",
    panel.grid.major.x = element_blank(),
    plot.title = element_text(hjust = 0.5, size = 8)
  )

# Save the plot
# ggsave(plot = time_category_plot, "024_plots/time_to_max_frp_distribution.pdf", 
#        width = 120, height = 80, units = "mm")

# Analyze size relationship with time to maximum FRP ---------------------
size_time_plot <- ggplot(firelevel_categorized, aes(x = time_category, y = log10(firesize))) +
  geom_boxplot(fill = "lightblue", alpha = 0.7) +
  labs(
    x = "Time to Maximum FRP",
    y = "Log10(Fire Size) (ha)",
    title = "Relationship Between Fire Size and Time to Maximum FRP"
  ) +
  theme_minimal() +
  theme(
    text = element_text(size = 7),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    plot.title = element_text(hjust = 0.5, size = 8)
  )

# Save the plot
# ggsave(plot = size_time_plot, "024_plots/size_vs_time_to_max_frp.pdf", 
#        width = 120, height = 80, units = "mm")