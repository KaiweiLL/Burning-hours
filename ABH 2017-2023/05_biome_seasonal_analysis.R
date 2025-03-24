# 08_biome_seasonal_analysis.R
# Analysis of seasonal burning patterns across different biomes
# This script examines how daily burning hours vary by biome and season

# Libraries --------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(tidyr)
library(sf)
library(viridis)

# Load data --------------------------------------------------------------
daily_fp <- read.csv('023_outputs/NAfires_GOES_to daily_combine/GOES_combined_daily_fire_pattern.csv')
daily_fp_active <- daily_fp %>% filter(all.frp.mean.mean > 0)
daily_fp_active <- daily_fp_active %>%
  mutate(howmanyhr = ifelse(howmanyhr == 25, 24, howmanyhr))

# Prepare datasets by season ---------------------------------------------
# Create separate datasets for each season and an "all seasons" dataset
df_all <- daily_fp_active %>%
  mutate(season_grp = "All")

df_s1 <- daily_fp_active %>%
  filter(season == 1) %>%
  mutate(season_grp = "Spring")

df_s2 <- daily_fp_active %>%
  filter(season == 2) %>%
  mutate(season_grp = "Summer")

df_s3 <- daily_fp_active %>%
  filter(season == 3) %>%
  mutate(season_grp = "Fall")

# Combine all datasets
df_combined <- bind_rows(df_all, df_s1, df_s2, df_s3)

# Select key biomes for focused analysis ---------------------------------
key_biomes <- c(
  "Boreal coniferous forest west",
  "Boreal mountain system",
  "Boreal tundra woodland west",
  "Subtropical mountain system",
  "Temperate mountain system west"
)

# Create density plots of daily burning hours by biome and season --------
p <- ggplot(df_combined %>%
              filter(biome %in% key_biomes), 
            aes(x = howmanyhr, color = season_grp, fill = season_grp)) +
  geom_density(alpha = 0.2) +
  facet_wrap(~ biome, scales = "free") +
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

# Calculate summary statistics by biome and season -----------------------
stats_combined <- df_combined %>%
  group_by(biome, season_grp) %>%
  summarise(
    mean_val = mean(howmanyhr, na.rm = TRUE),
    sd_val = sd(howmanyhr, na.rm = TRUE),
    .groups = 'drop'
  )

# Filter to focus on key biomes
stats_combined_filtered <- stats_combined %>%
  filter(biome %in% key_biomes)

# Add vertical offsets for text placement
stats_combined_filtered <- stats_combined_filtered %>%
  mutate(
    v_offset = case_when(
      season_grp == "All" ~ 1.5,
      season_grp == "Spring" ~ 3,
      season_grp == "Summer" ~ 4.5,
      season_grp == "Fall" ~ 6,
      TRUE ~ 1.2
    )
  )

# Create enhanced plot with means and stats ------------------------------
P1 <- p +
  # Add vertical lines at mean values
  geom_vline(
    data = stats_combined_filtered,
    aes(xintercept = mean_val, color = season_grp),
    linetype = "solid",
    size = 0.5
  ) +
  # Add text labels with mean and standard deviation
  geom_text(
    data = stats_combined_filtered,
    aes(
      x = Inf,
      y = Inf,
      label = paste0(round(mean_val, 2), "(", round(sd_val, 2), ")"),
      color = season_grp,
      vjust = v_offset
    ),
    hjust = 1.1,
    size = 2
  ) + 
  scale_x_continuous(breaks = seq(0, 24, 6)) +
  theme(
    text = element_text(size = 7),
    legend.key.size = unit(0.15, 'inches')
  ) +
  xlab("Mean daily active burning hours") +
  ylab("Density")

# Save the plot
# ggsave(plot = P1, "024_plots/fig1_dailyhours_perday.pdf", width = 130, height = 75, units = "mm")

# Create comprehensive table of seasonal burning patterns ----------------
# Count observations by biome
df_biome_counts <- daily_fp_active %>%
  group_by(biome) %>%
  summarise(n_points = n(), .groups = "drop") 

# Filter to biomes with sufficient data
df_biome_counts_filter <- df_biome_counts %>%
  filter(!biome %in% c("Polar", "Water")) %>%
  filter(n_points >= 100)

# Join with statistics
stats_filtered <- stats_combined %>%
  semi_join(df_biome_counts_filter, by = "biome")

# Create wide format table with means and standard deviations
df_wide <- stats_filtered %>%
  mutate(
    mean_sd = paste0(
      round(mean_val, 2),
      "(",
      round(sd_val, 2),
      ")"
    )
  ) %>%
  select(biome, season_grp, mean_sd) %>%
  pivot_wider(
    names_from = season_grp,
    values_from = mean_sd
  ) %>%
  select(biome, All, Spring, Summer, Fall) %>%
  arrange(biome)

# Print the table
print(df_wide)

# Export results
# write.csv(df_wide, "024_plots/biome_seasonal_burning_hours.csv", row.names = FALSE)

# Calculate overall seasonal statistics ----------------------------------
overall_stats <- daily_fp_active %>%
  group_by(season) %>%
  summarise(
    mean_val = mean(howmanyhr, na.rm = TRUE),
    sd_val = sd(howmanyhr, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    season_grp = case_when(
      season == 1 ~ "Spring",
      season == 2 ~ "Summer",
      season == 3 ~ "Fall",
      TRUE ~ "All"
    ),
    biome = "Overall"
  ) %>%
  select(biome, season_grp, mean_val, sd_val)

print(overall_stats)

# Export overall results
# write.csv(overall_stats, "024_plots/overall_seasonal_stats.csv", row.names = FALSE)

# Analyze regional variation in seasonal burning patterns ----------------
# Compare western boreal vs. other regions
western_boreal <- daily_fp_active %>%
  filter(biome %in% c("Boreal coniferous forest west", "Boreal tundra woodland west")) %>%
  group_by(season) %>%
  summarise(
    mean_hrs = mean(howmanyhr, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    region = "Western Boreal",
    season_name = case_when(
      season == 1 ~ "Spring",
      season == 2 ~ "Summer",
      season == 3 ~ "Fall",
      TRUE ~ "Unknown"
    )
  )

mountain_systems <- daily_fp_active %>%
  filter(biome %in% c("Temperate mountain system west", "Subtropical mountain system")) %>%
  group_by(season) %>%
  summarise(
    mean_hrs = mean(howmanyhr, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    region = "Mountain Systems",
    season_name = case_when(
      season == 1 ~ "Spring",
      season == 2 ~ "Summer",
      season == 3 ~ "Fall",
      TRUE ~ "Unknown"
    )
  )

# Combine and compare
regional_comparison <- bind_rows(western_boreal, mountain_systems)

# Create bar plot comparing seasonal patterns by region
regional_plot <- ggplot(regional_comparison, aes(x = season_name, y = mean_hrs, fill = region)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  geom_text(aes(label = round(mean_hrs, 1)), 
            position = position_dodge(width = 0.7),
            vjust = -0.5, 
            size = 2.5) +
  scale_fill_manual(values = c("Western Boreal" = "#3182BD", "Mountain Systems" = "#E6550D")) +
  labs(
    x = "Season",
    y = "Mean Daily Burning Hours",
    title = "Seasonal Burning Patterns by Region"
  ) +
  theme_minimal() +
  theme(
    text = element_text(size = 7),
    legend.position = "top",
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank()
  )

# Save the regional comparison plot
# ggsave(plot = regional_plot, "024_plots/regional_seasonal_comparison.pdf", 
#        width = 120, height = 80, units = "mm")