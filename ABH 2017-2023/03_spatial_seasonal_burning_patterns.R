# 06_spatial_seasonal_burning_patterns.R
# Analysis of spatial patterns and seasonal variations in active burning hours
# This script examines regional and seasonal differences in daily burning duration

# Libraries --------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(sf)
library(ggsci)

# Load data --------------------------------------------------------------
daily_fp <- read.csv('023_outputs/NAfires_GOES_to daily_combine/GOES_combined_daily_fire_pattern.csv')
daily_fp_active <- daily_fp %>% filter(all.frp.mean.mean > 0)
daily_fp_active <- daily_fp_active %>%
  mutate(howmanyhr = ifelse(howmanyhr == 25, 24, howmanyhr))

# Calculate average daily hours by fire ----------------------------------
# Group by individual fires and calculate average daily burning hours
avg.dailyhrs.perfire <- daily_fp_active %>% 
  group_by(lat, long, year, POLY_HA, biome) %>% 
  summarise(
    avg.dailyhrs = mean(howmanyhr),
    number_firedays = n()
  )

# Calculate overall average daily hours across all fires
mean_daily_hours <- sum(daily_fp_active$howmanyhr) / nrow(daily_fp_active)
print(paste("Overall average daily burning hours:", round(mean_daily_hours, 2)))

# Load shapefiles for mapping
shp.biome <- st_read('D:/000_collections/010_Nighttime Burning/011_Data/013_Biome_wwf2017/fire_biomes_continent_updated2/fire_biomes_USCA.shp')
shp.biome <- shp.biome %>% filter(gez_name != 'Water', gez_name != "Tropical dry forest")
shp.us <- st_read('D:/000_collections/010_Nighttime Burning/011_Data/013_Biome_wwf2017/cb_2018_us_state_20m/cb_2018_us_state_20m_CONUS_AK.shp')
shp.ca <- st_read('D:/000_collections/010_Nighttime Burning/011_Data/013_Biome_wwf2017/lpr_000b16a_e/lpr_000b16a_e.shp')

# Convert fire locations to sf objects
loc.sf <- st_as_sf(avg.dailyhrs.perfire, coords = c("long", "lat"), 
                   crs = 4326, agr = "constant")

# Create map of average daily hours by location
daily_hours_map <- ggplot() +
  geom_sf(data = shp.biome, aes(fill = gez_name), alpha = 0.7, size = 0.25, color = 'gray') +
  geom_sf(data = loc.sf %>% filter(number_firedays >= 3), aes(color = avg.dailyhrs), 
          size = 0.5, alpha = 0.8) +
  scale_color_viridis_c(
    option = "inferno",
    direction = -1,
    name = "Average Daily\nBurning Hours"
  ) +
  scale_fill_manual(values = c('slateblue4','slateblue3','slateblue1','gray95',
                               "olivedrab4", "olivedrab3", "olivedrab2","olivedrab1",'lightgreen',
                               'lightgoldenrod4','lightgoldenrod3','lightgoldenrod2',
                               'lightgoldenrod1','moccasin','palevioletred4')) +
  coord_sf(xlim = c(-170, -50), ylim = c(25, 75)) +
  theme_classic() +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.line.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    axis.line.y = element_blank(),
    axis.ticks.y = element_blank(),
    strip.text.x = element_text(size = 7, color = "black", face = "plain"),
    plot.title = element_text(color = "black", size = 7, face = "bold", vjust = -1),
    text = element_text(size = 7),
    legend.key.size = unit(0.15, 'inches')
  ) +
  coord_sf(crs = 3979, expand = F)

# Save the map
# ggsave(plot = daily_hours_map, "024_plots/sup_fig_spatial_dailyhours.pdf", 
#        width = 180, height = 180, units = "mm")

# Calculate average daily hours by biome
avg.dailyhrs.perday <- daily_fp_active %>% 
  group_by(biome) %>% 
  summarise(avg.dailyhrs = mean(howmanyhr))

print(avg.dailyhrs.perday)

# Analyze seasonal patterns in daily burning hours -----------------------
# Prepare data with season groups
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

# Combine datasets
df_combined <- bind_rows(df_all, df_s1, df_s2, df_s3)

# Create density plots of daily hours by biome and season
key_biomes <- c(
  "Boreal coniferous forest west",
  "Boreal mountain system",
  "Boreal tundra woodland west",
  "Subtropical mountain system",
  "Temperate mountain system west"
)

seasonal_density_plot <- ggplot(
  df_combined %>% filter(biome %in% key_biomes), 
  aes(x = howmanyhr, color = season_grp, fill = season_grp)
) +
  geom_density(alpha = 0.2) +
  facet_wrap(~ biome, scales = "free") +
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

# Calculate statistics by biome and season
stats_combined <- df_combined %>%
  group_by(biome, season_grp) %>%
  summarise(
    mean_val = mean(howmanyhr, na.rm = TRUE),
    sd_val = sd(howmanyhr, na.rm = TRUE),
    .groups = 'drop'
  )

# Filter to key biomes
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

# Create enhanced density plot with mean values
enhanced_density_plot <- seasonal_density_plot +
  geom_vline(
    data = stats_combined_filtered,
    aes(xintercept = mean_val, color = season_grp),
    linetype = "solid",
    size = 0.5
  ) +
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
# ggsave(plot = enhanced_density_plot, "024_plots/fig1_dailyhours_perday.pdf", 
#        width = 130, height = 75, units = "mm")

# Create table of seasonal burning hours by biome ------------------------
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

# Create wide format table
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
  tidyr::pivot_wider(
    names_from = season_grp,
    values_from = mean_sd
  ) %>%
  select(biome, All, Spring, Summer, Fall) %>%
  arrange(biome)

print(df_wide)

# Calculate overall seasonal statistics
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

# Write results to file
# write.csv(df_wide, "024_plots/seasonal_burning_hours_by_biome.csv", row.names = FALSE)
# write.csv(overall_stats, "024_plots/overall_seasonal_burning_hours.csv", row.names = FALSE)