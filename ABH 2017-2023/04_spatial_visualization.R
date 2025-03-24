# 07_spatial_visualization.R
# Spatial visualization of active burning hours across North America (2017-2023)
# This script creates maps of fire locations and active burning hours by region

# Libraries --------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(sf)
library(ggsci)
library(scales)

# Load data --------------------------------------------------------------
daily_fp <- read.csv('023_outputs/NAfires_GOES_to daily_combine/GOES_combined_daily_fire_pattern.csv')
daily_fp_active <- daily_fp %>% filter(all.frp.mean.mean > 0)
daily_fp_active <- daily_fp_active %>%
  mutate(howmanyhr = ifelse(howmanyhr == 25, 24, howmanyhr))

# Load shapefiles for mapping --------------------------------------------
shp.biome <- st_read('D:/000_collections/010_Nighttime Burning/011_Data/013_Biome_wwf2017/fire_biomes_continent_updated2/fire_biomes_USCA.shp')
shp.biome <- shp.biome %>% filter(gez_name != 'Water', gez_name != "Tropical dry forest")
shp.us <- st_read('D:/000_collections/010_Nighttime Burning/011_Data/013_Biome_wwf2017/cb_2018_us_state_20m/cb_2018_us_state_20m_CONUS_AK.shp')
shp.ca <- st_read('D:/000_collections/010_Nighttime Burning/011_Data/013_Biome_wwf2017/lpr_000b16a_e/lpr_000b16a_e.shp')

# Convert point locations to sf objects ----------------------------------
loc.df <- daily_fp_active %>% dplyr::select(lat, long)
loc.sf <- st_as_sf(loc.df, coords = c("long", "lat"), 
                   crs = 4326, agr = "constant")

# Select top 200 fires by active hours, 100 each from USA and Canada -----
top100can <- daily_fp_active %>% 
  filter(country == "CAN") %>%
  group_by(year, long, lat) %>%
  summarise(lat = lat[1], long = long[1], biome = biome[1], ba = POLY_HA[1], sumhr = sum(howmanyhr)) %>%
  ungroup() %>%
  arrange(desc(sumhr)) %>%
  slice(1:200)

top100usa <- daily_fp_active %>% 
  filter(country == "USA") %>%
  group_by(year, long, lat) %>%
  summarise(lat = lat[1], long = long[1], biome = biome[1], ba = POLY_HA[1], sumhr = sum(howmanyhr)) %>%
  ungroup() %>%
  arrange(desc(sumhr)) %>%
  slice(1:200)

# Combine data and convert to sf object
top200 <- bind_rows(
  mutate(top100can, country = "CAN"),
  mutate(top100usa, country = "USA")
) %>%
  arrange(sumhr)

top200.sf <- st_as_sf(top200, coords = c("long", "lat"), 
                      crs = 4326, agr = "constant")

# Create map with top 200 fires by active hours --------------------------
f1.top200 <- ggplot() + 
  geom_sf(data = shp.ca, fill = NA, alpha = 0.2, size = 0.25, color = 'black') +
  geom_sf(data = shp.us, fill = NA, alpha = 0.2, size = 0.25, color = 'black') +
  geom_sf(data = loc.sf, color = 'gray', size = 0.1, alpha = 0.8) +
  geom_sf(data = top200.sf, 
          aes(size = sumhr,
              color = sumhr), 
          alpha = 0.8) +
  scale_size_continuous(range = c(0.6, 3)) + 
  scale_color_viridis_c(
    option = "inferno",  
    direction = -1,       
    name = "Active hours",
    breaks = pretty_breaks(4)
  ) +
  coord_sf(xlim = c(-170, -50),  
           ylim = c(25, 75)) +    
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

# Save the plot
# ggsave(plot = f1.top200, "024_plots/fig1_top200.pdf", width = 180, height = 180, units = "mm")

# Create map with top 200 fires by FRP sum -------------------------------
# Select top 200 fires by FRP × hours
top100can.frp <- daily_fp_active %>% 
  filter(country == "CAN") %>%
  group_by(year, long, lat) %>%
  summarise(lat = lat[1], long = long[1], biome = biome[1], ba = POLY_HA[1], 
            sumfrp = sum(all.frp.max.mean * howmanyhr)) %>%
  ungroup() %>%
  arrange(desc(sumfrp)) %>%
  slice(1:200)

top100usa.frp <- daily_fp_active %>% 
  filter(country == "USA") %>%
  group_by(year, long, lat) %>%
  summarise(lat = lat[1], long = long[1], biome = biome[1], ba = POLY_HA[1], 
            sumfrp = sum(all.frp.max.mean * howmanyhr)) %>%
  ungroup() %>%
  arrange(desc(sumfrp)) %>%
  slice(1:200)

# Combine data and convert to sf object
top200.frp <- bind_rows(
  mutate(top100can.frp, country = "CAN"),
  mutate(top100usa.frp, country = "USA")
) %>%
  arrange(sumfrp)

top200.frp.sf <- st_as_sf(top200.frp, coords = c("long", "lat"), 
                          crs = 4326, agr = "constant")

# Create map with FRP sum
f1.2 <- ggplot() + 
  geom_sf(data = shp.ca, fill = NA, alpha = 0.2, size = 0.25, color = 'black') +
  geom_sf(data = shp.us, fill = NA, alpha = 0.2, size = 0.25, color = 'black') +
  geom_sf(data = loc.sf, color = 'gray', size = 0.1, alpha = 0.8) +
  geom_sf(data = top200.frp.sf, 
          aes(size = sumfrp,
              color = sumfrp), 
          alpha = 0.8) +
  scale_size_continuous(range = c(0.6, 3)) + 
  scale_color_viridis_c(
    option = "inferno",  
    direction = -1,       
    name = "FRP × hours",
    breaks = pretty_breaks(4)
  ) +
  coord_sf(xlim = c(-170, -50),  
           ylim = c(25, 75)) +    
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

# Save the plot
# ggsave(plot = f1.2, "024_plots/sup_fig1_frp_top200.pdf", width = 180, height = 180, units = "mm")

# Create separate maps for USA and Canada regions ------------------------
# For creating more detailed regional maps if needed:

# USA map example
# us_map <- ggplot() +  
#   geom_sf(data = shp.us, fill = NA, alpha = 0.2, size = 0.25, color = 'black') +
#   geom_sf(data = subset(loc.sf, daily_fp_active$country == "USA"), 
#           color = 'gray', size = 0.1, alpha = 0.8) +
#   geom_sf(data = st_as_sf(top100usa, coords = c("long", "lat"), crs = 4326) %>%
#             arrange(sumhr), 
#           aes(size = sumhr, color = sumhr),
#           alpha = 0.8) +
#   scale_size_continuous(range = c(0.6, 3)) + 
#   scale_color_viridis_c(
#     option = "inferno",  
#     direction = -1,
#     breaks = pretty_breaks(4)
#   ) +
#   theme_void() +
#   coord_sf(crs = 5070)  # Lambert Conformal Conic projection for contiguous US

# Canada map example
# canada_map <- ggplot() +  
#   geom_sf(data = shp.ca, fill = NA, alpha = 0.2, size = 0.25, color = 'black') +
#   geom_sf(data = subset(loc.sf, daily_fp_active$country == "CAN"), 
#           color = 'gray', size = 0.1, alpha = 0.8) +
#   geom_sf(data = st_as_sf(top100can, coords = c("long", "lat"), crs = 4326) %>%
#             arrange(sumhr), 
#           aes(size = sumhr, color = sumhr),
#           alpha = 0.8) +
#   scale_size_continuous(range = c(0.6, 3)) + 
#   scale_color_viridis_c(
#     option = "inferno",  
#     direction = -1,
#     breaks = pretty_breaks(4)
#   ) +
#   theme_void() +
#   coord_sf(crs = 3979)  # Lambert Conformal Conic projection for Canada# 07_spatial_visualization.R
# Spatial visualization of active burning hours across North America (2017-2023)
# This script creates maps of fire locations and active burning hours by region

# Libraries --------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(sf)
library(ggsci)
library(scales)

# Load data --------------------------------------------------------------
daily_fp <- read.csv('023_outputs/NAfires_GOES_to daily_combine/GOES_combined_daily_fire_pattern.csv')
daily_fp_active <- daily_fp %>% filter(all.frp.mean.mean > 0)
daily_fp_active <- daily_fp_active %>%
  mutate(howmanyhr = ifelse(howmanyhr == 25, 24, howmanyhr))

# Load shapefiles for mapping --------------------------------------------
shp.biome <- st_read('D:/000_collections/010_Nighttime Burning/011_Data/013_Biome_wwf2017/fire_biomes_continent_updated2/fire_biomes_USCA.shp')
shp.biome <- shp.biome %>% filter(gez_name != 'Water', gez_name != "Tropical dry forest")
shp.us <- st_read('D:/000_collections/010_Nighttime Burning/011_Data/013_Biome_wwf2017/cb_2018_us_state_20m/cb_2018_us_state_20m_CONUS_AK.shp')
shp.ca <- st_read('D:/000_collections/010_Nighttime Burning/011_Data/013_Biome_wwf2017/lpr_000b16a_e/lpr_000b16a_e.shp')

# Convert point locations to sf objects ----------------------------------
loc.df <- daily_fp_active %>% dplyr::select(lat, long)
loc.sf <- st_as_sf(loc.df, coords = c("long", "lat"), 
                   crs = 4326, agr = "constant")

# Select top 200 fires by active hours, 100 each from USA and Canada -----
top100can <- daily_fp_active %>% 
  filter(country == "CAN") %>%
  group_by(year, long, lat) %>%
  summarise(lat = lat[1], long = long[1], biome = biome[1], ba = POLY_HA[1], sumhr = sum(howmanyhr)) %>%
  ungroup() %>%
  arrange(desc(sumhr)) %>%
  slice(1:200)

top100usa <- daily_fp_active %>% 
  filter(country == "USA") %>%
  group_by(year, long, lat) %>%
  summarise(lat = lat[1], long = long[1], biome = biome[1], ba = POLY_HA[1], sumhr = sum(howmanyhr)) %>%
  ungroup() %>%
  arrange(desc(sumhr)) %>%
  slice(1:200)

# Combine data and convert to sf object
top200 <- bind_rows(
  mutate(top100can, country = "CAN"),
  mutate(top100usa, country = "USA")
) %>%
  arrange(sumhr)

top200.sf <- st_as_sf(top200, coords = c("long", "lat"), 
                      crs = 4326, agr = "constant")

# Create map with top 200 fires by active hours --------------------------
f1.top200 <- ggplot() + 
  geom_sf(data = shp.ca, fill = NA, alpha = 0.2, size = 0.25, color = 'black') +
  geom_sf(data = shp.us, fill = NA, alpha = 0.2, size = 0.25, color = 'black') +
  geom_sf(data = loc.sf, color = 'gray', size = 0.1, alpha = 0.8) +
  geom_sf(data = top200.sf, 
          aes(size = sumhr,
              color = sumhr), 
          alpha = 0.8) +
  scale_size_continuous(range = c(0.6, 3)) + 
  scale_color_viridis_c(
    option = "inferno",  
    direction = -1,       
    name = "Active hours",
    breaks = pretty_breaks(4)
  ) +
  coord_sf(xlim = c(-170, -50),  
           ylim = c(25, 75)) +    
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

# Save the plot
# ggsave(plot = f1.top200, "024_plots/fig1_top200.pdf", width = 180, height = 180, units = "mm")

# Create map with top 200 fires by FRP sum -------------------------------
# Select top 200 fires by FRP × hours
top100can.frp <- daily_fp_active %>% 
  filter(country == "CAN") %>%
  group_by(year, long, lat) %>%
  summarise(lat = lat[1], long = long[1], biome = biome[1], ba = POLY_HA[1], 
            sumfrp = sum(all.frp.max.mean * howmanyhr)) %>%
  ungroup() %>%
  arrange(desc(sumfrp)) %>%
  slice(1:200)

top100usa.frp <- daily_fp_active %>% 
  filter(country == "USA") %>%
  group_by(year, long, lat) %>%
  summarise(lat = lat[1], long = long[1], biome = biome[1], ba = POLY_HA[1], 
            sumfrp = sum(all.frp.max.mean * howmanyhr)) %>%
  ungroup() %>%
  arrange(desc(sumfrp)) %>%
  slice(1:200)

# Combine data and convert to sf object
top200.frp <- bind_rows(
  mutate(top100can.frp, country = "CAN"),
  mutate(top100usa.frp, country = "USA")
) %>%
  arrange(sumfrp)

top200.frp.sf <- st_as_sf(top200.frp, coords = c("long", "lat"), 
                          crs = 4326, agr = "constant")

# Create map with FRP sum
f1.2 <- ggplot() + 
  geom_sf(data = shp.ca, fill = NA, alpha = 0.2, size = 0.25, color = 'black') +
  geom_sf(data = shp.us, fill = NA, alpha = 0.2, size = 0.25, color = 'black') +
  geom_sf(data = loc.sf, color = 'gray', size = 0.1, alpha = 0.8) +
  geom_sf(data = top200.frp.sf, 
          aes(size = sumfrp,
              color = sumfrp), 
          alpha = 0.8) +
  scale_size_continuous(range = c(0.6, 3)) + 
  scale_color_viridis_c(
    option = "inferno",  
    direction = -1,       
    name = "FRP × hours",
    breaks = pretty_breaks(4)
  ) +
  coord_sf(xlim = c(-170, -50),  
           ylim = c(25, 75)) +    
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

# Save the plot
# ggsave(plot = f1.2, "024_plots/sup_fig1_frp_top200.pdf", width = 180, height = 180, units = "mm")

# Create separate maps for USA and Canada regions ------------------------
# For creating more detailed regional maps if needed:

# USA map example
# us_map <- ggplot() +  
#   geom_sf(data = shp.us, fill = NA, alpha = 0.2, size = 0.25, color = 'black') +
#   geom_sf(data = subset(loc.sf, daily_fp_active$country == "USA"), 
#           color = 'gray', size = 0.1, alpha = 0.8) +
#   geom_sf(data = st_as_sf(top100usa, coords = c("long", "lat"), crs = 4326) %>%
#             arrange(sumhr), 
#           aes(size = sumhr, color = sumhr),
#           alpha = 0.8) +
#   scale_size_continuous(range = c(0.6, 3)) + 
#   scale_color_viridis_c(
#     option = "inferno",  
#     direction = -1,
#     breaks = pretty_breaks(4)
#   ) +
#   theme_void() +
#   coord_sf(crs = 5070)  # Lambert Conformal Conic projection for contiguous US

# Canada map example
# canada_map <- ggplot() +  
#   geom_sf(data = shp.ca, fill = NA, alpha = 0.2, size = 0.25, color = 'black') +
#   geom_sf(data = subset(loc.sf, daily_fp_active$country == "CAN"), 
#           color = 'gray', size = 0.1, alpha = 0.8) +
#   geom_sf(data = st_as_sf(top100can, coords = c("long", "lat"), crs = 4326) %>%
#             arrange(sumhr), 
#           aes(size = sumhr, color = sumhr),
#           alpha = 0.8) +
#   scale_size_continuous(range = c(0.6, 3)) + 
#   scale_color_viridis_c(
#     option = "inferno",  
#     direction = -1,
#     breaks = pretty_breaks(4)
#   ) +
#   theme_void() +
#   coord_sf(crs = 3979)  # Lambert Conformal Conic projection for Canada