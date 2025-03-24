
# Libararies --------------------------------------------------------------


rm(list = ls())
getwd()
setwd("D:\\000_collections\\222_hourly fire potential")
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

# FIRE DATA - NBAC & MTBS -------------------------------------------------

################################################################################ fire data & biome
nbac17to23 <- unlist(readOGR("D:\\000_collections\\020_Chapter2_climate change\\021_Data\\fire data\\canada\\nbac\\nbac_2017_2023.shp"))
mtbs17to22 <- unlist(readOGR("D:\\000_collections\\020_Chapter2_climate change\\021_Data\\fire data\\usa\\mtbs\\mtbs_2017_to_2022.shp"))
nifc23 <- unlist(readOGR("D:\\000_collections\\020_Chapter2_climate change\\021_Data\\fire data\\usa\\nifc_2023_us.shp"))


####NBAC

nbac17to23$Country <- 'CAN'
nbac17to23$SDATE <- nbac17to23$HS_SDATE
nbac17to23$EDATE <- nbac17to23$HS_EDATE
nbac17to23$AFSDATE <- nbac17to23$AG_SDATE
nbac17to23$AFEDATE <- nbac17to23$AG_EDATE
nbac17to23$firename <-  paste(nbac17to23$YEAR,'_',nbac17to23$ADMIN_AREA,'_',nbac17to23$NFIREID,'_',nbac17to23$POLY_HA,sep = '')
nbac17to23$MTBS_ID = NA
nbac17to23$Ig_Date <- NA
nbac17to23$fireCausOrType <- nbac17to23$FIRECAUS
nbac17to23$source <- 'NBAC'
nbac17to23$inci_name <- NA

####MTBS
mtbs17to22$Country <- 'USA'
mtbs17to22$SDATE <- NA
mtbs17to22$EDATE <- NA
mtbs17to22$AFSDATE <- NA
mtbs17to22$AFEDATE <- NA
mtbs17to22$YEAR <- as.integer(format(as.Date(mtbs17to22$Ig_Date, "%Y/%m/%d"),"%Y"))
mtbs17to22$POLY_HA <- as.numeric(mtbs17to22$BurnBndAc)*0.404686
mtbs17to22$firename <-  paste(mtbs17to22$YEAR,'_',mtbs17to22$Incid_Name,'_',mtbs17to22$Event_ID,'_',mtbs17to22$POLY_HA)
mtbs17to22$MTBS_ID = mtbs17to22$Event_ID
mtbs17to22$fireCausOrType <- mtbs17to22$Incid_Type
mtbs17to22$source <- 'MTBS'
mtbs17to22$inci_name <- mtbs17to22$Incid_Name

####NIFC
nifc23$Country <- 'USA'
nifc23$SDATE <- NA
nifc23$EDATE <- NA
nifc23$AFSDATE <- NA
nifc23$AFEDATE <- NA
nifc23$Ig_Date <- NA
nifc23$YEAR <- 2023
nifc23$POLY_HA <- as.numeric(nifc23$GIS_ACRES)*0.404686
nifc23$firename <-  nifc23$GEO_ID
nifc23$MTBS_ID = nifc23$IRWINID
nifc23$fireCausOrType <- NA
nifc23$source <- 'NIFC'
nifc23$inci_name <- nifc23$GEO_ID


##aline with coords variables and fire data
raster.brick = brick("D:\\000_collections\\020_Chapter2\\021_Data\\0213_FireWeather79-16\\Hourly_FFMC_NA\\hFFMC1979_1..nc")
raster.brick = rotate(raster.brick)
nbac17to23 = spTransform(nbac17to23, crs(raster.brick))
mtbs17to22 = spTransform(mtbs17to22, crs(raster.brick))
nifc23 = spTransform(nifc23, crs(raster.brick))


# Start and end dates -----------------------------------------------------

##combine
NAfire17to23 <- bind(nbac17to23,mtbs17to22,nifc23)
NAfire17to23 <- NAfire17to23[,c('Country','YEAR','SDATE','EDATE','AFSDATE','AFEDATE','Ig_Date','POLY_HA','firename','MTBS_ID','fireCausOrType','source','inci_name')]

NAfire17to23$long<-coordinates(NAfire17to23)[,1]
NAfire17to23$lat <- coordinates(NAfire17to23)[,2]
NAfire17to23 <- NAfire17to23[NAfire17to23$POLY_HA > 200,]

##remove the duplicated ones
coord_poly <- lapply(NAfire17to23@polygons, function(x){lapply(x@Polygons, function(x){coordinates(x)})}) 
NAfire17to23 <- NAfire17to23[!duplicated(coord_poly),]

##separate and combine
merged_data <- NULL
for (year in unique(NAfire17to23$YEAR)) {
  current_year_data <- NAfire17to23[NAfire17to23$YEAR == year, ]
  current_year_data$id_yr <- 1:length(current_year_data)
  if (is.null(merged_data)) {
    merged_data <- current_year_data
  } else {
    merged_data <- rbind(merged_data, current_year_data)
  }
}
merged_data$YEAR <- as.numeric(merged_data$YEAR)
NAfire17to23 <- merged_data
NAfire17to23$id_glb <- 1:length(NAfire17to23)
length(NAfire17to23)

##start and end date
dfNAfire17to23 <- NAfire17to23@data %>% rename(year=YEAR) %>% mutate(year=as.numeric(year))
dfNAfire17to23 <- dfNAfire17to23 %>% mutate(startdate=pmin(as.numeric(strftime(SDATE, format = "%j")),as.numeric(strftime(AFSDATE, format = "%j")),as.numeric(strftime(Ig_Date, format = "%j")),na.rm = T),
                                            enddate=pmax(as.numeric(strftime(EDATE, format = "%j")),as.numeric(strftime(AFEDATE, format = "%j")),na.rm = T)) %>% 
  mutate(across(startdate:enddate,~replace_na(.x, 999)))

## biome
shp.biome <- readOGR('D:\\000_collections\\020_Chapter2\\021_Data\\0214_Biome_wwf2017\\fire_biomes_continent_updated2\\fire_biomes_continent_updated2.shp')

# GOES detection ----------------------------------------------------------

colselect <- c(NA,NA,'NULL','NULL',NA,NA,'NULL',NA,'NULL',NA,NA,NA,NA)
GOES17 <- read.csv(file='D:\\000_collections\\020_Chapter2\\021_Data\\0211_FireDetection_GOESR\\2017\\GOES2017_16_alldata.csv', colClasses = colselect, header=T)
GOES18 <- read.csv(file='D:\\000_collections\\020_Chapter2\\021_Data\\0211_FireDetection_GOESR\\2018/GOES2018_1617_alldata.csv', colClasses = colselect, header=T)
GOES19 <- read.csv(file='D:\\000_collections\\020_Chapter2\\021_Data\\0211_FireDetection_GOESR\\2019/GOES2019_1617_alldata.csv', colClasses = colselect, header=T)
GOES20 <- read.csv(file='D:\\000_collections\\020_Chapter2\\021_Data\\0211_FireDetection_GOESR\\2020\\GOES2020_1617_alldata.csv', colClasses = colselect, header=T)
GOES21 = readr::read_rds("D:\\000_collections\\020_Chapter2\\021_Data\\0211_FireDetection_GOESR\\2021/G1617_2021.Rds")
GOES22 = readr::read_rds("D:\\000_collections\\020_Chapter2\\021_Data\\0211_FireDetection_GOESR\\2022/G1617_2022.Rds")
GOES23 = readr::read_rds("D:\\000_collections\\020_Chapter2\\021_Data\\0211_FireDetection_GOESR\\2023/G1618_2023.Rds")


GOES17 <- GOES17 %>% filter(long>=raster.brick@extent@xmin&long<=raster.brick@extent@xmax&lat>=raster.brick@extent@ymin&lat<=raster.brick@extent@ymax)
GOES18 <- GOES18 %>% filter(long>=raster.brick@extent@xmin&long<=raster.brick@extent@xmax&lat>=raster.brick@extent@ymin&lat<=raster.brick@extent@ymax)
GOES19 <- GOES19 %>% filter(long>=raster.brick@extent@xmin&long<=raster.brick@extent@xmax&lat>=raster.brick@extent@ymin&lat<=raster.brick@extent@ymax)
GOES20 <- GOES20 %>% filter(long>=raster.brick@extent@xmin&long<=raster.brick@extent@xmax&lat>=raster.brick@extent@ymin&lat<=raster.brick@extent@ymax)
GOES21 <- GOES21 %>% filter(long>=raster.brick@extent@xmin&long<=raster.brick@extent@xmax&lat>=raster.brick@extent@ymin&lat<=raster.brick@extent@ymax)
GOES22 <- GOES22 %>% filter(long>=raster.brick@extent@xmin&long<=raster.brick@extent@xmax&lat>=raster.brick@extent@ymin&lat<=raster.brick@extent@ymax)
GOES23 <- GOES23 %>% filter(long>=raster.brick@extent@xmin&long<=raster.brick@extent@xmax&lat>=raster.brick@extent@ymin&lat<=raster.brick@extent@ymax)

GOES17_cp <- GOES17
GOES18_cp <- GOES18
GOES19_cp <- GOES19
GOES20_cp <- GOES20
GOES21 <- GOES21[order(row.names(GOES21)), ]
row.names(GOES21) <- NULL
GOES21 <- transform(GOES21, row_number = seq_len(nrow(GOES21)))
GOES21_cp <- GOES21

GOES22 <- GOES22[order(row.names(GOES22)), ]
row.names(GOES22) <- NULL
GOES22 <- transform(GOES22, row_number = seq_len(nrow(GOES22)))
GOES22_cp <- GOES22

GOES23 <- GOES23[order(row.names(GOES23)), ]
row.names(GOES23) <- NULL
GOES23 <- transform(GOES23, row_number = seq_len(nrow(GOES23)))
GOES23_cp <- GOES23

## csv to spatial df
coordinates(GOES17) <- ~ long + lat
coordinates(GOES18) <- ~ long + lat
coordinates(GOES19) <- ~ long + lat
coordinates(GOES20) <- ~ long + lat
coordinates(GOES21) <- ~ long + lat
coordinates(GOES22) <- ~ long + lat
coordinates(GOES23) <- ~ long + lat

##aline with
proj4string(GOES17) <- proj4string(NAfire17to23)#NAfire_year
proj4string(GOES18) <- proj4string(NAfire17to23)#NAfire_year
proj4string(GOES19) <- proj4string(NAfire17to23)#NAfire_year
proj4string(GOES20) <- proj4string(NAfire17to23)#NAfire_year
proj4string(GOES21) <- proj4string(NAfire17to23)#NAfire_year
proj4string(GOES22) <- proj4string(NAfire17to23)#NAfire_year
proj4string(GOES23) <- proj4string(NAfire17to23)#NAfire_year


## Fire weather variables --------------------------------------------------
BUI <- mixedsort(dir('D:\\000_collections\\020_Chapter2\\021_Data\\0213_FireWeather79-16\\Daily_BUI_NA\\', full.names = T,pattern = '(.nc)$'))
DC <- mixedsort(dir('D:\\000_collections\\020_Chapter2\\021_Data\\0213_FireWeather79-16\\Daily_DC_NA\\', full.names = T,pattern = '(.nc)$'))
DMC <- mixedsort(dir('D:\\000_collections\\020_Chapter2\\021_Data\\0213_FireWeather79-16\\Daily_DMC_NA\\', full.names = T,pattern = '(.nc)$'))
FWI <- mixedsort(dir('D:\\000_collections\\020_Chapter2\\021_Data\\0213_FireWeather79-16\\Daily_FWI_NA\\', full.names = T,pattern = '(.nc)$'))

var.bui.nc.2017 <- brick(BUI[39])
var.bui.nc.2018 <- brick(BUI[40])
var.bui.nc.2019 <- brick(BUI[41])
var.bui.nc.2020 <- brick(BUI[42])
var.bui.nc.2021 <- brick(BUI[43])
var.bui.nc.2022 <- brick(BUI[44])
var.bui.nc.2023 <- brick(BUI[45])

var.dmc.nc.2017 <- brick(DMC[39])
var.dmc.nc.2018 <- brick(DMC[40])
var.dmc.nc.2019 <- brick(DMC[41])
var.dmc.nc.2020 <- brick(DMC[42])
var.dmc.nc.2021 <- brick(DMC[43])
var.dmc.nc.2022 <- brick(DMC[44])
var.dmc.nc.2023 <- brick(DMC[45])

var.dc.nc.2017 <- brick(DC[39])
var.dc.nc.2018 <- brick(DC[40])
var.dc.nc.2019 <- brick(DC[41])
var.dc.nc.2020 <- brick(DC[42])
var.dc.nc.2021 <- brick(DC[43])
var.dc.nc.2022 <- brick(DC[44])
var.dc.nc.2023 <- brick(DC[45])

var.fwi.nc.2017 <- brick(FWI[39])
var.fwi.nc.2018 <- brick(FWI[40])
var.fwi.nc.2019 <- brick(FWI[41])
var.fwi.nc.2020 <- brick(FWI[42])
var.fwi.nc.2021 <- brick(FWI[43])
var.fwi.nc.2022 <- brick(FWI[44])
var.fwi.nc.2023 <- brick(FWI[45])

var.bui.nc.2017 <- rotate(var.bui.nc.2017)
var.bui.nc.2018 <- rotate(var.bui.nc.2018)
var.bui.nc.2019 <- rotate(var.bui.nc.2019)
var.bui.nc.2020 <- rotate(var.bui.nc.2020)
var.bui.nc.2021 <- rotate(var.bui.nc.2021)
var.bui.nc.2022 <- rotate(var.bui.nc.2022)
var.bui.nc.2023 <- rotate(var.bui.nc.2023)

var.dmc.nc.2017 <- rotate(var.dmc.nc.2017)
var.dmc.nc.2018 <- rotate(var.dmc.nc.2018)
var.dmc.nc.2019 <- rotate(var.dmc.nc.2019)
var.dmc.nc.2020 <- rotate(var.dmc.nc.2020)
var.dmc.nc.2021 <- rotate(var.dmc.nc.2021)
var.dmc.nc.2022 <- rotate(var.dmc.nc.2022)
var.dmc.nc.2023 <- rotate(var.dmc.nc.2023)

var.dc.nc.2017 <- rotate(var.dc.nc.2017)
var.dc.nc.2018 <- rotate(var.dc.nc.2018)
var.dc.nc.2019 <- rotate(var.dc.nc.2019)
var.dc.nc.2020 <- rotate(var.dc.nc.2020)
var.dc.nc.2021 <- rotate(var.dc.nc.2021)
var.dc.nc.2022 <- rotate(var.dc.nc.2022)
var.dc.nc.2023 <- rotate(var.dc.nc.2023)

var.fwi.nc.2017 <- rotate(var.fwi.nc.2017)
var.fwi.nc.2018 <- rotate(var.fwi.nc.2018)
var.fwi.nc.2019 <- rotate(var.fwi.nc.2019)
var.fwi.nc.2020 <- rotate(var.fwi.nc.2020)
var.fwi.nc.2021 <- rotate(var.fwi.nc.2021)
var.fwi.nc.2022 <- rotate(var.fwi.nc.2022)
var.fwi.nc.2023 <- rotate(var.fwi.nc.2023)


################################################################################extract function
f<-function(i){
  library(rgdal)
  library(raster)
  library(sp)   
  library(sf)
  library(dplyr)
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
  #select polygon
  shp.indi <- NAfire17to23[i,]
  #add buffer 2000m ~ i pixel of GOES-R AF
  sp_polygons <- st_as_sf(shp.indi) 
  sp_polygons <- sf::st_make_valid(sp_polygons) %>% st_buffer(2000)
  shp.indi = as(sp_polygons,'Spatial')
  
  year.indi <- shp.indi$YEAR
  id_yr.indi <- shp.indi$id_yr
  startdate.indi <- (dfNAfire17to23 %>% filter(year==year.indi,id_yr==id_yr.indi))$startdate
  enddate.indi <- (dfNAfire17to23 %>% filter(year==year.indi,id_yr==id_yr.indi))$enddate
  
  ##GOES starts on 2017-05-24
  if (year.indi==2017&enddate.indi<144) { 
    outputYoN = 0}else{outputYoN = 1}
  if (year.indi==2017&startdate.indi<144&enddate.indi>144) { 
    startdate.indi = 144
  }
  if (year.indi==2017) {    
    GOES <- GOES17
    GOES_cp <- GOES17_cp}
  if (year.indi==2018) {    
    GOES <- GOES18
    GOES_cp <- GOES18_cp}
  if (year.indi==2019) {    
    GOES <- GOES19
    GOES_cp <- GOES19_cp}
  if (year.indi==2020) {    
    GOES <- GOES20
    GOES_cp <- GOES20_cp}
  if (year.indi==2021) {    
    GOES <- GOES21
    GOES_cp <- GOES21_cp}
  if (year.indi==2022) {    
    GOES <- GOES22
    GOES_cp <- GOES22_cp}
  if (year.indi==2023) {    
    GOES <- GOES23
    GOES_cp <- GOES23_cp}
  
  #GOES over shp
  GOESovershp <- over(GOES, shp.indi)
  #remove all rows na
  GOESovershpna <- GOESovershp[complete.cases(GOESovershp$long),]
  #match the goes fire info
  GOES_rowind = as.numeric(rownames(GOESovershpna))
  GOES_fireinfo = GOES_cp[GOES_rowind,]
  ##summarize
  #keep other columns
  G19info1=GOES_fireinfo %>%
    group_by(day,hr) %>%
    summarise_each(funs(sum)) %>% 
    rename(frp.total=frp)
  #summarize frp and count hotspots number
  G19info2=GOES_fireinfo %>%
    group_by(year,day ,hr) %>%
    summarise(frp.mean=mean(frp),frp.total=sum(frp),spotsnum=n(),afarea.mean=mean(area,na.rm=T),
              aftemp.mean=mean(temp,na.rm=T))
  #join the full df
  GOES_fireinfo <- full_join(G19info1,G19info2,by=c('frp.total','day','hr')) %>% 
    rename(year=year.y) %>% dplyr::select(-year.x,-area,-temp)
  
  rm(GOESovershp,GOESovershpna,GOES_rowind,G19info1,G19info2)
  
  ## find the start day and end day for consectuive active hour if orignial date equals 999
  if (dim(GOES_fireinfo)[1] > 0){
    vals <- expand.grid(year=year.indi,day = seq(as.integer(GOES_fireinfo[1,1]), 
                                                 max(as.integer(GOES_fireinfo$day)), 1),
                        hr = seq(0, 23, 1)) %>% arrange(day,hr)
    GOES_fireinfo=merge(vals,GOES_fireinfo,all = TRUE)
    GOES_fireinfo = GOES_fireinfo %>% mutate(leadfrp = lead(frp.total))
    
    if (startdate.indi<999) {
      startdoy = startdate.indi
      startdate=as.Date(as.integer(startdoy), origin = paste(year.indi-1,'-12-31', sep = ""))
      startmonth <- as.integer(format(startdate,"%m"))
      startday <- as.integer(format(startdate,"%d"))
    }else {
      tworow = rowMeans(GOES_fireinfo %>% dplyr::select(frp.total,leadfrp),na.rm = F)
      startdoy = GOES_fireinfo[which(tworow>0),]$day[1]
      startdate=as.Date(as.integer(startdoy), origin = paste(year.indi-1,'-12-31', sep = ""))
      startmonth <- as.integer(format(startdate,"%m"))
      startday <- as.integer(format(startdate,"%d"))
    }
    if (enddate.indi<999) {
      enddoy = enddate.indi+1
      enddate=as.Date(as.integer(enddoy), origin = paste(year.indi-1,'-12-31', sep = ""))
      endmonth <- as.integer(format(enddate,"%m"))
      endday <- as.integer(format(enddate,"%d"))
    }else {
      tworow = rowMeans(GOES_fireinfo %>% dplyr::select(frp.total,leadfrp),na.rm = F)
      enddoy = max( GOES_fireinfo[which(tworow>0),]$day)+1
      enddate=as.Date(as.integer(enddoy), origin = paste(year.indi-1,'-12-31', sep = ""))
      endmonth <- as.integer(format(enddate,"%m"))
      endday <- as.integer(format(enddate,"%d"))
    }
    
    # remove rows before the fire
    GOES_fireinfo <- GOES_fireinfo %>% filter(day>=as.integer(startdoy),day<=enddoy) %>% 
      dplyr::select(-leadfrp)
    
  }
  
  
  ## no hot spots from GOES in this shapefile
  if (dim(GOES_fireinfo)[1] > 0){   
    # outputtest = data.frame(year.indi,seq.indi)
    
    #expand the df from start date to the last day with 1hour sequence
    startdoy <- as.integer(startdoy)-3
    startdate=as.Date(as.integer(startdoy), origin = paste(year.indi-1,'-12-31', sep = ""))
    startmonth <- as.integer(format(startdate,"%m"))
    startday <- as.integer(format(startdate,"%d"))
    if (startdoy<=0) {
      startdoy=1
      startdate=as.Date(as.integer(startdoy), origin = paste(year.indi-1,'-12-31', sep = ""))
      startmonth <- as.integer(format(startdate,"%m"))
      startday <- as.integer(format(startdate,"%d"))
    }
    if (enddoy>=365) {
      enddoy=365
      enddate=as.Date(as.integer(enddoy), origin = paste(year.indi-1,'-12-31', sep = ""))
      endmonth <- as.integer(format(enddate,"%m"))
      endday <- as.integer(format(enddate,"%d"))
    }
    vals <- expand.grid(year=year.indi,day = seq(startdoy, (as.integer(enddoy)), 1),
                        hr = seq(0, 23, 1))
    GOES_fireinfo=merge(vals,GOES_fireinfo,all = TRUE)
    GOES_fireinfo=GOES_fireinfo[which(GOES_fireinfo$day>=startdoy),]
    GOES_fireinfo=GOES_fireinfo[which(GOES_fireinfo$day<=enddoy),]
    rm(vals)
    
    #which(!is.na(GOES_fireinfo$frp.total), arr.ind=TRUE)
    
    ##extract biome id
    shp.indi.1 <- shp.indi
    pointtoplot <- data.frame(x=coordinates(shp.indi.1)[,1], y=coordinates(shp.indi.1)[,2])
    coordinates(pointtoplot) <- ~ x + y
    proj4string(pointtoplot) <- CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0")
    pointtoplot <- spTransform(pointtoplot, crs(shp.biome))
    #
    #function over from package sp
    testnip <- data.frame(xx=over(shp.biome, pointtoplot))
    testnip <- na.omit(testnip)
    combine <- shp.biome[row.names(shp.biome) %in% row.names(testnip), ]
    biomeid <- combine$gez_name2[1]
    
    rm(shp.indi.1,pointtoplot,testnip,combine)
    
    ##load variables
    #to select corresponding nc file
    if (year.indi==2017) {  
      var.bui.nc <- var.bui.nc.2017
      var.dmc.nc <- var.dmc.nc.2017
      var.dc.nc <- var.dc.nc.2017
      var.fwi.nc <- var.fwi.nc.2017
    } else if (year.indi==2018) {
      var.bui.nc <- var.bui.nc.2018
      var.dmc.nc <- var.dmc.nc.2018
      var.dc.nc <- var.dc.nc.2018
      var.fwi.nc <- var.fwi.nc.2018
    } else if (year.indi==2019) {
      var.bui.nc <- var.bui.nc.2019
      var.dmc.nc <- var.dmc.nc.2019
      var.dc.nc <- var.dc.nc.2019
      var.fwi.nc <- var.fwi.nc.2019
    }else if (year.indi==2020) { 
      var.bui.nc <- var.bui.nc.2020
      var.dmc.nc <- var.dmc.nc.2020
      var.dc.nc <- var.dc.nc.2020
      var.fwi.nc <- var.fwi.nc.2020
    }else if (year.indi==2021) { 
      var.bui.nc <- var.bui.nc.2021
      var.dmc.nc <- var.dmc.nc.2021
      var.dc.nc <- var.dc.nc.2021
      var.fwi.nc <- var.fwi.nc.2021
    }else if (year.indi==2022) { 
      var.bui.nc <- var.bui.nc.2022
      var.dmc.nc <- var.dmc.nc.2022
      var.dc.nc <- var.dc.nc.2022
      var.fwi.nc <- var.fwi.nc.2022
    }else if (year.indi==2023) { 
      var.bui.nc <- var.bui.nc.2023
      var.dmc.nc <- var.dmc.nc.2023
      var.dc.nc <- var.dc.nc.2023
      var.fwi.nc <- var.fwi.nc.2023}
    
    
    ext.val.bui <- raster::extract(var.bui.nc, shp.indi,weight=T)[[1]]
    nr <- dim(ext.val.bui)[2]-1
    ext.val.bui <- data.frame(ext.val.bui)[,1:nr]
    ext.val.dmc <- data.frame(raster::extract(var.dmc.nc, shp.indi,weight=T)[[1]])[,1:nr]
    ext.val.dc <- data.frame(raster::extract(var.dc.nc, shp.indi,weight=T)[[1]])[,1:nr]
    ext.val.fwi <- data.frame(raster::extract(var.fwi.nc, shp.indi,weight=T)[[1]])[,1:nr]
    
    out.val.bui <- apply(ext.val.bui, 1, 
                         function(x)  x[(startdoy): enddoy])
    out.val.dmc <- apply(ext.val.dmc, 1, 
                         function(x)  x[(startdoy): enddoy])
    out.val.dc <- apply(ext.val.dc, 1, 
                        function(x)  x[(startdoy): enddoy])
    out.val.fwi <- apply(ext.val.fwi, 1, 
                         function(x)  x[(startdoy): enddoy])
    
    out.val.bui <- cbind(rowMeans(out.val.bui,na.rm = T),apply(out.val.bui, 1, median,na.rm = T))
    out.val.dmc <- cbind(rowMeans(out.val.dmc,na.rm = T),apply(out.val.dmc, 1, median,na.rm = T))
    out.val.dc <- cbind(rowMeans(out.val.dc,na.rm = T),apply(out.val.dc, 1, median,na.rm = T))
    out.val.fwi <- cbind(rowMeans(out.val.fwi,na.rm = T),apply(out.val.fwi, 1, median,na.rm = T))
    
    colnames(out.val.bui) <- c('bui.mean','bui.median')
    colnames(out.val.dmc) <- c('dmc.mean','dmc.median')
    colnames(out.val.dc) <- c('dc.mean','dc.median')
    colnames(out.val.fwi) <- c('fwi.mean','fwi.median')
    
    out.val <- cbind(out.val.bui,out.val.dmc,out.val.dc,out.val.fwi)
    out.val <- data.frame(out.val)
    out.val$biome <- biomeid
    
    rm(var.bui.nc,var.dmc.nc,var.dc.nc,var.fwi.nc)
    rm(ext.val.bui,ext.val.dmc,ext.val.dc,ext.val.fwi)
    rm( out.val.bui,out.val.dmc,out.val.dc,out.val.fwi)
    
    ## add timezone, sunrise and sunset
    GOES_fireinfo <- GOES_fireinfo %>% group_by(year,day) %>% 
      summarise(lat=mean(lat),long=mean(long),frp.mean=mean(frp.mean),frp.total=mean(frp.total),spotsnum=mean(spotsnum))
    GOES_fireinfo.var <- cbind.data.frame(GOES_fireinfo,out.val)
    tz <- tz_offset(as.Date(as.integer(GOES_fireinfo.var$day), origin = paste(year.indi-1,'-12-31', sep = "")), tz_lookup_coords(shp.indi$lat, shp.indi$long, method = "accurate"))
    tzoffset <- tz$utc_offset_h
    GOES_fireinfo.var <- cbind(GOES_fireinfo.var,tzoffset)
    ## sunset and sunrise
    tz <- tz_offset(as.Date(as.integer(GOES_fireinfo.var$day), origin = paste(year.indi-1,'-12-31', sep = "")), tz_lookup_coords(shp.indi$lat, shp.indi$long, method = "accurate"))
    sundata <- data.frame(date=as.Date(tz$date_time),lat=shp.indi$lat,lon=shp.indi$long)
    sunrise <- getSunlightTimes(data=sundata,keep="sunrise",tz=tz$tz_name[1])$sunrise
    sunset <- getSunlightTimes(data=sundata,keep="sunset",tz=tz$tz_name[1])$sunset
    
    ## all are local time, even shown as UTC time
    GOES_fireinfo.var$sunrise <- as.character(sunrise)
    GOES_fireinfo.var$sunset <- as.character(sunset)
    
    
    GOES_fireinfo.var$id_yr <- shp.indi$id_yr
    GOES_fireinfo.var$long <- shp.indi$long
    GOES_fireinfo.var$lat <- shp.indi$lat
    
    if (startdate.indi<999) {
      GOES_fireinfo.var$startburnDOY <- startdate.indi
    }else {
      GOES_fireinfo.var$startburnDOY <- startdoy+3
      
    }
    if (enddate.indi<999) {
      GOES_fireinfo.var$endburnDOY <- enddate.indi
      
    }else {
      GOES_fireinfo.var$endburnDOY <- enddoy-1
      
    }
    
    GOES_fireinfo.var$country <- shp.indi$Country
    GOES_fireinfo.var$POLY_HA <- shp.indi@data$POLY_HA
    GOES_fireinfo.var$id_glb <- shp.indi$id_glb
    GOES_fireinfo.var$inci_name <- shp.indi$inci_name
    GOES_fireinfo.var$biome <- biomeid
    
    GOES_fireinfo.var <- GOES_fireinfo.var %>% dplyr::select(year:spotsnum,biome:inci_name,everything())
    
    out.name <- paste('D:\\000_collections\\222_hourly fire potential\\023_outputs\\NAfires_daily_extrect_2017_2023\\',year.indi,'_',shp.indi$id_yr,'_',as.integer( shp.indi$id_glb),'_',shp.indi$Country,'_',
                      round(GOES_fireinfo.var$long[1],2),'_',round(GOES_fireinfo.var$lat[1],2),'_',round(GOES_fireinfo.var$POLY_HA[1],2),'.csv',sep = '')
    
    if (outputYoN ==1) {
      write.table(GOES_fireinfo.var,file=out.name,sep = ',', row.names=FALSE) # keeps the rownames
    }
  }
  
}

################################################################################parallel running
log_error <- function(task_id, error_msg) {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  error_log <- paste("Error at task", task_id, "-", error_msg)
  write(error_log, file = "D:\\000_collections\\222_hourly fire potential\\error_log.txt", append = TRUE)
  cat(timestamp, error_log, "\n")
}
no_cores<-detectCores()-10
cl <- makeSOCKcluster(no_cores)
registerDoSNOW(cl)
pb <- txtProgressBar(min=1, max=length(NAfire17to23), style=3)#length(NAfire17to23)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress=progress)

start_time <- Sys.time()
nbac_csvwrite <- foreach(i=1:length(NAfire17to23), .options.snow = opts) %dopar%  {  
  tryCatch(
    f(i),
    error = function(e) {
      log_error(i, conditionMessage(e))
      NULL
    }
  )}
stopCluster(cl)
end_time <- Sys.time()
end_time - start_time