
# Libararies --------------------------------------------------------------


rm(list = ls())
getwd()
setwd("D:\\000_collections\\222_hourly fire potential")
library(gtools)
library(dplyr)
library(foreach)
library(doParallel)
library(parallel)
library(tcltk)
library(doSNOW)
library(ggplot2)

# active days for active or non-active hours ------------------------------

# Fire_all_daily <- read.csv('023_outputs\\daycombo_feature\\combo_all.csv')
firepath <- mixedsort(dir(path='D:\\000_collections\\222_hourly fire potential\\023_outputs\\NAfires_extrect_2017_2023\\',full.names = T,pattern = '(.csv)$'))
head(firepath)

Nights_FeatDesi <- function(i){
  library(dplyr)
  library(foreach)
  library(doParallel)
  library(parallel)
  library(lutz)
  library(suncalc)
  Fire_indi <- read.csv(firepath[i])
  year <- Fire_indi$year[1]
  Fire_indi <- Fire_indi %>% 
    mutate(naturalDay=ifelse((date<sunrise),day-1,day)) %>% 
    mutate(month=as.Date(day, origin = paste(as.character(year-1) ,'-12-31', sep = ""))) %>% 
    mutate(month=as.numeric(strftime(month, "%m"))) %>% 
    mutate(season = case_when((month >= 3 & month <=5) ~ 1,
                              (month >= 6 & month <=8) ~ 2,
                              (month >= 9 & month <=11) ~ 3,
                              (month==12|month==1|month==2) ~ 4)) %>%
    mutate(label = ifelse(is.na(frp.total),0,1))
  
  
  Fire_indi = Fire_indi %>% 
    mutate(rh=rh.mean,temp=temp.mean-273.15,ffmc=ffmc.mean,
           winds=winds.mean,prec=prec.mean,isi=isi.mean) %>% 
    dplyr::select(year:lat, frp.total,min,frp.mean:biome,date:sunset_lastday,dorn,startburnDOY,endburnDOY,POLY_HA,id_glb,naturalDay:isi)
  ## Attention! if dorn==na, dorn==1
  Fire_indi = Fire_indi %>% mutate(dorn = ifelse(is.na(dorn), 0, dorn),
                                   naturalDay = ifelse(is.na(naturalDay), day, naturalDay))
  
  startburnDOY <- Fire_indi$startburnDOY[1]
  endburnDOY <- Fire_indi$endburnDOY[1]
  if (startburnDOY==endburnDOY) {endburnDOY=endburnDOY+1}
  startburnDOY
  endburnDOY
  
  if (length(startburnDOY)>0&length(endburnDOY)>0) {
    
    Fire_indi_Nights_Feature <- function(j){
      sunrise2sunrise <- Fire_indi %>% filter(naturalDay==j)
      if (sum(sunrise2sunrise$frp.total,na.rm = T)>0) {
        first_row <- which(sunrise2sunrise$frp.total > 0)[1]
        last_row <- which(sunrise2sunrise$frp.total > 0) %>% tail(1)
        
        sunrise2sunrise$dailyspanlabel <- 0
        sunrise2sunrise$dailyspanlabel[first_row:last_row] <- 1
        sunrise2sunrise
      }
    }
    Fire_indi_Feature <- foreach(j=startburnDOY:(endburnDOY), .combine = 'rbind') %do%  Fire_indi_Nights_Feature(j)#, .combine = 'rbind'
    # replace_inf_na <- function(x) {
    #   x[is.infinite(x) | is.nan(x)] <- NA
    #   return(x)
    # }
    # 
    # # 应用该函数到统计结果中
    # Fire_indi_Feature <- Fire_indi_Feature %>% mutate_all(replace_inf_na)
    Fire_indi_Feature
  }
  
}

# 设置错误日志函数
log_error <- function(task_id, error_msg) {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  error_log <- paste("Error at task", task_id, "-", error_msg)
  write(error_log, file = "error_log.allinfo_hourly.txt", append = TRUE)
  cat(timestamp, error_log, "\n")
}

# 将firepath分为5个批次
num_batches <- 1
batch_size <- ceiling(length(firepath) / num_batches)

# 记录开始时间
start_time <- Sys.time()

# 遍历每个批次
for (batch_num in 1:num_batches) {
  # 检测可用的核心数量
  no_cores <- detectCores() - 14
  cl <- makeSOCKcluster(no_cores)
  registerDoSNOW(cl)
  
  print(paste0(batch_num,'batch_started'))
  # 计算当前批次的索引范围
  start_idx <- (batch_num - 1) * batch_size + 1
  end_idx <- min(batch_num * batch_size, length(firepath))
  
  # 设置进度条
  pb <- txtProgressBar(min = start_idx, max = end_idx, style = 3)
  progress <- function(n) setTxtProgressBar(pb, n)
  opts <- list(progress = progress)
  
  # 运行每个批次的foreach循环
  Nights_FeatDesi_batch <- foreach(i = start_idx:end_idx, .combine = 'rbind', .options.snow = opts) %dopar% {
    tryCatch(
      Nights_FeatDesi(i),
      error = function(e) {
        log_error(i, conditionMessage(e))
        NULL
      }
    )
  }
  
  # 将当前批次结果保存为临时文件
  batch_file <- paste0("023_outputs\\NAfires_active days_hourly burning or not_2017_2023\\activedays_hourly_", batch_num, ".csv")
  write.table(Nights_FeatDesi_batch, file = batch_file, sep = ',', row.names = FALSE)
  
  # 清理当前批次结果，释放内存
  rm(Nights_FeatDesi_batch)
  gc()  # 强制进行垃圾回收
  print(paste0(batch_num,'batch_finished'))
  # 停止集群
  stopCluster(cl)
}



# 合并所有批次的临时文件
all_batches <- do.call(rbind, lapply(1:num_batches, function(batch_num) {
  batch_file <- paste0("023_outputs\\NAfires_active days_hourly burning or not_2017_2023\\activedays_hourly_combo_", batch_num, ".csv")
  read.csv(batch_file)
}))

# 写入最终文件
# write.table(all_batches, file = '023_outputs/daily_fire_pattern_goes_hourly.csv', sep = ',', row.names = FALSE)

# 记录结束时间并计算总时间
end_time <- Sys.time()
print(end_time - start_time)



# Combo_Daily -------------------------------------------------------------


CsvCombDaily <- function(i)  {
  library(dplyr)
  library(tidyr)
  library(foreach)
  library(doParallel)
  library(parallel)
  Fire_indi <- read.csv(firepath[i])
  year <- Fire_indi$year[1]
  ##
  Fire_indi <-  Fire_indi %>%
    mutate(month=as.Date(day, origin = paste(as.character(year-1) ,'-12-31', sep = ""))) %>%
    mutate(month=as.numeric(strftime(month, "%m"))) %>%
    mutate(season = case_when((month >= 3 & month <=5) ~ 'spring',
                              (month >= 6 & month <=8) ~ 'summer',
                              (month >= 9 & month <=11) ~ 'fall',
                              (month==12|month==1|month==2) ~ 'winter')) %>%
    mutate(BUI=bui.mean,DMC=dmc.mean,DC=dc.mean,FWI=fwi.mean) %>%
    dplyr::select(year:day,lat,long,month,season,BUI:FWI)

  Fire_indi

}
####start
firepath <- mixedsort(dir(path='D:\\000_collections\\222_hourly fire potential\\023_outputs\\NAfires_daily_extrect_2017_2023\\',full.names = T,pattern = '(.csv)$'))
head(firepath)

log_error <- function(task_id, error_msg) {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  error_log <- paste("Error at task", task_id, "-", error_msg)
  write(error_log, file = "023_outputs\\error_log.txt", append = TRUE)
  cat(timestamp, error_log, "\n")
}
no_cores<-detectCores()-10
cl <- makeSOCKcluster(no_cores)
registerDoSNOW(cl)
pb <- txtProgressBar(min=1, max=length(firepath), style=3)#length(NAfire_year)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress=progress)
start_time <- Sys.time()
Fire_all_daily <- foreach(i=1:length(firepath), .combine = 'rbind', .options.snow = opts) %dopar%  {
  tryCatch(
    CsvCombDaily(i),
    error = function(e) {
      log_error(i, conditionMessage(e))
      NULL
    }
  )}
stopCluster(cl)
end_time <- Sys.time()
end_time - start_time

write.table(Fire_all_daily,file='D:\\000_collections\\222_hourly fire potential\\023_outputs\\NAfires_daily_combo\\NAfires_daily_combo.csv',sep = ',', row.names=FALSE)
