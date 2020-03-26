require(dplyr)
require(sf)
require(streamDepletr)
require(sp)
library(lubridate)
library(magrittr)

yrs.model <- c(1:35) 
weeks.model<-c(1:52)
DOYs.all<- rep(weeks.model*7, times = length(yrs.model)) + rep(52*(yrs.model-1)*7, each = length(weeks.model))

df.all<-read.csv("well_stream_geometry.csv")
#head(df.all)
df.all$segment = paste(df.all$seg,df.all$seg_num, sep= "_")

df.trans <- read.csv("well_T.csv")
#head(df.trans)
df.trans<-subset(df.trans, Trans >0 )

df.cond <- read.csv("well_stream_cond.csv")
#head(df.cond)
df.cond$segment = paste(df.cond$seg, df.cond$seg_num, sep= "_")
df.all <- df.all[df.all$seg != "Drain", ]
df.all <- unique(df.all)
min_frac <- 0.01  # minimum depletion considered

w.start.flag <- T
counter <- 0

df.all <- subset(df.all, well_num %in% df.trans$well_num)
df.all<-unique(df.all)

for (w in unique(df.all$well_num)){
  dist_w_pts <- subset(df.all, well_num==w)
  
  dist_w_pts <- dist_w_pts[!((dist_w_pts$well_row == dist_w_pts$seg_row) & (dist_w_pts$well_col == dist_w_pts$seg_col)), ]
  
  # closest point only
  dist_w_pts_closest <- 
    dist_w_pts %>% 
    group_by(segment) %>% 
    filter(distance == min(distance))
  
  # first: Theissen polygons to figure out adjacent catchments
  df.apportion.t <- 
    dist_w_pts_closest %>% 
    apportion_polygon(., 
                      wel_lon = dist_w_pts_closest$well_col[1],  # column
                      wel_lat = dist_w_pts_closest$well_row[1],  # row
                      crs = CRS(st_crs("+init=epsg:32610")[["proj4string"]]),  # any projected CRS (BC UTM?)
                      reach_name = "segment",
                      dist_name = "distance",
                      lon_name = "seg_col",
                      lat_name = "seg_row",
                      min_frac = min_frac) %>% 
    set_colnames(c("segment", "frac_depletion"))
  
  #expanding 
  for (time_days in DOYs.all){
    
    # find maximum distance, based on maximum observed S, Tr, lmda (inclusive estimate)
    max_dist <- depletion_max_distance(Qf_thres = min_frac,
                                       d_interval = 250,
                                       d_min = 500,
                                       d_max = max(dist_w_pts$distance),
                                       method="glover",  # hunt
                                       t = time_days,
                                       S = df.trans$S[df.trans$well_num==w],
                                       Tr = df.trans$Trans[df.trans$well_num==w]
                                       )
    
    # second: use web^2 to apportion to any stream segment that is within max_dist
    #         OR that is adjacent to well based on Theissen Polygon
    df.apportion.w.t <- 
      dist_w_pts %>% 
      subset((distance <= max_dist) | (segment %in% df.apportion.t$segment)) %>% 
      apportion_web(reach_dist = ., 
                    w = 2, 
                    min_frac = min_frac,
                    reach_name = "segment",
                    dist_name = "distance") %>% 
      set_colnames(c("segment", "frac_depletion")) %>% 
      transform(well_num = w,
                time_days = time_days) %>% 
      left_join(dist_w_pts_closest, by=c("segment", "well_num"))
    
    if (w.start.flag){
      df.out <- df.apportion.w.t[ , !(names(df.apportion.w.t) %in% c("lon", "lat", "SegNum_GrowNum"))]
      w.start.flag <- F
    } else {
      df.out <- rbind(df.out, df.apportion.w.t[ , !(names(df.apportion.w.t) %in% c("lon", "lat", "SegNum_GrowNum"))])
    }
    
    # update max_dist starting value
    max_dist_prev <- max_dist
    
  }
  
  # status update
  counter <- counter+1
  print(paste0("Well ", counter, " of ", length(unique(df.all$well_num)), " depletion apportionment complete, ", Sys.time()))
}

write.csv(df.out, "Glover_apportionment.csv", row.names = F)
