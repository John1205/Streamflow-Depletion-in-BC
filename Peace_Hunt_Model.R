require(dplyr)
require(sf)
require(streamDepletr)
require(sp)
library(lubridate)
library(magrittr)

## load depletion apportionment output
df.out <- read.csv("Hunt_apportionment.csv", stringsAsFactors=F)
df.cond<-read.csv("well_stream_cond.csv", stringsAsFactors=F)

df.out <- 
  left_join(df.out, df.cond[c('well_num', 'seg_num','cond')], by=c('well_num', 'seg_num')) 

df.out <-  unique(df.out)

df.trans <- read.csv("well_T_kh.csv")
#head(df.trans)
df.trans <- df.trans[df.trans$Trans>0, ]

### what dates do you want output for? (units: number of days since start of pumping)
# convert years and dates to DOY since pumping started
yrs.model <- c(1:35) 
weeks.model<-c(1:52)
DOYs.all<- rep(weeks.model*7, times = length(yrs.model)) + rep(52*(yrs.model-1)*7, each = length(weeks.model))

### load and pre-process data

df.pump <- data.frame(MonthNum = c(1:18, 19:22, 23:26, 27:30, 31:34, 35:38, 39:52),
                      Qw_m3day = c(rep(0, each = length(1:18)), 
                                   rep(19, each = length(19:22)),
                                   rep(108, each = length(23:26)),
                                   rep(214, each = length(27:30)),
                                   rep(171, each = length(31:34)),
                                   rep(84, each = length(35:38)),
                                   rep(0, each = length(39:52))))

df.pump$MonthLengthDays <- 7
# set up long data frames for intermittent_pumping script; separate for outdoor and greenhouse
t.max.yrs <- max(yrs.model)

df.pump.long <- 
  df.pump %>% 
  replicate(t.max.yrs, ., simplify = FALSE) %>% 
  dplyr::bind_rows() %>% 
  transform(EndOfMonthDays = cumsum(MonthLengthDays))
df.pump.long$StartOfMonthDays <- c(1, df.pump.long$EndOfMonthDays[1:(t.max.yrs*52)-1]+1)

# combine identical periods in df.pump.long
i.starts <- which(c(1, diff(df.pump.long$Qw_m3day)) != 0)
i.ends <- c((i.starts-1)[2:length(i.starts)], i.starts[length(i.starts)])

df.pump.compressed <-
  data.frame(StartOfMonthDays = df.pump.long$StartOfMonthDays[i.starts],
             EndOfMonthDays = df.pump.long$EndOfMonthDays[i.ends],
             Qw_m3day = df.pump.long$Qw_m3day[i.ends])

## unique well-seg combos
df.combos <- 
  df.out %>% 
  dplyr::select(segment, well_num, distance, cond) %>% 
  unique()

df.combos <- 
  left_join(df.combos, df.trans, by="well_num")%>% 
  unique()

## calculate depletion for all combos
start.flag.Qs <- T
for (i in 1:dim(df.combos)[1]){
  # identify well-seg combo
  seg <- df.combos$segment[i]
  well <- df.combos$well_num[i]
  
  # get times
  output_t_days <- df.out$time_days[df.out$segment==seg & df.out$well_num==well]
  output_frac <- df.out$frac_depletion[df.out$segment==seg & df.out$well_num==well]
  
  # calculate depletion
  Qs <- intermittent_pumping(t = output_t_days,
                             starts = df.pump.compressed$StartOfMonthDays,
                             stops  = df.pump.compressed$EndOfMonthDays,
                             rates  = df.pump.compressed$Qw_m3day,
                             method = "hunt",
                             d = df.combos$distance[i],
                             S = df.combos$S[i],
                             Tr = df.combos$Trans[i],
                             lmda = df.combos$cond[i]*10,
                             lmda_max = 1)
  
  # compile output
  df.depletion <- data.frame(segment = seg,
                             well_num = well,
                             time_days = output_t_days,
                             Qs = Qs)
  
  if (start.flag.Qs){
    df.Qs <- df.depletion
    start.flag.Qs <- F
  } else {
    df.Qs <- rbind(df.Qs, df.depletion)
  }
  
  print(paste0("Depletion ", i, " of ", dim(df.combos)[1], " complete, ", Sys.time()))
  
}

# combine and save
df.final <- 
  df.Qs %>% 
  left_join(df.out, by=c("segment", "well_num", "time_days")) %>% 
  transform(depletion_m3d = Qs*frac_depletion) %>% 
  dplyr::select(segment, well_num, time_days, Qs, frac_depletion, depletion_m3d) %>% 
  subset(depletion_m3d >= 1e-3) %>% 
  unique() %>%
  write.csv("Peace_Hunt_lmda10.csv", row.names=F)
