library(zoo)
library(plyr)
library(dplyr)
require(rsq)
library(ggplot2)
library(gtable)
library(grDevices)
require(graphics)
library(gridExtra)
library(hydroGOF)
library(reshape2)
library(viridis)
require(streamDepletr)

#senstivity analysis of BX Creek 
#read the files
#select files has same data, inner_join 
#to make sure each dataset has the same dataset
#plot pdf density plot
#compare MAE for the most affected streams
Glover<-read.csv("BX_Glover.csv")
Glover.k01<-read.csv("BX_Glover_01T.csv")
Glover.k001<-read.csv("BX_Glover_001T.csv")
Glover.k10<-read.csv("BX_Glover_10T.csv")
Glover.k100<-read.csv("BX_Glover_100T.csv")

Glover.si10<-read.csv("BX_Glover_SI10.csv")
Glover.si20<-read.csv("BX_Glover_SI20.csv")
Glover.sd10<-read.csv("BX_Glover_SD10.csv")
Glover.sd20<-read.csv("BX_Glover_SD20.csv")
###############################################
Hunt<-read.csv("BX_Hunt.csv")
Hunt.k01<-read.csv("BX_Hunt_01T.csv")
Hunt.k001<-read.csv("BX_Hunt_001T.csv")
Hunt.k10<-read.csv("BX_Hunt_10T.csv")
Hunt.k100<-read.csv("BX_Hunt_100T.csv")

Hunt.si10<-read.csv("BX_Hunt_SI10.csv")
Hunt.si20<-read.csv("BX_Hunt_SI20.csv")
Hunt.sd10<-read.csv("BX_Hunt_SD10.csv")
Hunt.sd20<-read.csv("BX_Hunt_SD20.csv")

Hunt.lmda001<-read.csv("BX_Hunt_lmda001.csv")
Hunt.lmda01<-read.csv("BX_Hunt_lmda01.csv")
Hunt.lmda10<-read.csv("BX_Hunt_lmda10.csv")
Hunt.lmda100<-read.csv("BX_Hunt_lmda100.csv")
###################################################
#select the most affected streams in analytical depletion functions
#select the greatest depletion for each segment
glover.great<-Glover %>%
  group_by(segment) %>%
  slice(which.max(frac_depletion))%>%   # greatest depletion rate != depletion; check whether there is depletion
  as.data.frame() %>%
  select(-c("time_days", "Qs", "frac_depletion", "depletion_m3d"))

glover.great$max<-1
glover.data<-inner_join(Glover, glover.great, by =c("segment","well_num"))

hunt.great<-Hunt %>%
  group_by(segment) %>%
  slice(which.max(frac_depletion))%>%
  as.data.frame() %>%
  select(-c("time_days", "Qs", "frac_depletion", "depletion_m3d"))
hunt.great$max<-1
hunt.data<-inner_join(Hunt, hunt.great, by =c("segment","well_num"))
##############################################################################
flow.k01<-inner_join(glover.data, Glover.k01, by =c("segment", "well_num", "time_days", "frac_depletion"), suffix = c("", ".k01"))
flow.k001<-inner_join(glover.data, Glover.k001, by =c("segment", "well_num", "time_days", "frac_depletion"), suffix = c("", ".k001"))
flow.k10<-inner_join(glover.data, Glover.k10, by =c("segment", "well_num", "time_days", "frac_depletion"), suffix = c("", ".k10"))
flow.k100<-inner_join(glover.data, Glover.k100, by =c("segment", "well_num", "time_days", "frac_depletion"), suffix = c("", ".k100"))
flow.si10<-inner_join(glover.data, Glover.si10, by =c("segment", "well_num", "time_days", "frac_depletion"), suffix = c("", ".SI10"))
flow.si20<-inner_join(glover.data, Glover.si20, by =c("segment", "well_num", "time_days", "frac_depletion"), suffix = c("", ".SI20"))
flow.sd10<-inner_join(glover.data, Glover.sd10, by =c("segment", "well_num", "time_days", "frac_depletion"), suffix = c("", ".SD10"))
flow.sd20<-inner_join(glover.data, Glover.sd20, by =c("segment", "well_num", "time_days", "frac_depletion"), suffix = c("", ".SD20"))

##############################################################################
flow.k01<-inner_join(hunt.data, Hunt.k01, by =c("segment", "well_num", "time_days", "frac_depletion"), suffix = c("", ".k01"))
flow.k001<-inner_join(hunt.data, Hunt.k001, by =c("segment", "well_num", "time_days", "frac_depletion"), suffix = c("", ".k001"))
flow.k10<-inner_join(hunt.data, Hunt.k10, by =c("segment", "well_num", "time_days", "frac_depletion"), suffix = c("", ".k10"))
flow.k100<-inner_join(hunt.data, Hunt.k100, by =c("segment", "well_num", "time_days", "frac_depletion"), suffix = c("", ".k100"))
flow.si10<-inner_join(hunt.data, Hunt.si10, by =c("segment", "well_num", "time_days", "frac_depletion"), suffix = c("", ".SI10"))
flow.si20<-inner_join(hunt.data, Hunt.si20, by =c("segment", "well_num", "time_days", "frac_depletion"), suffix = c("", ".SI20"))
flow.sd10<-inner_join(hunt.data, Hunt.sd10, by =c("segment", "well_num", "time_days", "frac_depletion"), suffix = c("", ".SD10"))
flow.sd20<-inner_join(hunt.data, Hunt.sd20, by =c("segment", "well_num", "time_days", "frac_depletion"), suffix = c("", ".SD20"))
flow.lmda01<-inner_join(hunt.data, Hunt.lmda01, by =c("segment", "well_num", "time_days", "frac_depletion"), suffix = c("", ".lmda01"))
flow.lmda001<-inner_join(hunt.data, Hunt.lmda001, by =c("segment", "well_num", "time_days", "frac_depletion"), suffix = c("", ".lmda001"))
flow.lmda10<-inner_join(hunt.data, Hunt.lmda10, by =c("segment", "well_num", "time_days", "frac_depletion"), suffix = c("", ".lmda10"))
flow.lmda100<-inner_join(hunt.data, Hunt.lmda100, by =c("segment", "well_num", "time_days", "frac_depletion"), suffix = c("", ".lmda100"))
##########################################################################
flow.k001$par<-"k001"
flow.k01$par<-"k01"
flow.k10$par<-"k10"
flow.k100$par<-"k100"
flow.si10$par<-"SI10"
flow.si20$par<-"SI20"
flow.sd10$par<-"SD10"
flow.sd20$par<-"SD20"

flow.lmda001$par<-"lmda001"
flow.lmda01$par<-"lmda01"
flow.lmda10$par<-"lmda10"
flow.lmda100$par<-"lmda100"
############################################
flow.k001$diff<-flow.k001$depletion_m3d.k001-flow.k001$depletion_m3d
flow.k01$diff<-flow.k01$depletion_m3d.k01-flow.k01$depletion_m3d
flow.k10$diff<-flow.k10$depletion_m3d.k10-flow.k10$depletion_m3d
flow.k100$diff<-flow.k100$depletion_m3d.k100-flow.k100$depletion_m3d

flow.si10$diff<-flow.si10$depletion_m3d.SI10-flow.si10$depletion_m3d
flow.si20$diff<-flow.si20$depletion_m3d.SI20-flow.si20$depletion_m3d
flow.sd10$diff<-flow.sd10$depletion_m3d.SD10-flow.sd10$depletion_m3d
flow.sd20$diff<-flow.sd20$depletion_m3d.SD20-flow.sd20$depletion_m3d

flow.lmda001$diff<-flow.lmda001$depletion_m3d.lmda001-flow.lmda001$depletion_m3d
flow.lmda01$diff<-flow.lmda01$depletion_m3d.lmda01-flow.lmda01$depletion_m3d
flow.lmda10$diff<-flow.lmda10$depletion_m3d.lmda10-flow.lmda10$depletion_m3d
flow.lmda100$diff<-flow.lmda100$depletion_m3d.lmda100-flow.lmda100$depletion_m3d
############################################
#select the diff k
flow.k001.diff<-select(flow.k001, diff, par)
flow.k01.diff<-select(flow.k01, diff, par)
flow.k10.diff<-select(flow.k10, diff, par)
flow.k100.diff<-select(flow.k100, diff, par)

flow.si10.diff<-select(flow.si10, diff, par)
flow.si20.diff<-select(flow.si20, diff, par)
flow.sd10.diff<-select(flow.sd10, diff, par)
flow.sd20.diff<-select(flow.sd20, diff, par)

flow.lmda001.diff<-select(flow.lmda001, diff, par)
flow.lmda01.diff<-select(flow.lmda01, diff, par)
flow.lmda10.diff<-select(flow.lmda10, diff, par)
flow.lmda100.diff<-select(flow.lmda100, diff, par)

flow.k<-rbind(flow.k001.diff, flow.k01.diff, flow.k10.diff, flow.k100.diff)
flow.s<-rbind(flow.si10.diff, flow.si20.diff, flow.sd10.diff, flow.sd20.diff)
flow.lmda<-rbind(flow.lmda001.diff, flow.lmda01.diff, flow.lmda10.diff, flow.lmda100.diff)

mu_k<-ddply(flow.k, "par", summarise, grp.mean = mean(diff))
col_values<-c('darkgreen','dodgerblue1', 'goldenrod', 'olivedrab')
k.label<-c('0.01 T','0.1 T', '10 T', '100 T')

ggplot(flow.k, aes(x=diff, color = par)) + 
  geom_density()+theme_bw() +
  #geom_vline(data=mu_k, aes(xintercept = grp.mean, color=par), linetype = "dashed" )+
  scale_color_manual(name = 'Sensitivity', values=col_values, labels = k.label)+
  labs(x = "Difference", y = " Density")+ 
  scale_x_continuous(limits = c(-120, 120), expand=c(0, 0)) +
  scale_y_continuous( limits = c(-0.001, 0.04), expand=c(0, 0)) +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = c(0.85, 0.70),
        panel.grid=element_blank(),
        axis.text.x = element_text(face = "bold"),
        axis.text.y = element_text(face = "bold"),
        legend.title = element_text(size=10, face="bold"),
        plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"))
ggsave("Glover_sensitivity_k.png", dpi = 1000, width = 6, height =5)
#ggsave("Hunt_sensitivity_k.png", dpi = 1000, width = 6, height =5)

mu_s<-ddply(flow.s, "par", summarise, grp.mean = mean(diff))
col_values<-c('darkgreen','dodgerblue1', 'goldenrod', 'olivedrab')
s.label<-c('-20% Sy ','-10% Sy', '+10% Sy', '+20% Sy')

ggplot(flow.s, aes(x=diff, color = par)) + 
  geom_density()+theme_bw() +
  #geom_vline(data=mu_s, aes(xintercept = grp.mean, color=par), linetype = "dashed" )+
  scale_color_manual(name = 'Sensitivity', values=col_values, labels = s.label)+
  labs(x = "Difference (m3/day)", y = " Density")+ 
  scale_x_continuous(limits = c(-120, 120), expand=c(0, 0)) +
  scale_y_continuous( limits = c(-0.01, 0.5), expand=c(0, 0)) +
  #ggtitle("Transimissivity")+
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = c(0.85, 0.70),
        panel.grid=element_blank(),
        axis.text.x = element_text(face = "bold"),
        axis.text.y = element_text(face = "bold"),
        legend.title = element_text(size=10, face="bold"),
        plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"))

ggsave("Glover_sensitivity_sy.png", dpi = 1000, width = 6, height = 5)
#ggsave("Hunt_sensitivity_sy.png", dpi = 1000, width = 6, height = 5)

mu_lmda<-ddply(flow.lmda, "par", summarise, grp.mean = mean(diff))
col_values<-c('darkgreen','dodgerblue1', 'goldenrod', 'olivedrab')
lmda.label<-c('0.01 lmda ','0.1 lmda', '10 lmda', '100 lmda')

ggplot(flow.lmda, aes(x=diff, color = par)) + 
  geom_density()+theme_bw() +
  #geom_vline(data=mu_lmda, aes(xintercept = grp.mean, color=par), linetype = "dashed" )+
  scale_color_manual(name = 'Sensitivity', values=col_values, labels = lmda.label)+
  labs(x = "Difference (m3/day)", y = " Density")+ 
  scale_x_continuous(limits = c(-120, 120), expand=c(0, 0)) +
  scale_y_continuous( limits = c(-1, 50), expand=c(0, 0)) +
  #ggtitle("Transimissivity")+
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = c(0.85, 0.80),
        panel.grid=element_blank(),
        axis.text.x = element_text(face = "bold"),
        axis.text.y = element_text(face = "bold"),
        legend.title = element_text(size=10, face="bold"),
        plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"))
ggsave("Hunt_sensitivity_lmda.png", dpi = 1000, width = 6, height = 5)
