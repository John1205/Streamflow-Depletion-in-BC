#Plot well resutls for each year
from IPython.display import clear_output, display
import time, sys, os, flopy, collections, csv, glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from numpy import array

path2mf = 'C:\\WRDAPP\\MF2005.1_12\\bin\\mf2005.exe'
file_path = 'C:\\Users\\User\\Documents\\BC-ENV_MODFLOW\\Peace\\Paleovalley_Sami'
os.chdir(file_path)
modelname ='SM_123'
mf = flopy.modflow.Modflow.load(modelname+"_MODFLOW_IN.NAM", verbose=True, check=False,  exe_name=path2mf)

#mf.wel.stress_period_data = well_input0
#change the file folder and names to get the other boundary conditions

#file_path = 'D:\\Peace_results\\data_sum'
file_path = 'D:\Peace_GW\data_sum'
stream = pd.read_csv(os.path.join(file_path, "Drain_BGT.csv"), header = 0, index_col = 0)
river = pd.read_csv(os.path.join(file_path, "River_BGT.csv"), header = 0, index_col = 0)

stream['years'] = stream.index/52-5
river['years'] = river.index/52-5

week = [0]*52*5 + [7]*52*30
pump = [0]*52*35
for k in range(30):
     pump[260 + 19+52*k]=19
     pump[260 + 20+52*k]=19
     pump[260 + 21+52*k]=19
     pump[260 + 22+52*k]=19
        
     pump[260 + 23+52*k]=108
     pump[260 + 24+52*k]=108
     pump[260 + 25+52*k]=108
     pump[260 + 26+52*k]=108
        
     pump[260 + 27+52*k]=214
     pump[260 + 28+52*k]=214
     pump[260 + 29+52*k]=214
     pump[260 + 30+52*k]=214
        
     pump[260 + 31+52*k]=171
     pump[260 + 32+52*k]=171
     pump[260 + 33+52*k]=171
     pump[260 + 34+52*k]=171
        
     pump[260 + 35+52*k]=84
     pump[260 + 36+52*k]=84
     pump[260 + 37+52*k]=84
     pump[260 + 38+52*k]=84
     
week_days = np.cumsum(week)
week_dates = pd.DataFrame(week_days, columns = ['days'])
week_dates['stress'] = week_dates.index
week_dates['pump'] = pump

stream_pump = stream.join(week_dates, how = 'outer')
river_pump = river.join(week_dates, how = 'outer')

#select the delpletion only occur for pumping season
stream_pump = stream_pump[stream_pump['pump']>0].reset_index()
river_pump = river_pump[river_pump['pump']>0].reset_index()

year = np.repeat(np.arange(1, 31), 20)
stream_pump['pump_year'] = year
river_pump['pump_year'] = year

stream_sum = stream_pump.groupby(['pump_year']).sum().reset_index()
stream_sum = stream_sum.drop(columns = ['index','PERCENT', 'years', 'days', 'stress', 'pump'])

river_pump = river_pump.groupby(['pump_year']).sum().reset_index()
river_pump = river_pump.drop(columns = ['index','PERCENT', 'years', 'days', 'stress', 'pump'])


ryear10 = river_pump[river_pump['pump_year']==1]
ryear20 = river_pump[river_pump['pump_year']==5]
ryear30 = river_pump[river_pump['pump_year']==20]

syear10 = stream_sum[stream_sum['pump_year']==1]
syear20 = stream_sum[stream_sum['pump_year']==5]
syear30 = stream_sum[stream_sum['pump_year']==20]

row_size = mf.dis.delc.array
col_size = mf.dis.delr.array
row_len = np.cumsum(row_size)
col_len = np.cumsum(col_size) 

well_input0 = pd.read_csv(os.path.join(file_path, "well_T.csv"))
"""pumping rate may: 19, June: 108, july: 214, Aug: 171, Sep: 84 """
pump = 0.84
colname = 16

##################################################################
#plot the postive and negetive values
SFD10=[]
for i in range(1, len(ryear10.columns)):
        no = int(ryear10.columns[i][colname:]) # change number 
        SFD10.append([well_input0['row'][well_input0['well_num'][no]], well_input0['col'][well_input0['well_num'][no]], ryear10.iloc[0, i]])

for i in range(1, len(syear10.columns)):
        no = int(syear10.columns[i][colname:]) # change number 
        SFD10.append([well_input0['row'][well_input0['well_num'][no]], well_input0['col'][well_input0['well_num'][no]], syear10.iloc[0, i]]) 
 
SFD10 = pd.DataFrame(SFD10, columns = ['row', 'col', 'q'])
#make two set of data, postive and negative 
SFD101 = SFD10[SFD10['q']>=0]
df10 = SFD101.groupby(['row', 'col']).sum().reset_index()
df10.columns = ['row', 'col', 'q']
df10_p=df10[df10['q']>=0]  # positive depletion
SFD10P = df10_p.values.tolist()

well_x10p = []
well_y10p = []
well_d10p = []
well_d101p = []

for i in range(len(SFD10P)):
    well_y10p.append(65400-row_len[int(SFD10P[i][0])])
    well_x10p.append(col_len[int(SFD10P[i][1])])
    well_d10p.append(SFD10P[i][2]/23.84)
    well_d101p.append(80)


SFD10N = pd.merge(df10_p, well_input0, how = 'outer')
SFD10N = SFD10N[SFD10N.isnull().any(axis=1)].reset_index()

well_x10n = []
well_y10n = []
well_d10n = []
well_d101n = []

for i in range(len(SFD10N)):
    well_y10n.append(65400-row_len[int(SFD10N['row'][i])])
    well_x10n.append(col_len[int(SFD10N['col'][i])])
    well_d10n.append(0)
    well_d101n.append(30)
#####################################################
SFD20=[]
for i in range(1, len(ryear20.columns)):
        no = int(ryear20.columns[i][colname:]) # change number 
        SFD20.append([well_input0['row'][well_input0['well_num'][no]], well_input0['col'][well_input0['well_num'][no]], ryear20.iloc[0, i]])

for i in range(1, len(syear20.columns)):
        no = int(syear20.columns[i][colname:]) # change number 
        SFD20.append([well_input0['row'][well_input0['well_num'][no]], well_input0['col'][well_input0['well_num'][no]], syear20.iloc[0, i]]) 
 
 
SFD20 = pd.DataFrame(SFD20, columns = ['row', 'col', 'q'])
SFD201 = SFD20[SFD20['q']>=0]
df20 = SFD201.groupby(['row', 'col']).sum().reset_index()
df20.columns = ['row', 'col', 'q']
df20_p=df20[df20['q']>0]  # positive depletion
SFD20P = df20_p.values.tolist()

well_x20p = []
well_y20p = []
well_d20p = []
well_d201p = []

for i in range(len(SFD20P)):
    well_y20p.append(65400-row_len[int(SFD20P[i][0])])
    well_x20p.append(col_len[int(SFD20P[i][1])])
    well_d20p.append(SFD20P[i][2]/23.84)
    well_d201p.append(80)   

SFD20N = pd.merge(df20_p, well_input0, how = 'outer')
SFD20N = SFD20N[SFD20N.isnull().any(axis=1)].reset_index()

well_x20n = []
well_y20n = []
well_d20n = []
well_d201n = []

for i in range(len(SFD20N)):
    well_y20n.append(65400-row_len[int(SFD20N['row'][i])])
    well_x20n.append(col_len[int(SFD20N['col'][i])])
    well_d20n.append(0)
    well_d201n.append(30) 
#####################################################
SFD30=[]
for i in range(1, len(ryear30.columns)):
        no = int(ryear30.columns[i][colname:]) # change number 
        SFD30.append([well_input0['row'][well_input0['well_num'][no]], well_input0['col'][well_input0['well_num'][no]], ryear30.iloc[0, i]])

for i in range(1, len(syear30.columns)):
        no = int(syear30.columns[i][colname:]) # change number 
        SFD30.append([well_input0['row'][well_input0['well_num'][no]], well_input0['col'][well_input0['well_num'][no]], syear30.iloc[0, i]]) 
 
SFD30 = pd.DataFrame(SFD30, columns = ['row', 'col', 'q'])
SFD301 = SFD30[SFD30['q']>=0]
df30 = SFD301.groupby(['row', 'col']).sum().reset_index()
df30.columns = ['row', 'col', 'q']
df30_p=df30[df30['q']>0]  # positive depletion
df30_n=df30[df30['q']<=0] # negative due to mass balance error

SFD30P = df30_p.values.tolist()


well_x30p = []
well_y30p = []
well_d30p = []
well_d301p = []
for i in range(len(SFD30P)):
    well_y30p.append(65400-row_len[int(SFD30P[i][0])])
    well_x30p.append(col_len[int(SFD30P[i][1])])
    well_d30p.append(SFD30P[i][2]/23.84)
    well_d301p.append(80)

SFD30N = pd.merge(df30_p, well_input0, how = 'outer')
SFD30N = SFD30N[SFD30N.isnull().any(axis=1)].reset_index()

well_x30n = []
well_y30n = []
well_d30n = []
well_d301n = []
for i in range(len(SFD30N)):
    well_y30n.append(65400-row_len[int(SFD30N['row'][i])])
    well_x30n.append(col_len[int(SFD30N['col'][i])])
    well_d30n.append(0)
    well_d301n.append(30)

###################################################################
hdobj = flopy.utils.HeadFile(modelname + '.hds')
times = hdobj.get_times()
head = hdobj.get_data(totim = times[-1])
thickness = flopy.utils.postprocessing.get_saturated_thickness(head, mf, nodata = ['-1e30'], per_idx=-1)
for i in range(20): 
    thickness[i][thickness[i] < 0] = 0 

active = mf.bas6.ibound.array[0]
active[active == -1] = 0
#np.unique(active)  
#array([0, 1])

#calculate water table below surface
#GW table  = layer thickness - saturated thickness

#1. calculate saturated thickness for active zones
thick1 = thickness[0]
thick={}
thick[0]= thickness[0]
for i in range(1, 20):
    thick[i]= (thick[i-1] + thickness[i])*active
"""    
for j in range(20):
    print(np.max(thick[j]))
"""    
#2. calculate layer thickness
nlay= 20
lthick = {}
for i in range(nlay): 
    lthick[i] = mf.dis.thickness[i,:, :]*active

laythick = {}    
laythick[0] = lthick[0] 
for i in range(1, nlay):
    laythick[i] = (laythick[i-1]+lthick[i])

#calculate water table
table = laythick[19]*active-thick[19]
##############################################
#read flow leakage file to plot stream status
baseflow = pd.read_csv(os.path.join(file_path, "boundary_leakage.csv"), index_col = 0)
#negetive Q is gaining stream
#postive Q is losing stream

stream_gain_x = []
stream_gain_y = []
stream_gain_q = []
stream_gain_q1 = []

stream_loss_x = []
stream_loss_y = []
stream_loss_q = []
stream_loss_q1 = []

stream_dry_x = []
stream_dry_y = []
stream_dry_q = []
stream_dry_q1 = []

for i in range(1, len(baseflow)):
    if baseflow['Q'][i] <0:
        stream_gain_y.append(65400-row_len[int(baseflow['row'][i])])
        stream_gain_x.append(col_len[int(baseflow['col'][i])])
        stream_gain_q.append(40)
        stream_gain_q1.append(0.1)   
            
for i in range(1, len(baseflow)):
    if baseflow['Q'][i] >0:
        stream_loss_y.append(65400-row_len[int(baseflow['row'][i])])
        stream_loss_x.append(col_len[int(baseflow['col'][i])])
        stream_loss_q.append(40)
        stream_loss_q1.append(0.1) 
        
for i in range(1, len(baseflow)):        
    if baseflow['Q'][i] ==0:
        stream_dry_y.append(65400-row_len[int(baseflow['row'][i])])
        stream_dry_x.append(col_len[int(baseflow['col'][i])])
        stream_dry_q.append(40)
        stream_dry_q1.append(0.1)

drain =[]
river = []    
        
"""
plot hydraulic head in the first figure
fig = plt.figure(figsize=(13, 9))
vmin = 0
vmax = 100
camp = 'Oranges_r'
#norm=matplotlib.colors.LogNorm()
aspect = 'auto'
ax1 = fig.add_subplot(2, 2, 1 , aspect = aspect)
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound(label  = 'ibound', color_noflow='White')
head = hdobj.get_data(totim = times[-1])[19]*active
levels = np.arange(200, 1200, 30)
contour_set = modelmap.contour_array(head,  levels =levels)
ax1.clabel(contour_set, inline = 1, fontsize  =14, fmt = '%1.0f')
plt.xticks([])
#plt.colorbar(hy, orientation = 'horizontal', shrink = 1)
ax1.set_title('Hydraulic Head')
"""
####################################################################
store_path = 'D:\\Peace_GW\\data_sum\\Figures'
fig = plt.figure(figsize=(13, 9))
aspect = 'equal'
ax1 = fig.add_subplot(2, 2, 1 , aspect = aspect)
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
hy = modelmap.plot_array(mf.dis.top.array, cmap = 'Greys', alpha = 1)
Boundary = modelmap.plot_ibound(label  = 'ibound', color_noflow='White')
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green', plotAll =True)
GHB = modelmap.plot_bc('GHB', label  = 'General head', plotAll =True)
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain', plotAll =True)
#plt.xticks([])
#plt.yticks([])
#plt.legend((gain, loss, dry), ("Gaining", "Losing","Ephemeral" ), ncol =1, loc='lower left',  fontsize=10, scatterpoints=60, frameon=False)
#plt.colorbar(hy, orientation = 'horizontal', shrink = 0.9)
#ax1.set_title('Stream Leakage')
#plt.savefig(os.path.join(store_path, "topography_colorbar.pdf"))  
plt.savefig(os.path.join(store_path, "Boundary_scale.jpg"), dpi = 1000) 

#########################################################################
store_path = 'D:\\Peace_GW\\data_sum\\Figures'
fig = plt.figure(figsize=(13, 9))
norm=matplotlib.colors.Normalize(vmin = 0, vmax = 100)
colors = ['g', 'c', 'y']
aspect = 'equal'
ax1 = fig.add_subplot(2, 2, 1 , aspect = aspect)
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
hy = modelmap.plot_array(table, cmap = 'Blues_r', alpha = 0.6)
Boundary = modelmap.plot_ibound(label  = 'ibound', color_noflow='White')
gain = ax1.scatter(stream_gain_x, stream_gain_y,  s= stream_gain_q1, marker = 'o',  label="Gaining", color=colors[0])
loss = ax1.scatter(stream_loss_x, stream_loss_y, s= stream_loss_q1, marker = 'o', label="Losing", color=colors[1])
dry = ax1.scatter(stream_dry_x, stream_dry_y, s= stream_dry_q1, marker = 'o', label="Ephmeral", color=colors[2])
#stream = ax1.scatter(stream_x, stream_y, c= stream_q, s= stream_q1, marker = 'o')
plt.xticks([])
plt.yticks([])
plt.legend((gain, loss, dry), ("Gaining", "Losing","Ephemeral" ), ncol =1, loc='lower left',  fontsize=10, scatterpoints=60, frameon=False)
#plt.colorbar(hy, orientation = 'horizontal', shrink = 0.9)
#ax1.set_title('Stream Leakage')
#plt.savefig(os.path.join(store_path, "stream_leakage_Blue_colorbar.pdf"))  
plt.savefig(os.path.join(store_path, "stream_leakage_Blue_legend.jpg"), dpi = 1000)  

#########################################################################
store_path = 'D:\\Peace_GW\\data_sum\\Figures'
fig = plt.figure(figsize=(13, 9))
vmin = 0
vmax = 100
camp = 'Reds'
ax2 = fig.add_subplot(2, 2, 2, aspect = aspect)
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
#hy = modelmap.plot_array(table, cmap = 'Greys', alpha = 0.8)
Boundary = modelmap.plot_ibound(label  = 'ibound', color_noflow='White')
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green', plotAll =True)
GHB = modelmap.plot_bc('GHB', label  = 'General head', plotAll =True)
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain', plotAll =True)
plt.scatter(well_x10p, well_y10p, c= well_d10p, s =well_d101p , marker = 'o', cmap = camp,  vmin=vmin, vmax=vmax, norm = norm)
#ax2.scatter(well_x10n, well_y10n, c= well_d10n, s =well_d101n, marker = 's', cmap = 'Greys_r')
plt.xticks([])
plt.yticks([])
#plt.scatter(well_x10, well_y10, c= well_d10, s = well_d101, marker = 'o', cmap = 'YlGn', vmin=0, vmax=100)
ax2.set_title('Year 1')
plt.savefig(os.path.join(store_path, "peace_baseflow_yr1.jpg"), dpi = 1000)  

#########################################################################
store_path = 'D:\\Peace_GW\\data_sum\\Figures'
#fig = plt.figure(figsize=(13, 9))
ax3 = fig.add_subplot(2, 2, 3, aspect = aspect)
modelmap = flopy.plot.ModelMap(model=mf, layer =0 )
#hy = modelmap.plot_array(table, cmap = 'Greys', alpha = 0.8)
Boundary = modelmap.plot_ibound(label  = 'ibound', color_noflow='White')
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green', plotAll =True)
GHB = modelmap.plot_bc('GHB', label  = 'General head', plotAll =True)
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain', plotAll =True)
plt.scatter(well_x20p, well_y20p, c= well_d20p,  s =well_d201p ,marker = 'o', cmap = camp,  vmin=vmin, vmax=vmax, norm = norm)
#ax3.scatter(well_x20n, well_y20n, c= well_d20n,  s =well_d201n ,marker = 's', cmap = 'Greys_r')
plt.yticks([])
plt.xticks([])
ax3.set_title('Year 5')
plt.savefig(os.path.join(store_path, "peace_baseflow_yr5.jpg"), dpi = 1000)  

#########################################################################
store_path = 'D:\\Peace_GW\\data_sum\\Figures'
fig = plt.figure(figsize=(13, 9))
ax4 = fig.add_subplot(2, 2, 4, aspect = aspect)
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
#hy = modelmap.plot_array(table, cmap = 'Greys', alpha = 0.8)
Boundary = modelmap.plot_ibound(label  = 'ibound', color_noflow='White' )
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green', plotAll =True)
GHB = modelmap.plot_bc('GHB', label  = 'General head', plotAll =True)
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain', plotAll =True)
plt.scatter(well_x30p, well_y30p, c= well_d30p,  s =well_d301p, marker = 'o', cmap = camp,  vmin=vmin, vmax=vmax, norm = norm)
#ax4.scatter(well_x30n, well_y30n, c= well_d30n,  s =well_d301n, marker = 's', cmap = 'Greys_r')
plt.yticks([])
plt.xticks([])
ax4.set_title('Year 20')
plt.savefig(os.path.join(store_path, "peace_baseflow_yr20.jpg"), dpi = 1000)  

plt.subplots_adjust(right=0.6)
cax = plt.axes([0.62, 0.13, 0.015, 0.75])
plt.colorbar(cax=cax)
plt.savefig(os.path.join(store_path, "peace_baseflow_yr20.pdf"))  


store_path = 'D:\\Peace_GW\\data_sum'
#plt.savefig(os.path.join(store_path, "baseflow_colorbar.jpg"), dpi = 1000)  
plt.savefig(os.path.join(store_path, "peace_baseflow_colorbar.pdf"))  

#############################################################
"""plot MAE for wells"""
mae_file = 'D:\\Peace_GW\\data_sum'
well = pd.read_csv(os.path.join(mae_file, "peace_mae_wells.csv"), index_col = 0)

well_loc = pd.merge(well, well_input0, how = 'inner')

well_loc_glover = well_loc[well_loc['model']== 'Glover'].reset_index()
well_loc_hunt = well_loc[well_loc['model']== 'Hunt'].reset_index()

glover_x = []
glover_y = []
glover_mae = []

for i in range(len(well_loc_glover)):        
        glover_y .append(65400-row_len[int(well_loc_glover['row'][i])])
        glover_x.append(col_len[int(well_loc_glover['col'][i])])
        glover_mae.append(well_loc_glover['MAE'][i])
      
hunt_x = []
hunt_y = []
hunt_mae = []
hunt_s =[]
for i in range(len(well_loc_hunt)):        
        hunt_y.append(65400-row_len[int(well_loc_hunt['row'][i])])
        hunt_x.append(col_len[int(well_loc_hunt['col'][i])])
        hunt_mae.append(well_loc_hunt['MAE'][i])
        hunt_s.append(80)

vmin = 0
vmax = 30
camp = 'Reds'
norm=matplotlib.colors.Normalize(vmin = 0, vmax = 30)
fig = plt.figure(figsize=(13, 9))
ax4 = fig.add_subplot(2, 2, 1, aspect = 'equal')
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound(label  = 'ibound', color_noflow='White')
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green')
GHB = modelmap.plot_bc('GHB', label  = 'General head')
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain')
plt.scatter(hunt_x, hunt_y, c= hunt_mae, s = hunt_s, marker = 'o', cmap = camp,  vmin=vmin, vmax=vmax, norm = norm)
plt.xticks([])
plt.yticks([])
plt.savefig(os.path.join(mae_file, "Hunt_MAE_spatial.jpg"), dpi= 1000)



vmin = 0
vmax = 30
camp = 'Reds'
norm=matplotlib.colors.Normalize(vmin = 0, vmax = 30)
fig = plt.figure(figsize=(13, 9))
ax4 = fig.add_subplot(2, 2, 1, aspect = 'equal')
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound(label  = 'ibound', color_noflow='White')
#RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green')
#GHB = modelmap.plot_bc('GHB', label  = 'General head')
#STR = modelmap.plot_bc('STR', label  = 'Streams',color = 'purple')
#DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain')
plt.scatter(glover_x, glover_y, c= glover_mae, s = hunt_s, marker = 'o', cmap = camp,  vmin=vmin, vmax=vmax, norm = norm)
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.savefig(os.path.join(mae_file, "spatial_MAE_coloarbar.pdf"))

plt.savefig(os.path.join(mae_file, "Glover_MAE_spatial.jpg"), dpi= 1000)
    