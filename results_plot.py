#Peace regions results plot

from IPython.display import clear_output, display
import time, os, sys, flopy,math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection
import matplotlib.colors as colors

path2mf = 'C:\\WRDAPP\\MF2005.1_12\\bin\\mf2005.exe'
file_path = 'C:\\Users\\User\\Documents\\BC-ENV_MODFLOW\\Peace\\Paleovalley_Sami'
os.chdir(file_path)
modelname ='SM_123'
mf = flopy.modflow.Modflow.load(modelname+"_MODFLOW_IN.NAM", verbose=True, check=False,  exe_name=path2mf)

file_loc = 'D:\\Peace_results\\data_sum'
r2 = pd.read_csv(os.path.join(file_loc, "well_num_R2.csv"), header =0)
r2.head()
r2.glover =pd.DataFrame(r2.loc[r2['Model']== 'Glover'])
r2.hunt =pd.DataFrame(r2.loc[r2['Model']== 'Hunt'])

r2.glover.drain =pd.DataFrame(r2.glover.loc[r2.glover['seg']== 'Drain']).reset_index()
r2.glover.river =pd.DataFrame(r2.glover.loc[r2.glover['seg']== 'River']).reset_index()

r2.hunt.drain =pd.DataFrame(r2.hunt[r2.hunt['seg']== 'Drain']).reset_index()
r2.hunt.river =pd.DataFrame(r2.hunt[r2.hunt['seg']== 'River']).reset_index()

##############################################################
#plot the short distance performance of each flow boundary

row_size = mf.dis.delc.array
col_size = mf.dis.delr.array
row_len = np.cumsum(row_size)
col_len = np.cumsum(col_size) 

well_x = []
well_y = []
well_d = []

#well_d11 = []
for i in range(len(r2.glover.drain)):
    well_y.append(65400-row_len[r2.glover.drain['well_row'][i]])
    well_x.append(col_len[r2.glover.drain['well_col'][i]])
    well_d.append(r2.glover.drain['min.r2'][i])
    
well_x1 = []
well_y1 = []
well_d1 = []

#well_d11 = []
for i in range(len(r2.glover.river)):
    well_y1.append(65400-row_len[r2.glover.river['well_row'][i]])
    well_x1.append(col_len[r2.glover.river['well_col'][i]])
    well_d1.append(r2.glover.river['min.r2'][i])   
    
well_x2 = []
well_y2 = []
well_d2 = []

#well_d11 = []
for i in range(len(r2.hunt.drain)):
    well_y2.append(65400-row_len[r2.hunt.drain['well_row'][i]])
    well_x2.append(col_len[r2.hunt.drain['well_col'][i]])
    well_d2.append(r2.hunt.drain['min.r2'][i])   
    
well_x3 = []
well_y3 = []
well_d3 = []

#well_d11 = []
for i in range(len(r2.hunt.river)):
    well_y3.append(65400-row_len[r2.hunt.river['well_row'][i]])
    well_x3.append(col_len[r2.hunt.river['well_col'][i]])
    well_d3.append(r2.hunt.river['min.r2'][i])       
     
    
fig = plt.figure(figsize=(14, 10))
cmap = 'viridis'
vmin =0
vmax =1
ax = fig.add_subplot(2, 2, 1, aspect = "equal")
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound()
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green', plotAll=True)
GHB = modelmap.plot_bc('GHB', label  = 'General head', plotAll=True)
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain', plotAll=True)
plt.scatter(well_x, well_y, c= well_d, marker = 'o', cmap = cmap, vmin=vmin, vmax=vmax)
ax.set_title("Glover Drain")

ax = fig.add_subplot(2, 2, 2, aspect = "equal")
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound()
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green', plotAll=True)
GHB = modelmap.plot_bc('GHB', label  = 'General head', plotAll=True)
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain', plotAll=True)
plt.scatter(well_x1, well_y1, c= well_d1, marker = 'o', cmap = cmap, vmin=vmin, vmax=vmax)
ax.set_title("Glover River")

ax = fig.add_subplot(2, 2, 3, aspect = "equal")
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound()
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green', plotAll=True)
GHB = modelmap.plot_bc('GHB', label  = 'General head', plotAll=True)
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain', plotAll=True)
plt.scatter(well_x2, well_y2, c= well_d2, marker = 'o', cmap = cmap, vmin=vmin, vmax=vmax)
ax.set_title("Hunt Drain")

ax = fig.add_subplot(2, 2, 4, aspect = "equal")
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound()
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green', plotAll=True)
GHB = modelmap.plot_bc('GHB', label  = 'General head', plotAll=True)
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain', plotAll=True)
plt.scatter(well_x3, well_y3, c= well_d3, marker = 'o', cmap = cmap, vmin=vmin, vmax=vmax)
ax.set_title("Hunt River")

cax = plt.axes([0.68, 0.12, 0.02, 0.76])
plt.colorbar(cax=cax)
plt.subplots_adjust(right=0.65)
cax.set_ylabel('R2', rotation=270, size = 14)
cax.yaxis.set_label_coords(4, 0.55)
#fig.savefig(os.path.join(file_loc,"Spatial_R2.jpg"), dpi = 1000)    
#####################################################################################  

#seasonal plots time series
river = pd.read_csv(os.path.join(file_loc, "River_BGT.csv"), header = 0, index_col = 0)
drain = pd.read_csv(os.path.join(file_loc, "Drain_BGT.csv"), header = 0, index_col = 0)
day = 38

dyear5 = drain.loc[day+52*(4)+260,]
dyear10 = drain.loc[day+52*(9)+260,]
dyear20 = drain.loc[day+52*(19)+260,]
dyear30 = drain.loc[day+52*(29)+260,]
dyear40 = drain.loc[day+52*(39)+260,]

ryear5 = river.loc[day+52*(4)+260,]
ryear10 = river.loc[day+52*(9)+260,]
ryear20 = river.loc[day+52*(19)+260,]
ryear30 = river.loc[day+52*(29)+260,]
ryear40 = river.loc[day+52*(39)+260,]

well_input0 = pd.read_csv(os.path.join(file_loc, "well_T_copy.csv"), header =0, index_col=0)
T = -500
colname = 16

SFD5=[]
for i in range(1, len(drain.columns)):
    if dyear5[i] > T:
        no = int(drain.columns[i][colname:])
        SFD5.append([well_input0['row'][no], well_input0['col'][no], dyear5[i]])

for i in range(1, len(river.columns)):
    if ryear5[i] > T:
        no = int(river.columns[i][colname:]) # change number 
        SFD5.append([well_input0['row'][no], well_input0['col'][no], ryear5[i]])

# remove duplicats in SFD ONLY Keep th maximum 
   
SFD5 = pd.DataFrame(SFD5, columns = ['row', 'col', 'q'])
df = SFD5.groupby(['row', 'col']).sum().sum(
        level=['row', 'col']).unstack(['row', 'col']).fillna(0).reset_index()
df.columns = ['level','row', 'col', 'q']
SFD5 = df.drop(columns = ['level']).values.tolist()
   
well_x5 = []
well_y5 = []
well_d5 = []
well_d51 = []

#well_d11 = []
for i in range(len(SFD5)):
    well_y5.append(65400-row_len[int(SFD5[i][0])])
    well_x5.append(col_len[int(SFD5[i][1])])
    well_d5.append(SFD5[i][2])
    well_d51.append(abs(SFD5[i][2]))

#######################################################################################
SFD10=[]
for i in range(1, len(drain.columns)):
    if dyear10[i] > T:
        no = int(drain.columns[i][colname:])
        SFD10.append([well_input0['row'][no], well_input0['col'][no], dyear10[i]])

for i in range(1, len(river.columns)):
    if ryear10[i] > T:
        no = int(river.columns[i][colname:]) # change number 
        SFD10.append([well_input0['row'][no], well_input0['col'][no], ryear10[i]])

# remove duplicats in SFD ONLY Keep th maximum 
   
SFD10 = pd.DataFrame(SFD10, columns = ['row', 'col', 'q'])
df = SFD10.groupby(['row', 'col']).sum().sum(
        level=['row', 'col']).unstack(['row', 'col']).fillna(0).reset_index()
df.columns = ['level','row', 'col', 'q']
SFD10 = df.drop(columns = ['level']).values.tolist()
   
well_x10 = []
well_y10 = []
well_d10 = []
well_d101 = []

#well_d11 = []
for i in range(len(SFD10)):
    well_y10.append(65400-row_len[int(SFD10[i][0])])
    well_x10.append(col_len[int(SFD10[i][1])])
    well_d10.append(SFD10[i][2])
    well_d101.append(abs(SFD10[i][2]))

#######################################################################################
SFD20=[]
for i in range(1, len(drain.columns)):
    if dyear20[i] > T:
        no = int(drain.columns[i][colname:])
        SFD20.append([well_input0['row'][no], well_input0['col'][no], dyear20[i]])

for i in range(1, len(river.columns)):
    if ryear20[i] > T:
        no = int(river.columns[i][colname:]) # change number 
        SFD20.append([well_input0['row'][no], well_input0['col'][no], ryear20[i]])

# remove duplicats in SFD ONLY Keep th maximum 
   
SFD20 = pd.DataFrame(SFD20, columns = ['row', 'col', 'q'])
df = SFD20.groupby(['row', 'col']).sum().sum(
        level=['row', 'col']).unstack(['row', 'col']).fillna(0).reset_index()
df.columns = ['level','row', 'col', 'q']
SFD20 = df.drop(columns = ['level']).values.tolist()
   
well_x20 = []
well_y20 = []
well_d20 = []
well_d201 = []

#well_d11 = []
for i in range(len(SFD20)):
    well_y20.append(65400-row_len[int(SFD20[i][0])])
    well_x20.append(col_len[int(SFD20[i][1])])
    well_d20.append(SFD20[i][2])
    well_d201.append(abs(SFD20[i][2]))
    
#######################################################################################
SFD30=[]
for i in range(1, len(drain.columns)):
    if dyear30[i] > T:
        no = int(drain.columns[i][colname:])
        SFD30.append([well_input0['row'][no], well_input0['col'][no], dyear30[i]])

for i in range(1, len(river.columns)):
    if ryear30[i] > T:
        no = int(river.columns[i][colname:]) # change number 
        SFD30.append([well_input0['row'][no], well_input0['col'][no], ryear30[i]])

# remove duplicats in SFD ONLY Keep th maximum 
   
SFD30 = pd.DataFrame(SFD30, columns = ['row', 'col', 'q'])
df = SFD30.groupby(['row', 'col']).sum().sum(
        level=['row', 'col']).unstack(['row', 'col']).fillna(0).reset_index()
df.columns = ['level','row', 'col', 'q']
SFD30 = df.drop(columns = ['level']).values.tolist()
   
well_x30 = []
well_y30 = []
well_d30 = []
well_d301 = []

#well_d11 = []
for i in range(len(SFD30)):
    well_y30.append(65400-row_len[int(SFD30[i][0])])
    well_x30.append(col_len[int(SFD30[i][1])])
    well_d30.append(SFD30[i][2])
    well_d301.append(abs(SFD30[i][2]))
    
#######################################################################################
SFD40=[]
for i in range(1, len(drain.columns)):
    if dyear40[i] > T:
        no = int(drain.columns[i][colname:])
        SFD40.append([well_input0['row'][no], well_input0['col'][no], dyear40[i]])

for i in range(1, len(river.columns)):
    if ryear40[i] > T:
        no = int(river.columns[i][colname:]) # change number 
        SFD40.append([well_input0['row'][no], well_input0['col'][no], ryear40[i]])

# remove duplicats in SFD ONLY Keep th maximum 
   
SFD40 = pd.DataFrame(SFD40, columns = ['row', 'col', 'q'])
df = SFD40.groupby(['row', 'col']).sum().sum(
        level=['row', 'col']).unstack(['row', 'col']).fillna(0).reset_index()
df.columns = ['level','row', 'col', 'q']
SFD40 = df.drop(columns = ['level']).values.tolist()
   
well_x40 = []
well_y40 = []
well_d40 = []
well_d401 = []

#well_d11 = []
for i in range(len(SFD40)):
    well_y40.append(65400-row_len[int(SFD40[i][0])])
    well_x40.append(col_len[int(SFD40[i][1])])
    well_d40.append(SFD40[i][2])
    well_d401.append(abs(SFD40[i][2]))

###################################################################################
fig = plt.figure(figsize=(14, 12))
vmin = 0
vmax = 150
camp = 'Oranges'
aspect = 'auto'
norm=colors.LogNorm()
ax = fig.add_subplot(2, 2, 1 , aspect = aspect)
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound()
file_path = 'D:\\Peace_Pumping_baseline'
hdobj = flopy.utils.HeadFile(os.path.join(file_path, 'SM_123_MODFLOW_IN.hds'))
times = hdobj.get_times()
head = hdobj.get_data(totim = times[-1])[19]
levels = np.arange(200, 760, 30)
contour_set = modelmap.contour_array(head,  levels =levels)
ax.clabel(contour_set, inline = 1, fontsize  = 9, fmt = '%1.0f')
ax.set_title('Hydraulic heads')
#########################################################################
ax = fig.add_subplot(2, 2, 2, aspect = aspect)
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound(label  = 'ibound')
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green')
GHB = modelmap.plot_bc('GHB', label  = 'General head')
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain')
plt.scatter(well_x5, well_y5, c= well_d5,  marker = 'o', cmap = camp, vmin=vmin, vmax=vmax)
#plt.scatter(well_x50, well_y50, c= well_d50, s = well_d501,  marker = 'o', cmap = 'YlGn', vmin=0, vmax=100)
ax.set_title('Year 5')
#########################################################################
ax = fig.add_subplot(2, 2, 3, aspect = aspect)
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound(label  = 'ibound')
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green')
GHB = modelmap.plot_bc('GHB', label  = 'General head')
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain')
plt.scatter(well_x10, well_y10, c= well_d10, marker = 'o', cmap = camp,  vmin=vmin, vmax=vmax)
#plt.scatter(well_x10, well_y10, c= well_d10, s = well_d101, marker = 'o', cmap = 'YlGn', vmin=0, vmax=100)
ax.set_title('Year 10')
#########################################################################
ax = fig.add_subplot(2, 2, 4, aspect = aspect)
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound(label  = 'ibound')
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green')
GHB = modelmap.plot_bc('GHB', label  = 'General head')
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain')
plt.scatter(well_x20, well_y20, c= well_d20,  marker = 'o', cmap = camp,  vmin=vmin, vmax=vmax)
#plt.scatter(well_x20, well_y20, c= well_d20,  s = well_d201, marker = 'o', cmap = 'YlGn', vmin=0, vmax=100)
ax.set_title('Year 20')




#########################################################################
ax = fig.add_subplot(3, 2, 5, aspect = aspect)
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound(label  = 'ibound')
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green')
GHB = modelmap.plot_bc('GHB', label  = 'General head')
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain')
plt.scatter(well_x30, well_y30, c= well_d30,  marker = 'o', cmap = camp,  vmin=vmin, vmax=vmax)
#plt.scatter(well_x30, well_y30, c= well_d30,  s = well_d301, marker = 'o', cmap = 'YlGn', vmin=0, vmax=100)
ax.set_title('Year 30')
#########################################################################
ax = fig.add_subplot(3, 2, 6, aspect = aspect)
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound(label  = 'ibound')
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green')
GHB = modelmap.plot_bc('GHB', label  = 'General head')
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain')
#plt.scatter(well_x30, well_y30, c= well_d30, marker = 'o', cmap = 'brg', vmin=0, vmax=100)
plt.scatter(well_x40, well_y40, c= well_d40,  marker = 'o', cmap = camp,  vmin=vmin, vmax=vmax)
#plt.scatter(well_x40, well_y40, c= well_d40, s = well_d401,  marker = 'o', cmap = 'YlGn',  vmin=0, vmax=100)
ax.set_title('Year 40')
plt.subplots_adjust(right=0.6)
cax = plt.axes([0.62, 0.12, 0.015, 0.76])
plt.colorbar(cax=cax)
cax.set_ylabel('Depletion (m3/day)', rotation=270, size = 14)
cax.yaxis.set_label_coords(3.6, 0.56)
fig.suptitle('Depletion in September', fontsize=16, x = 0.4, y= 0.93)
plt.savefig(os.path.join(file_loc, "peace_stream_depletion.jpg"), dpi = 1000, pad_inches=0.1)    

############################################################################################
file_loc = 'D:\\Peace_GW\\data_sum'
r2 = pd.read_csv(os.path.join(file_loc, "well_num_R2.csv"), header =0)
r2.head()
r2.glover =pd.DataFrame(r2.loc[r2['Model']== 'Glover']).reset_index()
r2.hunt =pd.DataFrame(r2.loc[r2['Model']== 'Hunt']).reset_index()

row_size = mf.dis.delc.array
col_size = mf.dis.delr.array
row_len = np.cumsum(row_size)
col_len = np.cumsum(col_size) 

well_x = []
well_y = []
well_d = []
well_d0 = []
#well_d11 = []
for i in range(len(r2.glover)):
    well_y.append(65400-row_len[r2.glover['row'][i]])
    well_x.append(col_len[r2.glover['col'][i]])
    well_d.append(r2.glover['r2'][i])
    well_d0.append(100)
    
well_x1 = []
well_y1 = []
well_d1 = []
well_d10 = []
#well_d11 = []
for i in range(len(r2.hunt)):
    well_y1.append(65400-row_len[r2.hunt['row'][i]])
    well_x1.append(col_len[r2.hunt['col'][i]])
    well_d1.append(r2.hunt['r2'][i])   
    well_d10.append(100)
    
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(1, 2, 1, aspect = "auto")
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound(label  = 'ibound', color_noflow='White',)
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green', plotAll=True)
GHB = modelmap.plot_bc('GHB', label  = 'General head', plotAll=True)
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain', plotAll=True)
plt.scatter(well_x, well_y, c= well_d, s = well_d0,  marker = 'o', cmap = 'Reds', vmin=0, vmax=1)
#plt.scatter(well_x50, well_y50, c= well_d50, s = well_d501,  marker = 'o', cmap = 'YlGn', vmin=0, vmax=100)
ax.set_title('Glover Model')

ax = fig.add_subplot(1, 2, 2, aspect = "auto")
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound(label  = 'ibound', color_noflow='White')
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green', plotAll=True)
GHB = modelmap.plot_bc('GHB', label  = 'General head', plotAll=True)
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain', plotAll=True)
plt.scatter(well_x1, well_y1, c= well_d1, s = well_d10,  marker = 'o', cmap = 'Reds', vmin=0, vmax=1)
#plt.scatter(well_x50, well_y50, c= well_d50, s = well_d501,  marker = 'o', cmap = 'YlGn', vmin=0, vmax=100)
ax.set_title('Hunt Model')
plt.subplots_adjust(right=0.78)
cax = plt.axes([0.80, 0.22, 0.018, 0.56])
cax.set_ylabel('R2', rotation=270, size = 14)
plt.colorbar(cax=cax)
plt.savefig(os.path.join(file_loc, "R2_wells.jpg"), dpi = 1000)
############################################################################################
file_loc = 'D:\\Peace_results\\data_sum'
r2_glover = pd.read_csv(os.path.join(file_loc, "Glover_wells_R2.csv"), header =0)
r2_hunt = pd.read_csv(os.path.join(file_loc, "Hunt_wells_R2.csv"), header =0)

row_size = mf.dis.delc.array
col_size = mf.dis.delr.array
row_len = np.cumsum(row_size)
col_len = np.cumsum(col_size) 

well_T = pd.read_csv(os.path.join(file_loc, "well_T.csv"), index_col=0)

well_x = []
well_y = []
well_d = []
well_d0 = []
#well_d11 = []
for i in range(len(r2_glover)):
    well_y.append(65400-row_len[well_T['row'][r2_glover['well_num'][i]]])
    well_x.append(col_len[well_T['col'][r2_glover['well_num'][i]]])
    well_d.append(r2_glover['r2'][i])
    well_d0.append(80)
    
   
well_x1 = []
well_y1 = []
well_d1 = []
well_d10 = []
#well_d11 = []
for i in range(len(r2_hunt)):
    well_y1.append(65400-row_len[well_T['row'][r2_hunt['well_num'][i]]])
    well_x1.append(col_len[well_T['col'][r2_hunt['well_num'][i]]])
    well_d1.append(r2_hunt['r2'][i])
    well_d10.append(80)
    
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(1, 2, 1, aspect = "auto")
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound(label  = 'ibound', color_noflow='White')
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green', plotAll=True)
GHB = modelmap.plot_bc('GHB', label  = 'General head', plotAll=True)
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain', plotAll=True)
plt.scatter(well_x, well_y, c= well_d, s = well_d0,  marker = 'o', cmap = 'Reds', vmin=0, vmax=1)
ax.set_title('Glover Model')

ax = fig.add_subplot(1, 2, 2, aspect = "auto")
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound(label  = 'ibound', color_noflow='White')
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green', plotAll=True)
GHB = modelmap.plot_bc('GHB', label  = 'General head', plotAll=True)
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain', plotAll=True)
plt.scatter(well_x1, well_y1, c= well_d1, s = well_d10,  marker = 'o', cmap = 'Reds', vmin=0, vmax=1)
plt.yticks([])
#plt.scatter(well_x50, well_y50, c= well_d50, s = well_d501,  marker = 'o', cmap = 'YlGn', vmin=0, vmax=100)
ax.set_title('Hunt Model')
plt.subplots_adjust(right=0.78)
cax = plt.axes([0.80, 0.22, 0.018, 0.56])
cax.set_ylabel('R2', rotation=270, size = 14)
plt.colorbar(cax=cax)
plt.savefig(os.path.join(file_loc, "R2_wells_Two.jpg"), dpi = 1000)