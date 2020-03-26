 """
Plot well resutls
"""
from IPython.display import clear_output, display
import time, sys, os, flopy, collections, csv, glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from numpy import array

path2mf = 'C:\\WRDAPP\\MF2005.1_12\\bin\\mf2005.exe'
fpath = 'C:\\Users\\User\\Documents\\BC-ENV_MODFLOW\\BX Creek Model'
model_ws = os.chdir(fpath)

modelname = 'BX_VERNON2'
mf = flopy.modflow.Modflow.load(modelname +"_MODFLOW_IN.nam", 
                                verbose=True,  check=True, 
                                exe_name=path2mf)

df=pd.read_csv('Drain_stress.csv', sep=',',header=None)
drn_dis = df.values
mf.drn.stress_period_data = drn_dis

#mf.wel.stress_period_data = well_input0
#change the file folder and names to get the other boundary conditions

file_path = 'D:\\BX_Results_GW_Tables\\Baseline'
constant = pd.read_csv(os.path.join(file_path, "Constant_bgt%.csv"), header = 0, index_col = 0)
stream = pd.read_csv(os.path.join(file_path, "stream_BGT%.csv"), header = 0, index_col = 0)
river = pd.read_csv(os.path.join(file_path, "River_BGT%.csv"), header = 0, index_col = 0)

#constant.set_index('no', inplace = True)
#constant.PERCENT
#constant[1:5]
#constant.loc[1,]
"""
may = 22
jun = 26
jul = 30
aug = 34
sep = 38
"""

day = 38
#cyear1 = constant.loc[day+52*(0)+260,]
cyear10 = constant.loc[day+52*(9)+260,]
cyear20 = constant.loc[day+52*(19)+260,]
cyear30 = constant.loc[day+52*(29)+260,]
#cyear40 = constant.loc[day+52*(39)+260,]
#cyear50 = constant.loc[day+52*(49)+260,]

#ryear1 = river.loc[day+52*(0)+260,]
ryear10 = river.loc[day+52*(9)+260,]
ryear20 = river.loc[day+52*(19)+260,]
ryear30 = river.loc[day+52*(29)+260,]
#ryear40 = river.loc[day+52*(39)+260,]
#ryear50 = river.loc[day+52*(49)+260,]

#syear1 = stream.loc[day+52*(0)+260,]
syear10 = stream.loc[day+52*(9)+260,]
syear20 = stream.loc[day+52*(19)+260,]
syear30 = stream.loc[day+52*(29)+260,]
#syear40 = stream.loc[day+52*(39)+260,]
#syear50 = stream.loc[day+52*(49)+260,]


"""pumping rate may: 19, June: 108, july: 214, Aug: 171, Sep: 84 """
pump = 0.84
colname = 16

T = 2
SFD10=[]
for i in range(1, len(constant.columns)):
    if cyear10[i] > T:
        no = int(constant.columns[i][colname:])
        SFD10.append([well_input0[no][1], well_input0[no][2], cyear10[i]])

for i in range(1, len(river.columns)):
    if ryear10[i] > T:
        no = int(river.columns[i][colname:]) # change number 
        SFD10.append([well_input0[no][1], well_input0[no][2], ryear10[i]])

for i in range(1, len(stream.columns)):
    if syear10[i] > T:
        no = int(stream.columns[i][colname:]) # change number 
        SFD10.append([well_input0[no][1], well_input0[no][2], syear10[i]]) 

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
    well_y10.append((327-SFD10[i][0])*50)
    well_x10.append(SFD10[i][1]*50)
    well_d10.append(SFD10[i][2])
    well_d101.append(80)

##########################################################################
SFD20=[]
for i in range(1, len(constant.columns)):
    if cyear20[i] > T:
        no = int(constant.columns[i][colname:])
        SFD20.append([well_input0[no][1], well_input0[no][2], cyear20[i]])

for i in range(1, len(river.columns)):
    if ryear20[i] > T:
        no = int(river.columns[i][colname:])
        SFD20.append([well_input0[no][1], well_input0[no][2], ryear20[i]])

for i in range(1, len(stream.columns)):
    if syear20[i] > T:
        no = int(stream.columns[i][colname:])
        SFD20.append([well_input0[no][1], well_input0[no][2], syear20[i]])

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
    well_y20.append((327-SFD20[i][0])*50)
    well_x20.append(SFD20[i][1]*50)
    well_d20.append(SFD20[i][2])
    well_d201.append(80)

##########################################################################
SFD30=[]
for i in range(1, len(constant.columns)):
    if cyear30[i]> T:
        no = int(constant.columns[i][colname:])
        SFD30.append([well_input0[no][1], well_input0[no][2], cyear30[i]])

for i in range(1, len(river.columns)):
    if ryear30[i] > T:
        no = int(river.columns[i][colname:])
        SFD30.append([well_input0[no][1], well_input0[no][2], ryear30[i]])

for i in range(1, len(stream.columns)):
    if syear30[i] > T:
        no = int(stream.columns[i][colname:])
        SFD30.append([well_input0[no][1], well_input0[no][2], syear30[i]]) 

SFD30 = pd.DataFrame(SFD30, columns = ['row', 'col', 'q'])
df = SFD30.groupby(['row', 'col']).sum().sum(
        level=['row', 'col']).unstack(['row', 'col']).fillna(0).reset_index()
df.columns = ['level','row', 'col', 'q']
SFD30 = df.drop(columns = ['level']).values.tolist()    
    
well_x30 = []
well_y30 = []
well_d30 = []
well_d301 = []
for i in range(len(SFD30)):
    well_y30.append((327-SFD30[i][0])*50)
    well_x30.append(SFD30[i][1]*50)
    well_d30.append(SFD30[i][2])
    well_d301.append(80)

##########################################################################
SFD40=[]
cwell_num = []
rwell_num = []
swell_num = []

for i in range(1, len(constant.columns)):
    if cyear40[i] > T:
        no = int(constant.columns[i][colname:])
        SFD40.append([well_input0[no][1], well_input0[no][2], cyear40[i]])
        cwell_num.append(no) 

for i in range(1, len(river.columns)):
    if ryear40[i] > T:
        no = int(river.columns[i][colname:])
        SFD40.append([well_input0[no][1], well_input0[no][2], ryear40[i]])
        rwell_num.append(no) 

for i in range(1, len(stream.columns)):
    if syear40[i] > T:
        no = int(stream.columns[i][colname:])
        SFD40.append([well_input0[no][1], well_input0[no][2], syear40[i]])        
        swell_num.append(no)

SFD40 = pd.DataFrame(SFD40, columns = ['row', 'col', 'q'])
df = SFD40.groupby(['row', 'col']).sum().sum(
        level=['row', 'col']).unstack(['row', 'col']).fillna(0).reset_index()
df.columns = ['level','row', 'col', 'q']
SFD40 = df.drop(columns = ['level']).values.tolist()   

well_x40 = []
well_y40 = []
well_d40 = []
well_d401 = []
for i in range(len(SFD40)):
    well_y40.append((327-SFD40[i][0])*50)
    well_x40.append(SFD40[i][1]*50)
    well_d40.append(SFD40[i][2])
    well_d401.append(abs(SFD40[i][2]))

#####################################################################
SFD50=[]
for i in range(1, len(constant.columns)):
    if cyear50[i]> T:
        no = int(constant.columns[i][colname:])
        SFD50.append([well_input0[no][1], well_input0[no][2], cyear50[i]])

for i in range(1, len(river.columns)):
    if ryear50[i] > T:
        no = int(river.columns[i][colname:]) # change number 
        SFD50.append([well_input0[no][1], well_input0[no][2], ryear50[i]])

for i in range(1, len(stream.columns)):
    if syear50[i]> T:
        no = int(stream.columns[i][colname:]) # change number 
        SFD50.append([well_input0[no][1], well_input0[no][2], syear50[i]])  

SFD50 = pd.DataFrame(SFD50, columns = ['row', 'col', 'q'])
df = SFD50.groupby(['row', 'col']).sum().sum(
        level=['row', 'col']).unstack(['row', 'col']).fillna(0).reset_index()
df.columns = ['level','row', 'col', 'q']
SFD50 = df.drop(columns = ['level']).values.tolist()
    
well_x50 = []
well_y50 = []
well_d50 = []
well_d501 = []
#well_d11 = []
for i in range(len(SFD50)):
    well_y50.append((327-SFD50[i][0])*50)
    well_x50.append(SFD50[i][1]*50)
    well_d50.append(SFD50[i][2])
    well_d501.append(80)   
#####################################################################    
#plot boundary conditions  
hdobj = flopy.utils.HeadFile(modelname + '.hds')
times = hdobj.get_times()
head = hdobj.get_data(totim = times[-1])
thickness = flopy.utils.postprocessing.get_saturated_thickness(head, mf, nodata = ['-1e30'], per_idx=-1)
for i in range(8): 
    thickness[i][thickness[i] < 0] = 0 

thick1 = thickness[0]
thick2 = thick1 + thickness[1]
thick3 = thick2 + thickness[2]
thick4 = thick3 + thickness[3]
thick5 = thick4 + thickness[4]
thick6 = thick5 + thickness[5]
thick7 = thick6 + thickness[6]
thick8 = thick7 + thickness[7]  
thick8= thick8*active


store_path = 'D:\\BX_Results_GW_Tables\\Baseline'
fig = plt.figure(figsize=(11, 9))
vmin = 0
vmax = 100
camp = 'Oranges'
norm=matplotlib.colors.LogNorm()
ax = fig.add_subplot(2, 2, 1 , aspect = 'auto')
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
hy = modelmap.plot_array(thick8, cmap = 'Greys', alpha = 0.8)
plt.xticks([])
#plt.colorbar(hy, orientation = 'horizontal', shrink = 1)
ax.set_title('Water Table')
#########################################################################
ax = fig.add_subplot(2, 2, 2, aspect = 'auto')
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
hy = modelmap.plot_array(thick8, cmap = 'Greys', alpha = 0.8)
Boundary = modelmap.plot_ibound(label  = 'ibound', color_noflow='White')
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green')
GHB = modelmap.plot_bc('GHB', label  = 'General head')
STR = modelmap.plot_bc('STR', label  = 'Streams',color = 'purple')
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain')
plt.scatter(well_x10, well_y10, c= well_d10, s = well_d101, marker = 'o', cmap = camp,  vmin=vmin, vmax=vmax)
plt.xticks([])
plt.yticks([])
ax.set_title('Year 10')
#########################################################################
ax = fig.add_subplot(2, 2, 3, aspect = 'equal')
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
hy = modelmap.plot_array(thick8, cmap = 'Greys', alpha = 0.8)
Boundary = modelmap.plot_ibound(label  = 'ibound', color_noflow='White')
#hy = modelmap.plot_array(mf.bcf6.hy.array[2, :,:], cmap = 'plasma', masked_values = [0.003456], alpha = 0.5)
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green')
GHB = modelmap.plot_bc('GHB', label  = 'General head')
STR = modelmap.plot_bc('STR', label  = 'Streams',color = 'purple')
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain')
plt.scatter(well_x20, well_y20, c= well_d20,  s = well_d201, marker = 'o', cmap = camp,  vmin=vmin, vmax=vmax)
#plt.scatter(well_x20, well_y20, c= well_d20,  s = well_d201, marker = 'o', cmap = 'YlGn', vmin=0, vmax=100)
xlabels= np.arange(0, 25000, 5000)
plt.xticks(xlabels)
ax.set_title('Year 20')
#########################################################################
ax = fig.add_subplot(2, 2, 4, aspect = 'equal')
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
hy = modelmap.plot_array(thick8, cmap = 'Greys', alpha = 0.8)
Boundary = modelmap.plot_ibound(label  = 'ibound', color_noflow='White')
#hy = modelmap.plot_array(mf.bcf6.hy.array[2, :,:], cmap = 'plasma', masked_values = [0.003456], alpha = 0.5)
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green')
GHB = modelmap.plot_bc('GHB', label  = 'General head')
STR = modelmap.plot_bc('STR', label  = 'Streams',color = 'purple')
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain')
plt.scatter(well_x30, well_y30, c= well_d30,  s = well_d301, marker = 'o', cmap = camp,  vmin=vmin, vmax=vmax)
#plt.scatter(well_x30, well_y30, c= well_d30,  s = well_d301, marker = 'o', cmap = 'YlGn', vmin=0, vmax=100)
ax.set_title('Year 30')
xlabels= np.arange(0, 25000, 5000)
plt.xticks(xlabels)
plt.yticks([])

plt.subplots_adjust(right=0.80)
cax = plt.axes([0.85, 0.12, 0.02, 0.76])
plt.colorbar(cax=cax)
cax.set_ylabel('Depletion (m3/day)', rotation=270, size = 14)
cax.yaxis.set_label_coords(3.6, 0.56)
#fig.suptitle('Depletion in September', fontsize=16, x = 0.4, y= 0.93)
plt.savefig(os.path.join(store_path, "Depletion_in_Sept1.jpg"), dpi = 1000, pad_inches=0.1)


#########################################################################
ax = fig.add_subplot(3, 2, 5, aspect = 'equal')
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound(label  = 'ibound')
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green')
GHB = modelmap.plot_bc('GHB', label  = 'General head')
STR = modelmap.plot_bc('STR', label  = 'Streams',color = 'purple')
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain')
#plt.scatter(well_x30, well_y30, c= well_d30, marker = 'o', cmap = 'brg', vmin=0, vmax=100)
plt.scatter(well_x40, well_y40, c= well_d40, s = well_d401,  marker = 'o', cmap = camp,  vmin=vmin, vmax=vmax)
#plt.scatter(well_x40, well_y40, c= well_d40, s = well_d401,  marker = 'o', cmap = 'YlGn',  vmin=0, vmax=100)
ax.set_title('Year 40')
#########################################################################
ax = fig.add_subplot(3, 2, 6, aspect = 'equal')
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound(label  = 'ibound')
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green')
GHB = modelmap.plot_bc('GHB', label  = 'General head')
STR = modelmap.plot_bc('STR', label  = 'Streams',color = 'purple')
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain')
plt.scatter(well_x50, well_y50, c= well_d50, s = well_d501,  marker = 'o', cmap = camp, vmin=vmin, vmax=vmax)
#plt.scatter(well_x50, well_y50, c= well_d50, s = well_d501,  marker = 'o', cmap = 'YlGn', vmin=0, vmax=100)
ax.set_title('Year 50')
plt.subplots_adjust(right=0.6)
cax = plt.axes([0.62, 0.12, 0.015, 0.76])
plt.colorbar(cax=cax)
cax.set_ylabel('Depletion (m3/day)', rotation=270, size = 14)
cax.yaxis.set_label_coords(3.6, 0.56)
fig.suptitle('Depletion in September', fontsize=16, x = 0.4, y= 0.93)
plt.savefig(os.path.join(store_path, "Flows_Sept.jpg"), dpi = 1000, pad_inches=0.1)

###########################################################################
#the numbering sequence of the python and modflow are different. 
#the row number of modflow start from the top, which python start from the bottom.
###########################################################################
#plot segment depletion 
#1. identify the segments
#2. plot the segment on the graph

file_path = 'D:\\BX_graph_seasonal'
river = pd.read_csv(os.path.join(file_path, "River_SFD%.csv"), header = 0, index_col = 0)
#constant.set_index('no', inplace = True)
#constant.PERCENT    
#constant[1:5]
#constant.loc[1,]
#################################################
#Rivers
seg22 = river.loc[river.segment == 22.0]
seg23 = river.loc[river.segment == 23.0]
seg24 = river.loc[river.segment == 24.0]
seg22.set_index('stress', inplace = True)
seg23.set_index('stress', inplace = True)
seg24.set_index('stress', inplace = True)
###########################################
"""#stream
seg19 = river.loc[river.segment == 19.0]
seg20 = river.loc[river.segment == 20.0]
seg21 = river.loc[river.segment == 21.0]

seg19.set_index('stress', inplace = True)
seg20.set_index('stress', inplace = True)
seg21.set_index('stress', inplace = True)"""
#############################################
year1 = seg24.loc[519+52*(1),]    #change seg numbers 
year5 = seg24.loc[52*(1+5)+519,]
year10 = seg24.loc[52*(1+10)+519,]
year15 = seg24.loc[52*(1+15)+519,]
year25 = seg24.loc[52*(1+24)+519,]
year30 = seg24.loc[52*(1+29)+519,]

###########################################################################
T = 0.5
seg_SFD1=[]
for i in range(2, len(year1)):
    if abs(year1[i]) > T:
        no = int(seg22.columns[i][5:]) #change number to locate numbers
        seg_SFD1.append([well_input0[no][1], well_input0[no][2], year1[i]])
     
well_x1 = []
well_y1 = []
well_d1 = []
well_d11 =[]
#well_d1 = []
for i in range(len(seg_SFD1)):
    well_y1.append((327-seg_SFD1[i][0])*50) 
    well_x1.append(seg_SFD1[i][1]*50)
    well_d1.append(seg_SFD1[i][2])
    well_d11.append(abs(seg_SFD1[i][2]))

###########################################################################
seg_SFD5=[]
for i in range(2, len(year5)):
    if abs(year5[i]) > T:
        no = int(seg22.columns[i][5:])
        seg_SFD5.append([well_input0[no][1], well_input0[no][2], year5[i]])
        
well_x5 = []
well_y5 = []
well_d5 = []
well_d51 = []
for i in range(len(seg_SFD5)):
    well_y5.append((327-seg_SFD5[i][0])*50) 
    well_x5.append(seg_SFD5[i][1]*50)
    well_d5.append(seg_SFD5[i][2])    
    well_d51.append(abs(seg_SFD5[i][2]))    

###########################################################################
seg_SFD10=[]
for i in range(2, len(year10)):
    if abs(year10[i]) > T:
        no = int(seg22.columns[i][5:])
        seg_SFD10.append([well_input0[no][1], well_input0[no][2], year10[i]])
       
well_x10 = []
well_y10 = []
well_d10 = []
well_d101 = []
for i in range(len(seg_SFD10)):
    well_y10.append((327-seg_SFD10[i][0])*50) 
    well_x10.append(seg_SFD10[i][1]*50)
    well_d10.append(seg_SFD10[i][2])
    well_d101.append(abs(seg_SFD10[i][2]))

###########################################################################
seg_SFD15=[]
for i in range(2, len(year15)):
    if abs(year15[i]) > T:
        no = int(seg22.columns[i][5:])
        seg_SFD15.append([well_input0[no][1], well_input0[no][2], year15[i]])

       
well_x15 = []
well_y15 = []
well_d15 = []
well_d151 = []
for i in range(len(seg_SFD15)):
    well_y15.append((327-seg_SFD15[i][0])*50) 
    well_x15.append(seg_SFD15[i][1]*50)
    well_d15.append(seg_SFD15[i][2])
    well_d151.append(abs(seg_SFD15[i][2]))

###########################################################################
seg_SFD25=[]
for i in range(2, len(year25)):
    if abs(year25[i]) > T:
        no = int(seg22.columns[i][5:])
        seg_SFD25.append([well_input0[no][1], well_input0[no][2], year25[i]])
 
    
well_x25 = []
well_y25 = []
well_d25 = []
well_d251 = []
for i in range(len(seg_SFD25)):
    well_y25.append((327-seg_SFD25[i][0])*50) 
    well_x25.append(seg_SFD25[i][1]*50)
    well_d25.append(seg_SFD25[i][2])
    well_d251.append(abs(seg_SFD25[i][2]))

###########################################################################
seg_SFD30=[]
for i in range(2, len(year30)):
    if abs(year30[i])> T:
        no = int(seg22.columns[i][5:])
        seg_SFD30.append([well_input0[no][1], well_input0[no][2], year30[i]])

well_x30 = []
well_y30 = []
well_d30 = []
well_d301 = []

#well_d1 = []
for i in range(len(seg_SFD30)):
    well_y30.append((327-seg_SFD30[i][0])*50) 
    well_x30.append(seg_SFD30[i][1]*50)
    well_d30.append(seg_SFD30[i][2])
    well_d301.append(abs(seg_SFD30[i][2]))

###########################################################################
seg_file = 'C:\\Users\\User\\Documents\\BC-ENV_MODFLOW\\BX Creek Model'
seg_no = pd.read_csv(os.path.join(seg_file, "Flow_Segments.csv"), header = 0, index_col = 0)
"""
seg_no22 = seg_no[seg_no.Segment == 22.0]
seg_no23 = seg_no[seg_no.Segment == 23.0]
seg_no24 = seg_no[seg_no.Segment == 24.0]

no22_y = (327 - seg_no22.row) * 50
no22_x = seg_no22.col *50
no23_y = (327 - seg_no23.row) * 50
no23_x = seg_no23.col *50
no24_y = (327 - seg_no24.row) * 50
no24_x = seg_no24.col *50
"""
###########################################################################
seg_no19 = seg_no[seg_no.Segment == 19.0]
seg_no20 = seg_no[seg_no.Segment == 20.0]
seg_no21 = seg_no[seg_no.Segment == 21.0]

no19_y = (327 - seg_no19.row) * 50
no19_x = seg_no19.col *50
no20_y = (327 - seg_no20.row) * 50
no20_x = seg_no20.col *50
no21_y = (327 - seg_no21.row) * 50
no21_x = seg_no21.col *50
 
###########################################################################
store_path = 'C:\\Users\\User\\Documents\\BC-ENV_MODFLOW\\BX Creek Model\\Pumping_Figures'
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(3, 2, 1 , aspect = 'equal')
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound(label  = 'ibound')
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green')
GHB = modelmap.plot_bc('GHB', label  = 'General head')
STR = modelmap.plot_bc('STR', label  = 'Streams',color = 'purple')
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain')

plt.scatter(no23_x, no23_y, marker = 'o', c='c', s = 2)
plt.scatter(no20_x, no20_y, marker = 'o', c='c', s = 2)
plt.scatter(no21_x, no21_y, marker = 'o', c='c', s = 2)
plt.text(2200, 3000, 'Rivers', size=9, zorder=10, color = 'green')
plt.text(5800, 5300, 'Streams', size=10, zorder=10, color = 'purple')
#plt.text(3800, 9000, 'Swan \n Lake', size=12, zorder=12, color = 'red')
plt.text(10500, 8500, 'Drains', size=10, zorder=10, color = 'navy')
plt.scatter(well_x1, well_y1, s = well_d11, c= well_d1, marker = 'o', cmap = 'brg', vmin=0, vmax=100)
#plt.scatter(well_x1, well_y1, c= well_d1, marker = 'o', cmap = 'brg', vmin=0, vmax=100)
ax.set_title('Year_1')
#########################################################################
ax = fig.add_subplot(3, 2, 2, aspect = 'equal')
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound(label  = 'ibound')
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green')
GHB = modelmap.plot_bc('GHB', label  = 'General head')
STR = modelmap.plot_bc('STR', label  = 'Streams',color = 'purple')
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain')
#plt.scatter(no19_x, no19_y, marker = 'o', c='c', s = 2)
#plt.scatter(no20_x, no20_y, marker = 'o', c='c', s = 2)
plt.scatter(no21_x, no21_y, marker = 'o', c='c', s = 2)
plt.scatter(well_x5, well_y5, s = well_d51, c= well_d5, marker = 'o', cmap = 'brg', vmin=0, vmax=100)
#plt.scatter(well_x5, well_y5, c= well_d5, marker = 'o', cmap = 'brg', vmin=0, vmax=100)
ax.set_title('Year_5')
#########################################################################
ax = fig.add_subplot(3, 2, 3, aspect = 'equal')
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound(label  = 'ibound')
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green')
GHB = modelmap.plot_bc('GHB', label  = 'General head')
STR = modelmap.plot_bc('STR', label  = 'Streams',color = 'purple')
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain')
#plt.scatter(no19_x, no19_y, marker = 'o', c='c', s = 2)
#plt.scatter(no20_x, no20_y, marker = 'o', c='c', s = 2)
plt.scatter(no21_x, no21_y, marker = 'o', c='c', s = 2)
plt.scatter(well_x10, well_y10, s = well_d101, c= well_d10, marker = 'o', cmap = 'brg', vmin=0, vmax=100)
#plt.scatter(well_x10, well_y10, c= well_d10, marker = 'o', cmap = 'brg', vmin=0, vmax=100)
ax.set_title('Year_10')

#########################################################################
ax = fig.add_subplot(3, 2, 4, aspect = 'equal')
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound(label  = 'ibound')
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green')
GHB = modelmap.plot_bc('GHB', label  = 'General head')
STR = modelmap.plot_bc('STR', label  = 'Streams',color = 'purple')
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain')
#plt.scatter(no19_x, no19_y, marker = 'o', c='c', s = 2)
#plt.scatter(no20_x, no20_y, marker = 'o', c='c', s = 2)
plt.scatter(no21_x, no21_y, marker = 'o', c='c', s = 2)
plt.scatter(well_x15, well_y15, s = well_d151, c= well_d15, marker = 'o', cmap = 'brg', vmin=0, vmax=100)
#plt.scatter(well_x15, well_y15, c= well_d15, marker = 'o', cmap = 'brg', vmin=0, vmax=100)
ax.set_title('Year_15')
#########################################################################
ax = fig.add_subplot(3, 2, 5, aspect = 'equal')
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound(label  = 'ibound')
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green')
GHB = modelmap.plot_bc('GHB', label  = 'General head')
STR = modelmap.plot_bc('STR', label  = 'Streams',color = 'purple')
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain')
#plt.scatter(no19_x, no19_y, marker = 'o', c='c', s = 2)
#plt.scatter(no20_x, no20_y, marker = 'o', c='c', s = 2)
plt.scatter(no21_x, no21_y, marker = 'o', c='c', s = 2)
plt.scatter(well_x25, well_y25, s = well_d251, c= well_d25, marker = 'o', cmap = 'brg', vmin=0, vmax=100)
#plt.scatter(well_x25, well_y25, c= well_d25, marker = 'o', cmap = 'brg', vmin=0, vmax=100)
ax.set_title('Year_25')
#########################################################################
ax = fig.add_subplot(3, 2, 6, aspect = 'equal')
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
Boundary = modelmap.plot_ibound(label  = 'ibound')
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green')
GHB = modelmap.plot_bc('GHB', label  = 'General head')
STR = modelmap.plot_bc('STR', label  = 'Streams',color = 'purple')
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain')
#plt.scatter(well_x30, well_y30, c= well_d30, marker = 'o', cmap = 'brg', vmin=0, vmax=100)
#plt.scatter(no19_x, no19_y, marker = 'o', c='c', s = 2)
#plt.scatter(no20_x, no20_y, marker = 'o', c='c', s = 2)
plt.scatter(no21_x, no21_y, marker = 'o', c='c', s = 2)
plt.scatter(well_x30, well_y30, s = well_d301, c= well_d30, marker = 'o', cmap = 'brg', vmin=0, vmax=100)
ax.set_title('Year_30')
plt.subplots_adjust(right=0.6)
cax = plt.axes([0.62, 0.12, 0.015, 0.76])
plt.colorbar(cax=cax)
plt.savefig(os.path.join(store_path, "Stream_21_5%.jpg"), dpi = 1000)


#########################################################################
#plot time series data
#file_path = 'D:\\BX_Week_Analysis'
file_path = 'D:\\BX_Results_Tables\\Baseline'

flow = pd.read_csv(os.path.join(file_path, "shorest_distance_R2.csv"), header = 0, index_col =0)
flow.hunt = flow[flow['Model'] == "Hunt"].reset_index()
flow.glover= flow[flow['Model'] == "Glover"].reset_index()

wells_R2_hunt = []
for i in range(len(flow.hunt)):
    well_no = flow.hunt['well_num'].loc[i] 
    row = well_input0[well_no][1]
    col = well_input0[well_no][2]
    wells_R2_hunt.append([well_no, row, col, flow.hunt['r2'].loc[i]])

well_x1 = []
well_y1 = []
well_d1 = []
well_d10 = []

#well_d1 = []
for i in range(len(wells_R2_hunt)):
    well_y1.append((327-wells_R2_hunt[i][1])*50) 
    well_x1.append(wells_R2_hunt[i][2]*50)
    well_d1.append(wells_R2_hunt[i][3])
    #well_d0.append(abs(wells_R2[i][3]*300))
    well_d10.append(100)
######################################################
wells_R2_glover= []
for i in range(len(flow.glover)):
    well_no = flow.glover['well_num'].loc[i] 
    row = well_input0[well_no][1]
    col = well_input0[well_no][2]
    wells_R2_glover.append([well_no, row, col, flow.glover['r2'].loc[i]])

well_x2 = []
well_y2 = []
well_d2 = []
well_d20 = []

#well_d1 = []
for i in range(len(wells_R2_glover)):
    well_y2.append((327-wells_R2_glover[i][1])*50) 
    well_x2.append(wells_R2_glover[i][2]*50)
    well_d2.append(wells_R2_glover[i][3])
    #well_d0.append(abs(wells_R2[i][3]*300))
    well_d20.append(100)

store_path = 'D:\\BX_Results_Tables\\Baseline'   
###plot
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1 , aspect = 'equal')
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
hy = modelmap.plot_array(mf.bcf6.hy.array[2, :,:], cmap = 'plasma', masked_values = [0.003456], alpha = 0.5)
Boundary = modelmap.plot_ibound(label  = 'ibound', color_noflow='White')
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green')
#GHB = modelmap.plot_bc('GHB', label  = 'General head')
STR = modelmap.plot_bc('STR', label  = 'Streams',color = 'purple')
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain')
ax.set_title("Glover Model")
plt.scatter(well_x2, well_y2, c= well_d2,  s= well_d20, marker = 'o', cmap = 'Reds', vmin=0, vmax=1)
plt.subplots_adjust(right=0.95)
cax = plt.axes([0.80, 0.10, 0.015, 0.78])
cax.set_ylabel('R2', rotation=270, size = 14)
plt.colorbar(cax=cax)
plt.savefig(os.path.join(store_path, "R2_Glover_wells.jpg"), dpi = 1000)


fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1 , aspect = 'equal')
modelmap = flopy.plot.ModelMap(model=mf, layer =0)
hy = modelmap.plot_array(mf.bcf6.hy.array[2, :,:], cmap = 'plasma', masked_values = [0.003456], alpha = 0.5)
Boundary = modelmap.plot_ibound(label  = 'ibound', color_noflow='White')
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green')
#GHB = modelmap.plot_bc('GHB', label  = 'General head')
STR = modelmap.plot_bc('STR', label  = 'Streams',color = 'purple')
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain')
#plt.text(2200, 2800, 'Rivers', size=12, zorder=10, color = 'green')
#plt.text(7500, 6000, 'Streams', size=12, zorder=10, color = 'purple')
#plt.text(3800, 10000, 'Swan \n Lake', size=12, zorder=12, color = 'blue')
#plt.text(10500, 8500, 'Drains', size=12, zorder=12, color = 'navy')
plt.scatter(well_x1, well_y1, c= well_d1,  s= well_d10, marker = 'o', cmap = 'Reds', vmin=0, vmax=1)
ax.set_title("Hunt Model")
plt.subplots_adjust(right=0.98)
cax = plt.axes([0.82, 0.10, 0.015, 0.78])
cax.set_ylabel('R2', rotation=270, size = 14)
plt.colorbar(cax=cax)
plt.savefig(os.path.join(store_path, "R2_Hunt_wells.jpg"), dpi = 1000)


