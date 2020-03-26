from __future__ import print_function, division
from IPython.display import clear_output, display
import time, os, sys, flopy, math, collections, shutil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
import flopy.utils.binaryfile as bf

path2mf = 'C:\\WRDAPP\\MF2005.1_12\\bin\\mf2005.exe'
file_path = 'C:\\Users\\User\\Documents\\BC-ENV_MODFLOW\\Peace\\Paleovalley_Sami'
os.chdir(file_path)
modelname ='SM_123'
mf = flopy.modflow.Modflow.load(modelname+"_MODFLOW_IN.NAM", verbose=True, check=True,  exe_name=path2mf)

"""MODFLOW 20 layer(s), 327 row(s), 308 column(s)"""

row_size = mf.dis.delc.array
col_size = mf.dis.delr.array
row_len = np.cumsum(row_size)
col_len = np.cumsum(col_size)  
nlay = mf.dis.nlay 
"""
row and col size are defined in row_len and col_len
"""

"""1. define the formula to calculate distance
2. calculate the distance of of each well to each segment cell
3. find the closest distance of the wells
4. compile well-stream pairs
"""
def calculateDistance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist  

seg=pd.read_csv(os.path.join(file_path, 'flow_segments_num.csv'), sep=',',header=0)
store_path = 'C:\\Users\\User\\Documents\\R\\Peace_GW_Analytical'
geometry = []
for i in range(len(well_input0)):
    well_row = well_input0[i][1]
    well_col = well_input0[i][2]
    well_row_len = row_len[well_row]
    well_col_len = col_len[well_col]
   
    for j in range(len(seg)):
        seg_row = seg['row'][j]
        seg_col = seg['col'][j]
        seg_row_len = row_len[seg_row]
        seg_col_len = col_len[seg_col]
        
        #distance.append(calculateDistance(well_row, well_col, seg_row, seg_col))
        geometry.append([i, seg['boundary'][j], seg['Segment'][j], well_row, well_col, seg_row, seg_col, calculateDistance(well_row_len, well_col_len, seg_row_len, seg_col_len)])
 
df = pd.DataFrame(geometry, columns = ['well_num', 'seg', 'seg_num', 'well_row', 'well_col', 'seg_row', 'seg_col', 'distance']) 
df.to_csv(os.path.join(store_path,'well_stream_geometry.csv'), index = False)

#############################################################################################################
"""select the closest distance of a well to river segment"""
indices = df.groupby(['well_num','seg_num'])['distance'].idxmin; indices
short_dis = df.loc[indices].reset_index()
short_dis = short_dis.drop(columns = ['index']) 
short_dis.to_csv(os.path.join(store_path,'well_stream_pairs.csv'), index = False)

#############################################################################################################
#extract conductance for stream
river = mf.riv.stress_period_data[0].tolist()
riv = pd.DataFrame(river, columns = ['lay', 'seg_row', 'seg_col', 'stage', 'cond', 'rbot'])
riv['boundary'] = 'River'
riv_cond = riv[['seg_row', 'seg_col', 'cond', 'boundary']]

drain = mf.drn.stress_period_data[0].tolist()
drn = pd.DataFrame(drain, columns = ['lay', 'seg_row', 'seg_col', 'stage', 'cond'])
drn['boundary'] = 'Drain'
drn_cond = drn[['seg_row', 'seg_col', 'cond', 'boundary']]

ghb = mf.ghb.stress_period_data[0].tolist()
ghb = pd.DataFrame(ghb, columns = ['lay', 'seg_row', 'seg_col', 'stage', 'cond'])
ghb['boundary'] = 'GHB'
ghb_cond = ghb[['seg_row', 'seg_col', 'cond', 'boundary']]

three = [riv_cond, drn_cond, ghb_cond]
flow_cond = pd.concat(three)
         
well_stream = pd.merge(short_dis, flow_cond, on=['seg_row', 'seg_col'], how='inner')
well_stream = well_stream.drop(columns = ['boundary']) 
well_stream = well_stream.sort_values(by = ['well_num', 'seg_num'])
well_stream.to_csv(os.path.join(store_path,'well_stream_cond.csv'), index = False)

#############################################################################################################
"""file_path = 'D:\\Peace_Pumping_baseline'
hdobj = flopy.utils.HeadFile(os.path.join(file_path, 'SM_123_MODFLOW_IN.hds'))
times = hdobj.get_times()
head = hdobj.get_data(totim = times[-1])
thickness = flopy.utils.postprocessing.get_saturated_thickness(head, mf, nodata = ['-1e30'], per_idx=-1)
"""
fpath = 'C:\\Users\\User\\Documents\\BC-ENV_MODFLOW\\Peace\\Paleovalley_Sami'
thickness =np.load(os.path.join(fpath, "thickness.npy"))

for i in range(nlay): 
    thickness[i][thickness[i] < 0] = 0  
 
Athick= thickness
for k in range(1, nlay):
    Athick[k] = thickness[k-1] + Athick[k]
   
tran = {}
for i in range(nlay):
    tran[i] = mf.bcf6.tran.array[i, :, :]
    
lthick = {}
for i in range(nlay): 
    lthick[i] = mf.dis.thickness[i,:, :]

laythick = {}    
laythick[0] = lthick[0] 
for i in range(1, nlay):
    laythick[i] = laythick[i-1]+lthick[i]
    
kh = {}
for i in range(nlay):
    kh[i] = tran[i]/lthick[i] #layer 1 horionizontal     

#deriving the K values
k_ave1 = {}
for i in range(nlay):
    k_ave1[i] = kh[i] * lthick[i]

k_ave ={}
k_ave[0] = k_ave1[0]
for i in range(1, nlay):
    k_ave[i] = k_ave[i-1]+ k_ave1[i]


k_input = {}
for i in range(nlay):
    k_input[i] = k_ave[i]/laythick[i]

hy ={}
for i in range(nlay):
    hy[i] = tran[i]/lthick[i] 
     
sy_data = hy[0].copy()
sy_data[sy_data < 3.0e-2] =  0.02
sy_data[sy_data > 3.0e-2] =  0.15

ss_data = hy[0].copy()
ss_data[ss_data < 3.0e-2] =  1e-5
ss_data[ss_data > 3.0e-2] =  2e-4

sy = sy_data
ss = ss_data   
 
well_kh = []
for i in range(len(well_input0)):
    lay = well_input0[i][0]
    row = well_input0[i][1]
    col = well_input0[i][2]
    s = sy[row][col] + ss[row][col]**Athick[lay][row][col]
   # kh = hy[lay][row][col]
    k = k_input[lay][row][col]
    T = k*Athick[lay][row][col]
    well_kh.append([i, lay, row, col, kh, k, T, s]) # row, col, k
            
well_T = pd.DataFrame(well_kh, columns = ['well_num','lay', 'row', 'col', 'kh', 'k', 'Trans', 'S']) 
well_T.to_csv(os.path.join(store_path,'well_T_SS.csv'), index = False)     

"""
well_kh = []
for i in range(len(well_input0)):
    for lay in range(20):
        row = well_input0[i][1]
        col = well_input0[i][2]
        s = sy[row][col] 
        kh = hy[lay][row][col]
        k = k_input[lay][row][col]
        T = k*Athick[lay][row][col]
        well_kh.append([i, lay, row, col, kh, k, T, s]) # row, col, k
            
well_T = pd.DataFrame(well_kh, columns = ['well_num','lay', 'row', 'col', 'kh', 'k', 'Trans', 'S']) 
well_T.to_csv(os.path.join(store_path,'wells_highest_K.csv'), index = False)     
"""