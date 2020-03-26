#run creat pumping well first
from __future__ import print_function
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
file_path = 'C:\\Users\\User\\Documents\\Bevan Models_Backup\\Transient_state'
#os.makedirs('D:\\Bevan_well_ss')
#file_path = 'C:\\Users\\User\\Documents\\BC-ENV_MODFLOW\\Bevan Models\\Transient_state'

modelname ='TRANSIENT_STATE'
mf = flopy.modflow.Modflow.load(os.path.join(file_path, modelname + "_MODFLOW_IN"), 
                                verbose=True, check=True, exe_name=path2mf)
"""MODFLOW 11 layer(s), 106 row(s), 124 column(s)"""

row_size = mf.dis.delc.array
col_size = mf.dis.delr.array

row_len = np.cumsum(row_size)
col_len = np.cumsum(col_size)   
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


fpath = 'C:\\Users\\User\\Documents\\BC-ENV_MODFLOW\\B'
seg=pd.read_csv(os.path.join(fpath, 'Segment_no.csv'), sep=',',header=0)

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
        geometry.append([i,seg['flow'][j], seg['segment'][j], well_row, well_col, seg_row, seg_col, calculateDistance(well_row_len, well_col_len, seg_row_len, seg_col_len)])
 
df = pd.DataFrame(geometry, columns = ['well_num', 'seg', 'seg_num', 'well_row', 'well_col', 'seg_row', 'seg_col', 'distance']) 
df.to_csv(os.path.join(fpath,'well_stream_geometry.csv'), index = False)

#############################################################################################################

"""select the closest distance of a well to river segment"""
indices = df.groupby(['well_num','seg_num'])['distance'].idxmin; indices
short_dis = df.loc[indices].reset_index()
short_dis = short_dis.drop(columns = ['index']) 
short_dis.to_csv(os.path.join(fpath,'well_stream_pairs.csv'),index = False)

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

ghb = seg.loc[seg['flow'] == 'ghb']
ghb['cond'] = riv_cond['cond'].max()
ghb_cond= ghb[['row', 'col', 'cond', 'flow']]
ghb_cond = ghb_cond.rename(index =str, columns = {'row': 'seg_row', 'col' : "seg_col", 'flow': "boundary"})

cst = seg.loc[seg['flow'] == 'constant']
cst['cond'] = riv_cond['cond'].max()
cst_cond = cst[['row', 'col', 'cond', 'flow']]
cst_cond =cst_cond.rename(index =str, columns = {'row': 'seg_row', 'col' : "seg_col", 'flow': "boundary"})


four = [riv_cond, drn_cond, ghb_cond, cst_cond]
flow_cond = pd.concat(four)
         
well_stream = pd.merge(short_dis, flow_cond, on=['seg_row', 'seg_col'], how='inner')
well_stream = well_stream.drop(columns = ['boundary']) 
well_stream = well_stream.sort_values(by = ['well_num', 'seg_num'])
well_stream.to_csv(os.path.join(fpath,'well_stream_cond.csv'), index = False)
#############################################################################################################
file_path = 'D:\\Bevan_50_Years'
hdobj = flopy.utils.HeadFile(os.path.join(file_path, 'TRANSIENT_STATE_MODFLOW_IN.hds'))
times = hdobj.get_times()
head = hdobj.get_data(totim = times[-1])
thickness = flopy.utils.postprocessing.get_saturated_thickness(head, mf, nodata = ['-1e30'], per_idx=-1)
modelname ='TRANSIENT_STATE'
mf = flopy.modflow.Modflow.load(os.path.join(file_path, modelname + "_MODFLOW_IN"), 
                                verbose=True, check=True, exe_name=path2mf)

  
Athick= thickness
for i in range(0, 11): 
    thickness[i][thickness[i] < 0] =0   
for k in range(1,11):
    Athick[k] = thickness[k-1] + Athick[k]
    
tran = {}
for i in range(11):
    tran[i] = mf.bcf6.tran.array[i, :, :]

lthick = {}
for i in range(11): 
    lthick[i] = mf.dis.thickness[i,:, :]
  
kh = {}
for i in range(11):
    kh[i] = tran[i]/lthick[i] #layer 1 horionizontal     

k_ave1 = {}
for i in range(11):
    k_ave1[i] = kh[i] * lthick[i]

k_ave ={}
k_ave[0] = k_ave1[0]
for i in range(1,11):
    k_ave[i] = k_ave1[i-1]+ k_ave1[i]
    
kh3 = mf.bcf6.hy.array[2, :, :]
sy_data = kh3.copy()
sy_data[sy_data < 0.8] =  0.01
sy_data[sy_data > 0.8] =  0.18

well_kh = []
for i in range(len(well_input0)):
    lay = well_input0[i][0]
    row = well_input0[i][1]
    col = well_input0[i][2]
    s = sy_data[row][col]  
    k = k_ave[lay][row][col]/lthick[lay][row][col]
    T = k*Athick[lay][row][col]
    well_kh.append([i, lay, row, col, k, T, s]) # row, col, k
            
well_T = pd.DataFrame(well_kh, columns = ['well_num','lay', 'row', 'col', 'k', 'Trans', 'S']) 
well_T.to_csv(os.path.join(fpath,'well_T.csv'), index = False)



