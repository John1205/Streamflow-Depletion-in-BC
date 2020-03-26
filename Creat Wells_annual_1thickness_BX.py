import time, sys, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import flopy
from numpy import array
import collections

path2mf = 'C:\\WRDAPP\\MF2005.1_12\\bin\\mf2005.exe'

fpath = 'C:\\Users\\User\\Documents\\BC-ENV_MODFLOW\\BX Creek Model'
model_ws = os.chdir(fpath)

modelname = 'BX_VERNON2'
mf = flopy.modflow.Modflow.load(modelname +"_MODFLOW_IN.nam", 
                                verbose=True,
                                check=True, 
                                exe_name=path2mf)

df=pd.read_csv('Drain_stress.csv', sep=',',header=None)
drn_dis = df.values
mf.drn.stress_period_data = drn_dis

nrow = mf.dis.nrow
ncol = mf.dis.ncol

"""define wells 10 determine the space"""
cell = np.zeros((nrow, ncol))
for i in range(0,nrow, 18):
    for j in range(0,ncol, 18):
        cell[i,j] = 2   
        
#plt.imshow(cell)
"""extract the active boundary in the study domain
#0-non-active, 1, active; -1 constant boundary"""
active = mf.bas6.ibound.array[0]
active[active == -1] = 0
#make the constance boundary as inactive zones

#define active cell in the domain
wells = cell*active
wells[wells<0] = 0  # exclude the wells in constant head

##identify the rows and col of the wells
wel_loc=[]
for row in range(wells.shape[0]):
  for col in range(wells.shape[1]):
       if wells[row, col] == 2 :
          # print(row, col)
           wel_loc.append([row, col])
           
####################################################################
riv = mf.riv.stress_period_data[0]
drain = mf.drn.stress_period_data[0]
stream = mf.str.stress_period_data[0]

for item in range(len(wel_loc)):
    row = wel_loc[item][0]
    col = wel_loc[item][1] 
    for i in range(len(riv)):   
        if riv[i][1] == row and riv[i][2] == col:
             #print(row, col)
             wells[row, col] = 0
             
for item in range(len(wel_loc)):
    row = wel_loc[item][0]
    col = wel_loc[item][1] 
    for i in range(len(drain)):   
        if drain[i][1] == row and drain[i][2] == col:
             #print(row, col)
             wells[row, col] = 0             
             
for item in range(len(wel_loc)):
    row = wel_loc[item][0]
    col = wel_loc[item][1] 
    for i in range(len(stream)):   
        if stream[i][1] == row and stream[i][2] == col:
             #print(row, col)
             wells[row, col] = 0        
    
########################################
#count well No. in the domain
"""unique_elements, counts_elements = np.unique(wells, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))"""

#define the layers of the wells
ele= mf.dis.top.array    
ele = ele*active


lthick1 = mf.dis.thickness.array[0]           
lthick2 = mf.dis.thickness.array[1]
lthick3 = mf.dis.thickness.array[2]           
lthick4 = mf.dis.thickness.array[3]           
lthick5 = mf.dis.thickness.array[4]           
lthick6 = mf.dis.thickness.array[5]           
lthick7 = mf.dis.thickness.array[6]           
lthick8 = mf.dis.thickness.array[7]  

thick1 = lthick1
thick2 = thick1 + lthick2
thick3 = thick2 + lthick3
thick4 = thick3 + lthick4
thick5 = thick4 + lthick5
thick6 = thick5 + lthick6

screen = 35
well1 = wells.copy()
well_stress =[]
for i in range(len(wel_loc)):
    row = wel_loc[i][0]
    col = wel_loc[i][1]    
    if  screen < thick1[row, col]: 
         layer = 0 
         well_stress.append([layer, row, col])
    elif screen < thick2[row, col]: 
         layer = 1
         well_stress.append([layer, row, col])      
    elif screen < thick3[row, col]: 
         layer = 2
         well_stress.append([layer, row, col])
    elif screen < thick4[row, col]: 
         layer = 3
         well_stress.append([layer, row, col])
    elif screen< thick5[row, col]: 
         layer = 4
         well_stress.append([layer, row, col])
         #print(layer, row, col)
    elif screen < thick6[row, col]: 
         layer = 5
         well_stress.append([layer, row, col])
    else: 
         layer = 6
         well_stress.append([layer, row, col])    
        
        
well_input0 = well_stress.copy()
Q =0       
for item in range(len(well_input0)):
        well_input0[item].append(Q)


"""
kh4 = mf.bcf6.hy.array[3, :, :]

well_aquifer = []
for i in range(len(well_input0)):
    row = well_input0[i][1]
    col = well_input0[i][2]
    k = kh4[row, col]
    well_aquifer.append([i, row, col, k])

well_zones = pd.DataFrame(well_aquifer, columns = ['number', 'row', 'col', 'K'])
well_zones.to_csv('well_zones.csv', index = False)
    
""" 
        
##################################################################
well_stress =[]
    
for i in range(len(wel_loc)):
    row = wel_loc[i][0]
    col = wel_loc[i][1]    
    if  screen < thick1[row, col]: 
         layer = 0 
         well_stress.append([layer, row, col])
        # print(layer, row, col)
    elif screen < thick2[row, col]: 
         layer = 1
         well_stress.append([layer, row, col])
         #print(layer, row, col)
        
    elif screen < thick3[row, col]: 
         layer = 2
         well_stress.append([layer, row, col])
         #print(layer, row, col)
    elif screen < thick4[row, col]: 
         layer = 3
         well_stress.append([layer, row, col])
         #print(layer, row, col)
    elif screen< thick5[row, col]: 
         layer = 4
         well_stress.append([layer, row, col])
         #print(layer, row, col)
    elif screen < thick6[row, col]: 
         layer = 5
         well_stress.append([layer, row, col])
    else: 
         layer = 6
         well_stress.append([layer, row, col])                    

well_input5 = well_stress.copy()
Q5 =-19       
for item in range(len(well_input5)):
        well_input5[item].append(Q5)

##################################################################
well_stress =[]
for i in range(len(wel_loc)):
    row = wel_loc[i][0]
    col = wel_loc[i][1]    
    if  screen < thick1[row, col]: 
         layer = 0 
         well_stress.append([layer, row, col])
        # print(layer, row, col)
    elif screen < thick2[row, col]: 
         layer = 1
         well_stress.append([layer, row, col])
         #print(layer, row, col)
        
    elif 30 < thick3[row, col]: 
         layer = 2
         well_stress.append([layer, row, col])
         #print(layer, row, col)
    elif screen < thick4[row, col]: 
         layer = 3
         well_stress.append([layer, row, col])
         #print(layer, row, col)
    elif screen< thick5[row, col]: 
         layer = 4
         well_stress.append([layer, row, col])
         #print(layer, row, col)
    elif screen < thick6[row, col]: 
         layer = 5
         well_stress.append([layer, row, col])
    else: 
         layer = 6
         well_stress.append([layer, row, col])    
        #print(layer, row, col)                        

well_input6 = well_stress     

Q6 = -108      
for item in range(len(well_input6)):
        well_input6[item].append(Q6)
           

##################################################################
well_stress =[]
    
for i in range(len(wel_loc)):
    row = wel_loc[i][0]
    col = wel_loc[i][1]    
    if  screen < thick1[row, col]: 
         layer = 0 
         well_stress.append([layer, row, col])
        # print(layer, row, col)
    elif screen < thick2[row, col]: 
         layer = 1
         well_stress.append([layer, row, col])
         #print(layer, row, col)  
    elif screen < thick3[row, col]: 
         layer = 2
         well_stress.append([layer, row, col])
         #print(layer, row, col)
    elif screen < thick4[row, col]: 
         layer = 3
         well_stress.append([layer, row, col])
         #print(layer, row, col)
    elif screen< thick5[row, col]: 
         layer = 4
         well_stress.append([layer, row, col])
         #print(layer, row, col)
    elif screen < thick6[row, col]: 
         layer = 5
         well_stress.append([layer, row, col])
    else: 
         layer = 6
         well_stress.append([layer, row, col])                           

well_input7 = well_stress     

Q7 = -214      
for item in range(len(well_input7)):
        well_input7[item].append(Q7)
        
##################################################################
well_stress =[]
    
for i in range(len(wel_loc)):
    row = wel_loc[i][0]
    col = wel_loc[i][1]    
    if  screen < thick1[row, col]: 
         layer = 0 
         well_stress.append([layer, row, col])
        # print(layer, row, col)
    elif screen < thick2[row, col]: 
         layer = 1
         well_stress.append([layer, row, col])
         #print(layer, row, col)      
    elif screen < thick3[row, col]: 
         layer = 2
         well_stress.append([layer, row, col])
         #print(layer, row, col)
    elif screen < thick4[row, col]: 
         layer = 3
         well_stress.append([layer, row, col])
         #print(layer, row, col)
    elif screen< thick5[row, col]: 
         layer = 4
         well_stress.append([layer, row, col])
         #print(layer, row, col)
    elif screen < thick6[row, col]: 
         layer = 5
         well_stress.append([layer, row, col])
    else: 
         layer = 6
         well_stress.append([layer, row, col])       

well_input8 = well_stress     

Q8 = -171      
for item in range(len(well_input8)):
        well_input8[item].append(Q8)
        
##################################################################
well_stress =[]
    
for i in range(len(wel_loc)):
    row = wel_loc[i][0]
    col = wel_loc[i][1]    
    if  screen < thick1[row, col]: 
         layer = 0 
         well_stress.append([layer, row, col])
        # print(layer, row, col)
    elif screen < thick2[row, col]: 
         layer = 1
         well_stress.append([layer, row, col])
         #print(layer, row, col)
        
    elif screen < thick3[row, col]: 
         layer = 2
         well_stress.append([layer, row, col])
         #print(layer, row, col)
    elif screen < thick4[row, col]: 
         layer = 3
         well_stress.append([layer, row, col])
         #print(layer, row, col)
    elif screen< thick5[row, col]: 
         layer = 4
         well_stress.append([layer, row, col])
         #print(layer, row, col)
    elif screen < thick6[row, col]: 
         layer = 5
         well_stress.append([layer, row, col])
    else: 
         layer = 6
         well_stress.append([layer, row, col])                         

well_input9 = well_stress

Q9 =-84       
for t in range(len(well_input9)):
        well_input9[t].append(Q9)
        
             