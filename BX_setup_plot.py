
from IPython.display import clear_output, display
import time, sys, os, flopy, collections, shutil
import numpy as np
import matplotlib.pyplot as plt
import flopy.utils.binaryfile as bf
import pandas as pd
from numpy import array

path2mf = 'C:\\WRDAPP\\mf2005.1_12\\bin\\mf2005.exe'
fpath = 'C:\\Users\\User\\Documents\\BC-ENV_MODFLOW\\BX Creek Model'
model_ws = os.chdir(fpath)
modelname = 'BX_VERNON2'

ml = flopy.modflow.Modflow.load(modelname +"_MODFLOW_IN.nam", verbose=True,
                                check=True, exe_name=path2mf)

df=pd.read_csv('Drain_stress.csv', sep=',',header=None)
drn_dis = df.values
ml.drn.stress_period_data = drn_dis

ml.wel.stress_period_data = well_input0

a= ml.bcf6.hy.array[3,:,:]

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(1, 1, 1, aspect = 'equal')
modelmap = flopy.plot.ModelMap(model=ml, layer = 0)
hy= modelmap.plot_array(a)
Boundary = modelmap.plot_ibound(label  = 'ibound')
RIV = modelmap.plot_bc('RIV', label  = 'Rivers', color = 'green', plotAll=True)
STR = modelmap.plot_bc('STR', label  = 'Stream', color = 'purple', plotAll=True)
DRB = modelmap.plot_bc('DRN', color='navy', label  = 'Drain', plotAll=True)
WEL = modelmap.plot_bc('WEL', plotAll=True, )
#plt.text(2200, 3000, 'Rivers', size=15, zorder=10, color = 'green')
#plt.text(5800, 5300, 'Streams', size=15, zorder=10, color = 'purple')
#plt.text(3800, 9000, 'Swan \n Lake', size=15, zorder=12, color = 'red')
#plt.text(10500, 8500, 'Drains', size=15, zorder=10, color = 'navy')
#linecollection = modelmap.plot_grid(linestyle='-', linewidth=2, alpha = 0.01)
ax.set_title('Wells location')
fig.savefig("Wells_locations.jpg", dpi = 1000)