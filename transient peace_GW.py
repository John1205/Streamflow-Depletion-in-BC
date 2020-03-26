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
ml = flopy.modflow.Modflow.load(modelname+"_MODFLOW_IN.NAM", verbose=True, check=True,  exe_name=path2mf)

#['DIS', 'BAS6', 'BCF6', 'WEL', 'DRN', 'RIV', 'GHB', 'RCH', 'OC']
#MODFLOW 20 layer(s), 327 row(s), 308 column(s), 1 stress period(s)

head = np.load('initial head.npy') 
ml.bas6.strt = head
#############################################################################
numyears = 35
sp_per_year = 52
sp_length_days = [7]
# define stress period data
nper = numyears*sp_per_year
perlen = sp_length_days*numyears*sp_per_year
nstp = [2]*nper  
steady = [False]*nper

nlay = ml.dis.nlay
ncol = ml.dis.ncol
nrow = ml.dis.nrow
delr = ml.dis.delr
delc = ml.dis.delc
ztop = ml.dis.top
botm = ml.dis.botm
# make new dis object
dis = flopy.modflow.ModflowDis(ml, nlay, nrow, ncol, delr=delr, delc=delc, top=ztop, botm=botm,
                               nper=nper, perlen=perlen,  nstp=nstp, steady=steady)
#################################################################################
trans ={}
for i in range(nlay):
    trans[i] = ml.bcf6.tran.array[i,:,:]
thick = {}
for j in range(nlay-1):
    thick[j] = ml.dis.thickness[j,:, :]    
hy ={}
for i in range(nlay-1):
    hy[i] = trans[i]/thick[i] 
   
#hyunquie = np.array([1.00e-04, 9.00e-04, 3.46e-02, 0.864, 4.32, 259])
                    # 1.16e-9, 1.0e-8, 4.0e-7, 1.0e-5, 5.0e-5, 3.0e-3  m/s
                #sy:  0.18     0.18,       0.25,  0.25,  0.25, 0.25
                #ss: 1e-5,     1e-5,  1e-4, 1e-4, 1e-4
                    
sy_data = hy[0].copy()
sy_data[sy_data < 3.0e-2] =  0.02
sy_data[sy_data > 3.0e-2] =  0.15

ss_data = hy[0].copy()
ss_data[ss_data < 3.0e-2] =  1e-5
ss_data[ss_data > 3.0e-2] =  2e-4

sy = sy_data
ss = ss_data
laycon = ml.bcf6.laycon
hdry = ml.bcf6.hdry
wetdry = ml.bcf6.wetdry
tran = ml.bcf6.tran
trpy = ml.bcf6.trpy
intercellt = ml.bcf6.intercellt
laycon = ml.bcf6.laycon
hy = ml.bcf6.hy
vcont = ml.bcf6.vcont

bcf = flopy.modflow.mfbcf.ModflowBcf(ml, intercellt=intercellt, laycon=laycon, trpy=trpy, hdry=-1e+30, iwdflg=0, 
                               wetfct= 1.0, iwetit=1, ihdwet=0, tran=tran, hy=hy , vcont=vcont, sf1=ss, sf2=sy, wetdry=wetdry)

del trans, thick, hy, vcont, tran
##############################################   
river ={}
for i in range(nper):
    river[i] = ml.riv.stress_period_data[0]
ml.riv.stress_period_data = river
ml.riv.unit_number = [66]
del river 
##############################################
"re-write the drain package"
drain = {}
for i in range(nper):
    drain[i] = ml.drn.stress_period_data[0]
ml.drn.stress_period_data = drain
del drain
##############################################################
ghb = {}
for i in range(nper):
    ghb[i] = ml.ghb.stress_period_data[0]
ml.ghb.stress_period_data = ghb
del ghb
################################################################
well = {}
for i in range(nper):
    well[i] = [6, 102, 183, 0]
ml.wel.stress_period_data = well
del well
####################################################################
"recharge"
"""
ratio = 365/7/4
#monthly = [0.014*ratio]*4 + [0.012*ratio]*4 + [0.012*ratio]*4 +[0.034*ratio]*4 + [0.106*ratio]*4 +[0.302*ratio]*4 + [0.209*ratio]*4 + [0.10*ratio]*4 + [0.099*ratio]*4 + [0.0705*ratio]*4 + [0.0471*ratio]*4 + [0.0249*ratio]*4 +[0.0180*ratio]*4
monthly = [0.062*ratio]*4 + [0.051*ratio]*4 + [0.053*ratio]*4 +[0.089*ratio]*4 + [0.103*ratio]*4 +[0.124*ratio]*4 + [0.114*ratio]*4 + [0.09*ratio]*4 + [0.099*ratio]*4 + [0.064*ratio]*4 + [0.06*ratio]*4 + [0.064*ratio]*4 +[0.063*ratio]*4
monthly = monthly *  numyears     """   
##list mulitipl by 30 means repeat the data pattern for 30 times              
annual_rech = ml.rch.rech.array[0]
#mean_rech = annual_rech[np.nonzero(annual_rech)].mean()
recharge = {}
for i in range(nper):
     recharge [i] = annual_rech
"""    
rech={}
for j in range(nper):
      rech[j] = recharge[j] * monthly[j]  """
ml.rch.rech = recharge
del annual_rech, recharge
###########################################
flopy.modflow.mfpcg.ModflowPcg(ml, mxiter=2500, iter1=1000, npcond=2, rclose=0.5, hclose=1, nbpol=2,
                               relax = 0.5, dampt = 0.6, ihcofadd =2)
############################### output control
oc_stress = {}
for kper in range(nper):
    for kstp in range(nstp[kper]):
        oc_stress[(kper, kstp)] = ['save budget']
        
oc = flopy.modflow.ModflowOc(ml, stress_period_data=oc_stress, compact=True, unitnumber= [16])
del oc_stress
##############################################################################
#### run baseline model (no pumping)
store_path = 'D:\\Peace_GW_3' 
ml.change_model_ws(store_path)
ml.write_input()
#ml.write_input(SelPackList = ['PCG', 'DIS'])

success, mloutput = ml.run_model()
if not success:
   raise Exception('MODFLOW did not terminate normally.')  
   # postprocess
###########################################################################
#read list files and export budget file
os.chdir(store_path)
mfl = flopy.utils.MfListBudget(os.path.join(store_path, modelname+'.LST'))
df_flux, df_vol = mfl.get_dataframes(start_datetime=None)
df_flux.to_csv(os.path.join(store_path, 'SummarizeBudget_baseline.csv'), index=False)
del mfl, df_flux, df_vol

segments = pd.read_csv(os.path.join(file_path, 'flow_segments_num.csv'), header = 0)
drain_seg = segments[segments.boundary == 'Drain'].reset_index()
river_seg = segments[segments.boundary== 'River'].reset_index()
#ghb_seg = segments[segments.boundary == 'GHB'].reset_index()
del segments

cbb = bf.CellBudgetFile(os.path.join(store_path, modelname+".bgt"))
drn = cbb.get_data(text='drains', full3D=True)
drain_cells=[]
for t in range(nper):
   for i in range(len(drain_seg)):  
       drn_flow = drn[t][drain_seg['layer'][i], drain_seg['row'][i], drain_seg['col'][i]]     
       drain_cells.append([t,  drain_seg['Segment'][i], drn_flow])        

drain_bgt = pd.DataFrame(drain_cells, columns = ["period", "segment", "Q"])
drain_sum = drain_bgt.groupby(['period', 'segment']).sum()
drain_sum.to_csv(os.path.join(store_path, 'drain_sum_baseline.csv'), index = True)
del drn, drain_cells, drain_bgt, drain_sum

cbb = bf.CellBudgetFile(os.path.join(store_path, modelname+".bgt"))
riv = cbb.get_data(text='river', full3D=True)
del cbb
river_cells = []  
for t in range(nper):
   for i in range(len(river_seg)):     
      river_cells.append([t,  river_seg['Segment'][i], riv[t][river_seg['layer'][i], river_seg['row'][i], river_seg['col'][i]]])        

river_bgt = pd.DataFrame(river_cells, columns = ["period", "segment", "Q"])
river_sum = river_bgt.groupby(['period', 'segment']).sum()
river_sum.to_csv(os.path.join(store_path, 'river_sum_baseline.csv'), index = True)
del riv, river_cells, river_bgt, river_sum

"""
cbb = bf.CellBudgetFile(os.path.join(store_path, modelname+".bgt"))
ghb = cbb.get_data(text='head', full3D=True)
del cbb

ghb_cells = []  
for t in range(nper):
   for i in range(len(ghb_seg)):     
      ghb_cells.append([t,  ghb_seg['Segment'][i], ghb[t][ghb_seg['layer'][i], ghb_seg['row'][i], ghb_seg['col'][i]]])        

ghb_bgt = pd.DataFrame(ghb_cells, columns = ["period", "segment", "Q"])
ghb_sum = ghb_bgt.groupby(['period', 'segment']).sum()
del ghb
ghb_sum.to_csv(os.path.join(store_path, 'GHB_sum_baseline.csv'), index = True)
del ghb_cells, ghb_bgt, ghb_sum"""
#########################################################################################
for WellNum in range(len(well_input0)):   
    well = {}
    for i in range(0,nper):
         well[i] = well_input0[WellNum] #define the start of the the pumping          
    for k in range(30):
         well[260 + 19+52*k]=well_input5[WellNum]
         well[260 + 20+52*k]=well_input5[WellNum]
         well[260 + 21+52*k]=well_input5[WellNum]
         well[260 + 22+52*k]=well_input5[WellNum]
            
         well[260 + 23+52*k]=well_input6[WellNum]
         well[260 + 24+52*k]=well_input6[WellNum]
         well[260 + 25+52*k]=well_input6[WellNum]
         well[260 + 26+52*k]=well_input6[WellNum]
            
         well[260 + 27+52*k]=well_input7[WellNum]
         well[260 + 28+52*k]=well_input7[WellNum]
         well[260 + 29+52*k]=well_input7[WellNum]
         well[260 + 30+52*k]=well_input7[WellNum]
            
         well[260 + 31+52*k]=well_input8[WellNum]
         well[260 + 32+52*k]=well_input8[WellNum]
         well[260 + 33+52*k]=well_input8[WellNum]
         well[260 + 34+52*k]=well_input8[WellNum]
            
         well[260 + 35+52*k]=well_input9[WellNum]
         well[260 + 36+52*k]=well_input9[WellNum]
         well[260 + 37+52*k]=well_input9[WellNum]
         well[260 + 38+52*k]=well_input9[WellNum]
            
    ml.wel.stress_period_data = well   
    # write input, run model
    ml.write_input(SelPackList = ['WEL'])
    success,  mloutput = ml.run_model()
    if not success:
       # raise Exception('MODFLOW did not terminate normally.')  
       continue 
     
    # postprocess
        
    mfl = flopy.utils.MfListBudget(os.path.join(store_path, modelname+'.LST'))
    df_flux, df_vol = mfl.get_dataframes(start_datetime=None)
    df_flux.to_csv(os.path.join(store_path, 'SummarizeBudget_' + str(WellNum) + '.csv'), index=False)
    del mfl, df_flux, df_vol
    
    cbb = bf.CellBudgetFile(os.path.join(store_path, modelname+".bgt"))
    drn = cbb.get_data(text='drains', full3D=True)
 
    drain_cells=[]
    for t in range(nper):
       for i in range(len(drain_seg)):  
           drn_flow = drn[t][drain_seg['layer'][i], drain_seg['row'][i], drain_seg['col'][i]]     
           drain_cells.append([t,  drain_seg['Segment'][i], drn_flow])        
    
    drain_bgt = pd.DataFrame(drain_cells, columns = ["period", "segment", "Q"])
    drain_sum = drain_bgt.groupby(['period', 'segment']).sum()
    drain_sum.to_csv(os.path.join(store_path, 'drain' + '_' + str(WellNum) + '.csv'), index = True)
    
    del drn, drain_cells, drain_bgt, drain_sum
    
    riv = cbb.get_data(text='river', full3D=True)
    river_cells = []  
    for t in range(nper):
       for i in range(len(river_seg)):     
          river_cells.append([t,  river_seg['Segment'][i], riv[t][river_seg['layer'][i], river_seg['row'][i], river_seg['col'][i]]])        
    
    del riv
    river_bgt = pd.DataFrame(river_cells, columns = ["period", "segment", "Q"])
    river_sum = river_bgt.groupby(['period', 'segment']).sum()
    river_sum.to_csv(os.path.join(store_path, 'river' + '_' + str(WellNum) + '.csv'), index = True)
    del river_cells, river_bgt, river_sum 
    
    """
    cbb = bf.CellBudgetFile(os.path.join(store_path, modelname+".bgt"))
    ghb = cbb.get_data(text='head', full3D=True)
    del cbb
    
    ghb_cells = []  
    for t in range(nper):
       for i in range(len(ghb_seg)):     
          ghb_cells.append([t,  ghb_seg['Segment'][i], ghb[t][ghb_seg['layer'][i], ghb_seg['row'][i], ghb_seg['col'][i]]])        
    
    ghb_bgt = pd.DataFrame(ghb_cells, columns = ["period", "segment", "Q"])
    ghb_sum = ghb_bgt.groupby(['period', 'segment']).sum()
    del ghb
    ghb_sum.to_csv(os.path.join(store_path, 'GHB' + '_' + str(WellNum) + '.csv'), index = True)
    del ghb_cells, ghb_bgt, ghb_sum
    """