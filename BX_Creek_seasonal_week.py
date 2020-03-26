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

file_path = 'D:\\BX_Results\\Transient_90_Years'
os.chdir(file_path)
hdobj = flopy.utils.HeadFile('BX_VERNON2_MODFLOW_IN.hds')
times = hdobj.get_times()
head = hdobj.get_data(totim = times[-1])
ml.bas6.strt = head
os.chdir(fpath)

laycon = [3,3,3,3,3,3,3,3]
ml.bcf6.laycon = laycon
#############################################################################
#Remove the drain_blocks
df=pd.read_csv('Drain_stress.csv', sep=',',header=None)
drn_dis = df.values
ml.drn.stress_period_data = drn_dis
###########################################################
numyears = 55
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
dis = flopy.modflow.ModflowDis(ml, nlay, nrow, ncol, 
                               delr=delr, delc=delc,
                               top=ztop, botm=botm,
                               nper=nper, perlen=perlen, 
                               nstp=nstp, steady=steady)

kh3 = ml.bcf6.hy.array[2, :, :]

"""0.003456, 0.06048 , 0.0864  , 0.864"""
sy_data = kh3.copy()
sy_data[sy_data < 0.8] =  0.01
sy_data[sy_data > 0.8] =  0.15

ss_data = kh3.copy()
ss_data[ss_data < 0.8] =  1e-5
ss_data[ss_data > 0.8] =  1e-4

sy = sy_data
ss = ss_data
ml.bcf6.sf1 = ss  #sf1 specific storage
ml.bcf6.sf2 = sy  #sf2 specific yield
ml.bcf6.unit_number = [88]
laycon = [3,3,3,3,0,0,0,0] 
ml.bcf6.laycon = laycon
"re-write the streamflow package"
stream = {}
for i in range(nper):
       stream [i] = ml.str.stress_period_data[0]
ml.str.stress_period_data = stream

segment = {}
for j in range(nper):
    segment[j]= ml.str.segment_data[0]
ml.str.segment_data = segment
ml.str.ipakcb = 154
##############################################   
river ={}
for i in range(nper):
    river[i] = ml.riv.stress_period_data[0]
ml.riv.stress_period_data = river
ml.riv.unit_number = [66] 
##############################################
"re-write the drain package"
drain = {}
for i in range(nper):
    drain[i] = ml.drn.stress_period_data[0]
ml.drn.stress_period_data = drain
##############################################################
ghb = {}
for i in range(nper):
    ghb[i] = ml.ghb.stress_period_data[0]
ml.ghb.stress_period_data = ghb
################################################################
well = {}
for i in range(nper):
    well[i] = [3, 109, 224, 0]
ml.wel.stress_period_data = well
####################################################################
"recharge"
ratio = 365*0.013/4/7
monthly = [0.013*ratio]*4 + [0.014*ratio]*4 + [0.029*ratio]*4 +[0.117*ratio]*4 + [0.313*ratio]*4 +[0.289*ratio]*4 + [0.102*ratio]*4 + [0.10*ratio]*4 + [0.042*ratio]*4 + [0.027*ratio]*4 + [0.0195*ratio]*4 + [0.0195*ratio]*4 +[0.015*ratio]*4
monthly = monthly *  numyears        
##list mulitipl by 30 means repeat the data pattern for 30 times              
annual_rech = ml.rch.rech.array[0]
#mean_rech = annual_rech[np.nonzero(annual_rech)].mean()
recharge = {}
for i in range(nper):
     recharge [i] = annual_rech
rech={}
for j in range(nper):
      rech[j] = recharge[j] * monthly[j]     
ml.rch.rech = rech
###########################################
flopy.modflow.mfpcg.ModflowPcg(ml, mxiter=2500, iter1=1000, npcond=2, 
                              rclose=0.1, hclose=1, nbpol=2,relax = 0.99, 
                               damp=0.8, dampt = 0.5, ihcofadd =2)
############################### output control
oc_stress = {}
for kper in range(nper):
    for kstp in range(nstp[kper]):
        oc_stress[(kper, kstp)] = ['save budget']
        
oc = flopy.modflow.ModflowOc(ml, stress_period_data=oc_stress,
                             compact=True, unitnumber= [16])
##############################################################################
#### run baseline model (no pumping)
file_path = 'C:\\Users\\User\\Documents\\Pumping_week'
store_path = 'D:\\BX_Week' 

ml.change_model_ws(file_path)
ml.write_input()
success, mloutput = ml.run_model()
if not success:
   raise Exception('MODFLOW did not terminate normally.')  
   # postprocess
###########################################################################
#read list files and export budget file
os.chdir(file_path)
mfl = flopy.utils.MfListBudget(os.path.join(file_path, modelname+'.LST'))
cbb = bf.CellBudgetFile(os.path.join(file_path, modelname+".bgt"))
df_flux, df_vol = mfl.get_dataframes(start_datetime=None)
df_flux.to_csv(os.path.join(store_path, 'SummarizeBudget_baseline.csv'), index=False)
drn = cbb.get_data(text='Drains', full3D=True)
riv = cbb.get_data(text='river', full3D=True)
strm = cbb.get_data(text='stream leakage',full3D=True)

os.chdir(fpath)
seg=pd.read_csv('Flow_Segments.csv', sep=',',header=0)
#print(seg.loc[seg['Boundary'] == 'Drain'])
#drainlist = seg.loc[seg['Boundary'] == 'Drain']
flow_segment = seg.values.tolist()
#segments = np.asarray(segment)
os.chdir(file_path)
#############################################################
drain_cells = []  
for t in range(nper):
   for i in range(941):  
      drain_cells.append([t,  int(flow_segment[i][4]), drn[t][0, int(flow_segment[i][2]), int(flow_segment[i][3])]])        

from operator import itemgetter
SUM={} # create blank dictionary called SUM
for l in drain_cells:
    t = l[0]
    i = l[1]
    q = l[2]
   
    # make the key the unique combination of t and i
    key = (t, i)
    if key in SUM:
        SUM[key] += q
    else:   
        SUM[key] = q # sum q for each key of t and i

# convert back to a list of lists (called sum_drain) to match the output format that you showed above
sum_drain = []
for k, v in sorted(SUM.items(), key=itemgetter(0,1)): # note this SUM.items() is for Python 3, if you are using python version 2, then use d.iteritems() instead.
    t = k[0]
    i = k[1]
    #print k
    sum_drain.append([t,i,v])
#########################################################################
riv_cells = []  
for t in range(nper):
   for i in range(1071,1323,1):  
     riv_cells.append([t,  int(flow_segment[i][4]), riv[t][0, int(flow_segment[i][2]), int(flow_segment[i][3])]])        

from operator import itemgetter
SUM={} # create blank dictionary called SUM
for l in riv_cells:
    t = l[0]
    i = l[1]
    q = l[2]
# make the key the unique combination of t and i
    key = (t, i)
    if key in SUM:
        SUM[key] += q
    else:   
        SUM[key] = q # sum q for each key of t and i

# convert back to a list of lists (called sum_drain) to match the output format that you showed above
sum_riv = []
for k, v in sorted(SUM.items(), key=itemgetter(0,1)): 
    t = k[0]
    i = k[1]
    #print k
    sum_riv.append([t,i,v]) 
#################################################################################    
strm_cells = []  
for t in range(nper):
   for i in range(943,1071,1):  
     strm_cells.append([t,  int(flow_segment[i][4]), strm[t][0, int(flow_segment[i][2]), int(flow_segment[i][3])]])        

from operator import itemgetter
SUM={} # create blank dictionary called SUM
for l in strm_cells:
    t = l[0]
    i = l[1]
    q = l[2]
  
    # make the key the unique combination of t and i
    key = (t, i)
    if key in SUM:
        SUM[key] += q
    else:   
        SUM[key] = q # sum q for each key of t and i

# convert back to a list of lists (called sum_drain) to match the output format that you showed above
sum_strm = []
for k, v in sorted(SUM.items(), key=itemgetter(0,1)): # note this SUM.items() is for Python 3, if you are using python version 2, then use d.iteritems() instead.
    t = k[0]
    i = k[1]
    #print k
    sum_strm.append([t,i,v])   

####################################################################################################    
os.chdir(store_path)
np.savetxt("riv_baseline.csv", sum_riv, delimiter=",",  fmt='%1.3f', header = "stress_period, segment, Q")
np.savetxt("drain_baseline.csv", sum_drain, delimiter=",",  fmt='%1.3f', header = "stress_period, segment, Q")
np.savetxt("stream_baseline.csv", sum_strm, delimiter=",",  fmt='%1.3f', header = "stress_period, segment, Q")

##############################################################################
#read wells located that has saturated 
well_path = 'C:\\Users\\User\\Documents\\BC-ENV_MODFLOW\\BX Creek Model\\Analytical input'
well_loc= pd.read_csv(os.path.join(well_path, 'well_T.csv'), sep=',',header=0)
well_no =well_loc[ well_loc['Trans']>0]['well_num'].tolist()

for WellNum in well_no:   
    well = {}
    for i in range(0,nper):
         well[i] = well_input0[WellNum] #define the start of the the pumping          
    for k in range(50):
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
    mfl = flopy.utils.MfListBudget(os.path.join(file_path, modelname+'.LST'))
    df_flux, df_vol = mfl.get_dataframes(start_datetime=None)
    df_flux.to_csv(os.path.join(store_path, 'SummarizeBudget' + '_' + str(WellNum) + '.csv'), index=False)
    cbb = bf.CellBudgetFile(os.path.join(file_path, modelname+".bgt"))
    drn = cbb.get_data(text='Drains', full3D=True)
    riv = cbb.get_data(text='river', full3D=True)
    strm = cbb.get_data(text='stream leakage',full3D=True)

#################################################################################################
    drain_cells = []  
    for t in range(nper):
       for i in range(941):  
          drain_cells.append([t,  int(flow_segment[i][4]), drn[t][0, int(flow_segment[i][2]), int(flow_segment[i][3])]])        
    
    from operator import itemgetter
    SUM={} # create blank dictionary called SUM
    for l in drain_cells:
        t = l[0]
        i = l[1]
        q = l[2]
       
        # make the key the unique combination of t and i
        key = (t, i)
        if key in SUM:
            SUM[key] += q
        else:   
            SUM[key] = q # sum q for each key of t and i
    
    # convert back to a list of lists (called sum_drain) to match the output format that you showed above
    sum_drain = []
    for k, v in sorted(SUM.items(), key=itemgetter(0,1)): # note this SUM.items() is for Python 3, if you are using python version 2, then use d.iteritems() instead.
        t = k[0]
        i = k[1]
        #print k
        sum_drain.append([t,i,v])
    #########################################################################
    riv_cells = []  
    for t in range(nper):
       for i in range(1071,1323,1):  
         riv_cells.append([t,  int(flow_segment[i][4]), riv[t][0, int(flow_segment[i][2]), int(flow_segment[i][3])]])        
    
    from operator import itemgetter
    SUM={} # create blank dictionary called SUM
    for l in riv_cells:
        t = l[0]
        i = l[1]
        q = l[2]
    # make the key the unique combination of t and i
        key = (t, i)
        if key in SUM:
            SUM[key] += q
        else:   
            SUM[key] = q # sum q for each key of t and i
    
    # convert back to a list of lists (called sum_drain) to match the output format that you showed above
    sum_riv = []
    for k, v in sorted(SUM.items(), key=itemgetter(0,1)): 
        t = k[0]
        i = k[1]
        #print k
        sum_riv.append([t,i,v]) 
    #################################################################################    
    strm_cells = []  
    for t in range(nper):
       for i in range(943,1071,1):  
         strm_cells.append([t,  int(flow_segment[i][4]), strm[t][0, int(flow_segment[i][2]), int(flow_segment[i][3])]])        
    
    from operator import itemgetter
    SUM={} # create blank dictionary called SUM
    for l in strm_cells:
        t = l[0]
        i = l[1]
        q = l[2]
      
        # make the key the unique combination of t and i
        key = (t, i)
        if key in SUM:
            SUM[key] += q
        else:   
            SUM[key] = q # sum q for each key of t and i
    
    # convert back to a list of lists (called sum_drain) to match the output format that you showed above
    sum_strm = []
    for k, v in sorted(SUM.items(), key=itemgetter(0,1)): # note this SUM.items() is for Python 3, if you are using python version 2, then use d.iteritems() instead.
        t = k[0]
        i = k[1]
        #print k
        sum_strm.append([t,i,v])   
    
    ####################################################################################################    
    os.chdir(store_path)
    np.savetxt('riv' + '_' + str(WellNum) + '.csv', sum_riv, delimiter=",",  fmt='%1.3f', header = "stress_period, segment, Q")
    np.savetxt('drain' + '_' + str(WellNum) + '.csv', sum_drain, delimiter=",",  fmt='%1.3f', header = "stress_period, segment, Q")
    np.savetxt('strm' + '_' + str(WellNum) + '.csv', sum_strm, delimiter=",",  fmt='%1.3f', header = "stress_period, segment, Q")
    
