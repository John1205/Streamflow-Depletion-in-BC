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

ml = flopy.modflow.Modflow.load(modelname +"_MODFLOW_IN.nam", 
                                verbose=True,
                                check=True, 
                                exe_name=path2mf)

file_path = 'D:\\BX_Results\\Transient_90_Years'
os.chdir(file_path)
hdobj = flopy.utils.HeadFile('BX_VERNON2_MODFLOW_IN.hds')
times = hdobj.get_times()
head = hdobj.get_data(totim = times[-1])
ml.bas6.strt = head
os.chdir(fpath)

#laycon = [3,3,3,3,0,0,0,0]
#ml.bcf6.laycon = laycon
#BCF = flopy.modflow.ModflowBcf.load('BX_VERNON2.bcf', ml)
#####################################################################
"""over-write the first four layers horizontal conductivity
change the horizontal conductivity into two units"""
kh1 = ml.bcf6.hy.array[0, :, :] #layer 1 horionizontal 
kh2 = ml.bcf6.hy.array[1, :, :]
kh3 = ml.bcf6.hy.array[2, :, :]
kh4 = ml.bcf6.hy.array[3, :, :]
kh5 = ml.bcf6.hy.array[4, :, :]
kh6 = ml.bcf6.hy.array[5, :, :]
kh7 = ml.bcf6.hy.array[6, :, :]
kh8 = ml.bcf6.hy.array[7, :, :]

kh1[kh1 < 0.864] = 0.003456
kh2[kh2 < 0.864] = 0.003456
kh3[kh3 < 0.864] = 0.003456
kh4[kh4 < 0.864] = 0.003456
############################################################################
"""over-write the rest four-layers transmissivity"""
"""Hk for the rest four layers are uniform"""
T1 = ml.bcf6.tran.array[0, :, :] #layer 1 horionizontal 
T2 = ml.bcf6.tran.array[1, :, :]
T3 = ml.bcf6.tran.array[2, :, :]
T4 = ml.bcf6.tran.array[3, :, :]
T5 = ml.bcf6.tran.array[4, :, :]
T6 = ml.bcf6.tran.array[5, :, :]
T7 = ml.bcf6.tran.array[6, :, :]
T8 = ml.bcf6.tran.array[7, :, :]

thick1 = ml.dis.thickness[0,:, :]
thick2 = ml.dis.thickness[1,:, :]
thick3 = ml.dis.thickness[2,:, :]
thick4 = ml.dis.thickness[3,:, :]
thick5 = ml.dis.thickness[4,:, :]
thick6 = ml.dis.thickness[5,:, :]
thick7 = ml.dis.thickness[6,:, :]
thick8 = ml.dis.thickness[7,:, :]

kh_5 = T5/thick5
kh_6 = T6/thick6
kh_7 = T7/thick7
kh_8 = T8/thick8

kh_5 = kh_5.round(6) ##refine the model
kh_6 = kh_6.round(6)
kh_7 = kh_7.round(6)
kh_8 = kh_8.round(6)

"""creat the new data for horizontal K"""
h_data = np.zeros((8,327, 400))
h_data [0, :, : ]= kh1
h_data [1, :, : ]= kh2
h_data [2, :, : ]= kh3
h_data [3, :, : ]= kh4
h_data [4, :, : ]= kh_5
h_data [5, :, : ]= kh_6
h_data [6, :, : ]= kh_7
h_data [7, :, : ]= kh_8
ml.bcf6.hy = h_data  ## overwrite the K
################################################################################
#extract the vertical conductivity
kv1 = ml.bcf6.vcont.array[0, :, :] #layer 1 horionizontal 
kv2 = ml.bcf6.vcont.array[1, :, :]
kv3 = ml.bcf6.vcont.array[2, :, :]
kv4 = ml.bcf6.vcont.array[3, :, :]
kv5 = ml.bcf6.vcont.array[4, :, :]
kv6 = ml.bcf6.vcont.array[5, :, :]
kv7 = ml.bcf6.vcont.array[6, :, :]
#create ta dataframe for the vertical K
kz1 = kh1
kz2 = kh2
kz3 = kh3
kz4 = kh4
kz5 = kh_5
kz6 = kh_6
kz7 = kh_7
kz8 = kh_8
#simplify kz 
z_data = np.zeros((8,327, 400))
z_data [0, :, : ]= kz1
z_data [1, :, : ]= kz2
z_data [2, :, : ]= kz3
z_data [3, :, : ]= kz4
z_data [4, :, : ]= kz5
z_data [5, :, : ]= kz6
z_data [6, :, : ]= kz7
z_data [7, :, : ]= kz8

kz1[kz1 == 0.864] = 0.0864
kz2[kz2 == 0.864] = 0.0864
kz3[kz2 == 0.864] = 0.0864
kz4[kz2 == 0.864] = 0.0864

"""Calculate the vertical conductance for the model"""
vcont1 = 1/ (0.5*thick1/kz1 + 0.5*thick2/kz2)
vcont2 = 1/ (0.5*thick2/kz2 + 0.5*thick3/kz3)
vcont3 = 1/ (0.5*thick3/kz3 + 0.5*thick4/kz4)
vcont4 = 1/ (0.5*thick4/kz4 + 0.5*thick5/kz4)
vcont5 = 1/ (0.5*thick5/kz5 + 0.5*thick6/kz6)
vcont6 = 1/ (0.5*thick6/kz6 + 0.5*thick7/kz7)
vcont7 = 1/ (0.5*thick7/kz7 + 0.5*thick8/kz8)

vcont_data = np.zeros((7,327, 400))
vcont_data [0, :, : ]= vcont1
vcont_data [1, :, : ]= vcont2
vcont_data [2, :, : ]= vcont3
vcont_data [3, :, : ]= vcont4
vcont_data [4, :, : ]= vcont5
vcont_data [5, :, : ]= vcont6
vcont_data [6, :, : ]= vcont7
ml.bcf6.vcont = vcont_data
#############################################################################
#Remove the drain_blocks
df=pd.read_csv('Drain_stress.csv', sep=',',header=None)
drn_dis = df.values
ml.drn.stress_period_data = drn_dis

###########################################################
numyears = 50
sp_per_year = 52
sp_length_days = [7]
# define stress period data
nper = numyears*sp_per_year
perlen = sp_length_days*numyears*sp_per_year
nstp = [1]*nper  
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

sy_data = kh1
sy_data[sy_data == 0.0864] =  0.18
sy_data[sy_data == 0.003456] =  0.02

ss_data = kh1
ss_data[ss_data == 0.0864] =  1e-4
ss_data[ss_data == 0.003456] =  1e-5

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
                               damp=0.8, dampt = 0.6, ihcofadd =2)

############################### output control
oc_stress = {}
for kper in range(nper):
    for kstp in range(nstp[kper]):
        oc_stress[(kper, kstp)] = ['save budget', 'print budget']
        
oc = flopy.modflow.ModflowOc(ml, stress_period_data=oc_stress,
                             compact=True, unitnumber= [16])
##############################################################################
#### run baseline model (no pumping)
file_path = 'C:\\Users\\User\\Documents\\Pumping_annual'
#os.makedirs('D:\\SFD_annual')
store_path = 'D:\\SFD_annual' 

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
df_flux, df_vol = mfl.get_dataframes()
df_flux.to_csv(os.path.join(store_path, 'SummarizeBudget_baseline.csv'), index=False)
drn = cbb.get_data(text='Drains', full3D=True)
riv = cbb.get_data(text='river', full3D=True)
strm = cbb.get_data(text='stream leakage',full3D=True)


seg=pd.read_csv(os.path.join(fpath, 'Flow_Segments.csv'), sep=',',header=0)
#print(seg.loc[seg['Boundary'] == 'Drain'])
#drainlist = seg.loc[seg['Boundary'] == 'Drain']
flow_segment = seg.values.tolist()
#segments = np.asarray(segment)
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
for WellNum in range(0, len(well_input0)):  #well_input in the creat well script
    well = {}
    for i in range(0,260):
        well[i] = well_input0[WellNum] #define the start of the the pumping
    for j in range(260,nper):
        well[j] = well_input1[WellNum] 
        
    ml.wel.stress_period_data = well   
    # write input, run model
    ml.write_input(SelPackList = ['WEL'])
    success,  mloutput = ml.run_model()
    if not success:
       # raise Exception('MODFLOW did not terminate normally.')  
       continue 
        
     
    # postprocess
    mfl = flopy.utils.MfListBudget(os.path.join(file_path, modelname+'.LST'))
    df_flux, df_vol = mfl.get_dataframes()
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
    
