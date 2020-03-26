import glob, csv, os 
import pandas as pd
import matplotlib.pyplot as plt

#base = pd.read_csv(os.path.join(file_path, "drain_baseline.csv"), index_col='title')
'''
base = pd.read_csv("drain__0.csv", header = 0)
base.columns = ['stress','segment','Q']
base.head()
base.tail()
base.all
base.dtypes
base.columns = ['stress','segment','Q']
base['Q']
base.sort_values(by = ['segment', 'stress'], ascending = True)
base['cal'] = base.segment + base.Q
for i in range(3):
    base['cal' + str(i)] = base.segment + base.Q
'''
#############################################################
fpath = 'D:\\Peace_results\\Baseline'
store_path = 'D:\\Peace_results\\data_sum'
file_path = 'D:\\Peace_results'
os.chdir(file_path) 
drain_base = pd.read_csv(os.path.join(fpath, "drain_sum_baseline.csv"), header = 0)
drain_base.columns = ['stress','segment','Q']
drn_names = sorted(glob.glob('drain*.csv')) 

for f in drn_names: 
    df = pd.read_csv(f, header = 0)
    df.columns = ['stress','segment','Q']
    #print(df.head())
    drain_base[str(f[:-4])] = (df.Q - drain_base.Q)
    
drain_base.to_csv(os.path.join(store_path,'Drain_SFD.csv'), index = False)
drain_base.describe().to_csv(os.path.join(store_path,'Drain_summary.csv'))
#################################################################
riv_base = pd.read_csv(os.path.join(fpath,"river_sum_baseline.csv"), header = 0)
riv_base.columns = ['stress','segment','Q']
riv_names = sorted(glob.glob('riv*.csv')) 

for r in riv_names: 
    df = pd.read_csv(r, header = 0)
    df.columns = ['stress','segment','Q']
    riv_base[str(r[:-4])] = (df.Q - riv_base.Q)
 
riv_base.to_csv(os.path.join(store_path,'River_SFD.csv'), index = False)
riv_base.describe().to_csv(os.path.join(store_path,'River_summary.csv'))
#########################################################################
ghb_base = pd.read_csv(os.path.join(fpath, "GHB_sum_baseline.csv"), header = 0)
ghb_base.columns = ['stress','segment','Q']
ghb_names = sorted(glob.glob('GHB*.csv')) 

for s in ghb_names: 
    df = pd.read_csv(s, header = 0)
    df.columns = ['stress','segment','Q']
    #print(df.head())
    ghb_base[str(s[:-4])] = (df.Q - ghb_base.Q)
 
ghb_base.to_csv(os.path.join(store_path,'GHB_SFD.csv'), index = False)
ghb_base.describe().to_csv(os.path.join(store_path,'ghb_summary.csv'))
###################################################################
bgt = pd.read_csv(os.path.join(fpath,"SummarizeBudget_baseline.csv"), header = 0)
bgt.columns = ['STORAGE_IN','CONSTANT_IN','WELLS_IN', 'DRAIN_IN', 'RIVER_IN', 'Head_IN', 'RECHARGE_IN', 'TOTAL_IN', 
               'STORAGE_OUT','CONSTANT_OUT','WELLS_OUT', 'DRAIN_OUT', 'RIVER_OUT', 'RECHARGE_OUT', 'Head_OUT','TOTAL_OUT','IN-OUT', 'PERCENT']
#bgt.head()
store = bgt[['PERCENT']]
ghb = bgt[['PERCENT']]
river = bgt[['PERCENT']]
drain = bgt[['PERCENT']]
bgt_names = sorted(glob.glob('SummarizeBudget*.csv')) 
#########################################################################
for f in bgt_names: 
    df = pd.read_csv(f, header = 0)
    df.columns = ['STORAGE_IN','CONSTANT_IN','WELLS_IN', 'DRAIN_IN', 'RIVER_IN', 'Head_IN', 'RECHARGE_IN', 'TOTAL_IN', 
               'STORAGE_OUT','CONSTANT_OUT','WELLS_OUT', 'DRAIN_OUT', 'RIVER_OUT', 'RECHARGE_OUT', 'Head_OUT','TOTAL_OUT','IN-OUT', 'PERCENT']
    store[f[:-4]] = (df.STORAGE_IN-df.STORAGE_OUT)-(bgt.STORAGE_IN-bgt.STORAGE_OUT)
 
store.to_csv(os.path.join(store_path,'Storge.csv'))
#store.describe().to_csv(os.path.join(store_path,'Storage_BGT_Summary.csv'))

#########################################################################
#storage
for f in bgt_names: 
    df = pd.read_csv(f, header = 0)
    df.columns = ['STORAGE_IN','CONSTANT_IN','WELLS_IN', 'DRAIN_IN', 'RIVER_IN', 'Head_IN', 'RECHARGE_IN', 'TOTAL_IN', 
               'STORAGE_OUT','CONSTANT_OUT','WELLS_OUT', 'DRAIN_OUT', 'RIVER_OUT', 'RECHARGE_OUT', 'Head_OUT','TOTAL_OUT','IN-OUT', 'PERCENT']
    
    store[f[:-4]] = ((df.STORAGE_IN- df.STORAGE_OUT) - (bgt.STORAGE_IN- bgt.STORAGE_OUT))
 
store.to_csv(os.path.join(store_path,'Storage_BGT.csv'))
#store.describe().to_csv(os.path.join(store_path,'Storage_BGT_Summary.csv'))

###################################################################################################################
#constant_head
for f in bgt_names: 
    df = pd.read_csv(f, header = 0)
    df.columns = ['STORAGE_IN','CONSTANT_IN','WELLS_IN', 'DRAIN_IN', 'RIVER_IN', 'Head_IN', 'RECHARGE_IN', 'TOTAL_IN', 
               'STORAGE_OUT','CONSTANT_OUT','WELLS_OUT', 'DRAIN_OUT', 'RIVER_OUT', 'RECHARGE_OUT', 'Head_OUT','TOTAL_OUT','IN-OUT', 'PERCENT']
    
    ghb[f[:-4]] = ((df.Head_IN- df.Head_OUT)-(bgt.Head_IN- bgt.Head_OUT)) 
   
ghb.to_csv(os.path.join(store_path,'GHB_BGT.csv'))  
#ghb.describe().to_csv(os.path.join(store_path,'GHB_bgt_summary.csv'))

#################################################################################################################  
#river_budget
for f in bgt_names: 
    df = pd.read_csv(f, header = 0)
    df.columns = ['STORAGE_IN','CONSTANT_IN','WELLS_IN', 'DRAIN_IN', 'RIVER_IN', 'Head_IN', 'RECHARGE_IN', 'TOTAL_IN', 
               'STORAGE_OUT','CONSTANT_OUT','WELLS_OUT', 'DRAIN_OUT', 'RIVER_OUT', 'RECHARGE_OUT', 'Head_OUT','TOTAL_OUT','IN-OUT', 'PERCENT']
    
    river[f[:-4]] = ((df.RIVER_IN - df.RIVER_OUT) -(bgt.RIVER_IN - bgt.RIVER_OUT))
   
river.to_csv(os.path.join(store_path,'River_BGT.csv'))  
#river.describe().to_csv(os.path.join(store_path,'River_BGT_summary.csv'))

########################################################################################################
#Drain_budget
for f in bgt_names: 
    df = pd.read_csv(f, header = 0)
    df.columns = ['STORAGE_IN','CONSTANT_IN','WELLS_IN', 'DRAIN_IN', 'RIVER_IN', 'Head_IN', 'RECHARGE_IN', 'TOTAL_IN', 
               'STORAGE_OUT','CONSTANT_OUT','WELLS_OUT', 'DRAIN_OUT', 'RIVER_OUT', 'RECHARGE_OUT', 'Head_OUT','TOTAL_OUT','IN-OUT', 'PERCENT']
    
    drain[f[:-4]] = ((df.DRAIN_IN - df.DRAIN_OUT) - (bgt.DRAIN_IN - bgt.DRAIN_OUT))
    
drain.to_csv(os.path.join(store_path,'Drain_BGT.csv'))   
#drain.describe().to_csv(os.path.join(store_path,'Drain_bgt_summary.csv'))
########################################################################################################
