
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd

LOWEST_TEMP = 0
HIGHEST_TEMP = 150

def bins(x):
    if(x <= 40):
        return "0 > x<= 40"
    if(x <= 60):
        return "40 > X >= 60"
    if(x <=80):
        return "60 > X >= 80"
    if(x <100):
        return "80 > X >100"
    else:
        return "100 = X"

def label_maker_9000(x, thres=50):
    if x <= thres:
        return r"Mesophilic"
    else:
        return "Thermophilic"
    
def tem(x, df_tm):
    try:
        return float(df_tm[x])
    except KeyError:
        return np.nan

def label_maker_x9(x, thres=50):
    if x <= thres:
        return 0
    else:
        return 1
    
def tem(x, df_tm):
    try:
        return float(df_tm[x])
    except KeyError:
        return np.nan

def coef_det_k(y_true, y_pred):
    """Computer coefficient of determination R^2
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

def get_class_sizes(args, data_df):
    temp_ranges = [LOWEST_TEMP]+args.ranges
    temp_ranges.append(HIGHEST_TEMP)
    print(temp_ranges)
    class_sizes = []
    for idx in range(len(temp_ranges)-1):
        class_sizes.append(np.sum(np.logical_and(data_df['TM']>temp_ranges[idx], data_df['TM']<=temp_ranges[idx+1])))
    
    return np.array(class_sizes), np.array(temp_ranges)

class ARGUMENTS():
    def __init__(self, ranges , enumerate, fixed_scale, name):
        self.ranges = ranges
        self.enumerate = enumerate
        self.fixed_scale = fixed_scale
        self.name = name

        self.verbose = False




def fixed_scale(args, data_df, use_min = False):

    counts_df = data_df.groupby("TM").size().reset_index(name ="counts")
    
    if not use_min:
        name = args.name
        print(name)
        if args.enumerate == 0:
            temp_ranges = [LOWEST_TEMP]+args.ranges
            temp_ranges.append(HIGHEST_TEMP)
        else:
            min_temp = int(min(data_df[name]))
            max_temp = round(max(data_df[name]))
            temp_ranges = list(range(min_temp, max_temp+1, args.enumerate))
        print(temp_ranges)
        data_df_list = []
        for idx in range(len(temp_ranges)-1):
            logic_vec = np.logical_and(data_df[name]>temp_ranges[idx], data_df[name]<=temp_ranges[idx+1])
            if data_df[logic_vec].shape[0]==0:
                print(f"no sequences at temperature range [{temp_ranges[idx]}, {temp_ranges[idx+1]}]")
                continue
            #print(f"temperature sanity check {min(data_df[logic_vec].Temperature)} {max(data_df[logic_vec].Temperature)}")
            data_df_list.append(data_df[logic_vec])
    
        data_df_list_sampled = []
        for idx, df in enumerate(data_df_list):
            
            if args.fixed_scale >= df.shape[0]:
                replace = True
            else:
                replace = False
                
            data_df_list_sampled.append(df.sample(n = args.fixed_scale, replace=replace))
       
            if(args.verbose):
                print(f"Saving {args.fixed_scale} Sequences from temperature range [{temp_ranges[idx]}, {temp_ranges[idx+1]}]")
                print(f"sanity check {data_df_list_sampled[-1].shape[0]}")
        data_df = pd.concat(data_df_list_sampled)
        return data_df

    else:
        name = args.name
        print(name)
        if args.enumerate == 0:
            temp_ranges = [LOWEST_TEMP]+args.ranges
            temp_ranges.append(HIGHEST_TEMP)
        else:
            min_temp = int(min(data_df[name]))
            max_temp = round(max(data_df[name]))
            temp_ranges = list(range(min_temp, max_temp+1, args.enumerate))
        print(temp_ranges)
        data_df_list = []
        for idx in range(len(temp_ranges)-1):
            logic_vec = np.logical_and(data_df[name]>temp_ranges[idx], data_df[name]<=temp_ranges[idx+1])
            if data_df[logic_vec].shape[0]==0:
                print(f"no sequences at temperature range [{temp_ranges[idx]}, {temp_ranges[idx+1]}]")
                continue
            #print(f"temperature sanity check {min(data_df[logic_vec].Temperature)} {max(data_df[logic_vec].Temperature)}")
            data_df_list.append(data_df[logic_vec])
    
        data_df_list_sampled = []
        for idx, df in enumerate(data_df_list):
            count = counts_df.loc[counts_df["TM"] == df["TM"].values[0]]["counts"].values  

            scale = max(args.fixed_scale, count)
            
            if  scale >= df.shape[0]:
                replace = True
            else:
                replace = False
            
            data_df_list_sampled.append(df.sample(n = scale, replace=replace))
       
            if(args.verbose):
                print(f"Saving {args.fixed_scale} Sequences from temperature range [{temp_ranges[idx]}, {temp_ranges[idx+1]}]")
                print(f"sanity check {data_df_list_sampled[-1].shape[0]}")
        data_df = pd.concat(data_df_list_sampled)
        return data_df

