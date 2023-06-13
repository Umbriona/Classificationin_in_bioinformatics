

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
    
def tem(x, df_tm):
    try:
        return float(df_tm[x])
    except KeyError:
        return np.nan