# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 17:27:36 2022

@author: giova
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

def calculate_autocorr(logret, max_lag, start):
    lag = [j for j in range(start, max_lag+1)]
    T = len(logret)
   
    f_value = lambda j: logret.autocorr(lag = j)
    f_threshold = lambda j: 2 / ( ( T - 3 - j)**0.5)
    
    value = list(map(f_value, lag))
    threshold = list(map(f_threshold, lag))
    
    return pd.Series(value), pd.Series(threshold)



def plot_autocorr(logreturn, maxlag, start, save = False):
    plt.figure(" ")
    autocorr = calculate_autocorr(logreturn, maxlag,start)[0]
    threshold = calculate_autocorr(logreturn, maxlag, start)[1]
    plt.plot(list(range(start,maxlag+1)), autocorr, marker="o", linewidth=0, color = "blue")
    plt.plot(list(range(start, maxlag+1)),threshold, color = "red")
    plt.plot(list(range(start,maxlag+1)), -threshold, color = "red")
    plt.plot(list(range(start,maxlag+1)), [0 for j in range(start, maxlag+1)], color = "black")
    plt.xlabel("Lag")
    plt.title(logreturn.name)
    #adding vertical line instead of the connection line 
    for j in range(start, maxlag+1):
        x = [j, j]
        y = [0,autocorr[j-start]]
        plt.plot(x,y, color = "blue")
    if save: 
        plt.savefig(f"Autocorrelation_{logreturn.name}.pdf")
    plt.show()
 
   

def autocorrelation(logreturn, maxlag=50, plot=True, include_zero = True, save = False):
   
    if include_zero:
       start = 0
    else: 
       start = 1
        
    if isinstance(logreturn, pd.DataFrame):
        autocorr = lambda x: calculate_autocorr(x, maxlag, start)[0]
        toreturn  = logreturn.apply(autocorr, 0)
        if plot:
            for j in logreturn.columns:
                plot_autocorr(logreturn[j], maxlag, start, save)
        return toreturn 
    

    else:
        toreturn = calculate_autocorr(logreturn, maxlag, start)[0]
        if plot: 
            plot_autocorr(logreturn, maxlag, start, save)
        return toreturn
    
    
    

def get_log_mean_squared(serie, t):
        
        diff = serie - serie.shift(t)
        diff = diff.dropna()
        mean_squared = diff.pow(2).mean()
        log_mean_squared = np.log(mean_squared)
        return log_mean_squared

    
def get_H(serie):
    
    t = np.arange(1, 100, 1)
    f = lambda t: get_log_mean_squared(serie, t)
    result = np.array( list(map(f, t)))
    log_t = np.log(t)

    X = np.array([np.ones(len(t)), log_t]).transpose()
    y = np.array(result).reshape(len(result), 1)

    beta = np.linalg.inv((X.transpose() @ X)) @ X.transpose() @ result
    H = beta[1]/2
    a = np.exp(beta[0])
    
    return a, H



def download_daily_data(name):
    import yfinance as yf
    interval = "1d"
    data = yf.download(name, interval = interval)
    data.to_csv(f"data/raw/daily/YF/{name}.csv.gz")
    
        
def load_daily_raw_data(name):
    data = pd.read_csv(f"data/raw/daily/YF/{name}.csv.gz")
    return data 
    
    
    
def abs_ecdf(logreturn):
    from statsmodels.distributions.empirical_distribution import ECDF
    absolute_data = logreturn.abs()
    ecdf = ECDF(logreturn.abs())
    plt.yscale("log")
    plt.xscale("log")
    plt.plot(ecdf.x, 1 - ecdf.y)
    plt.show()
    
    
    
    
def yang_zhang_vol(data, k):
   
    date = data.Date 
    close_ = pd.DataFrame(data.Close).set_index(date)
    open_ = pd.DataFrame(data.Open).set_index(date)

    close_shift = close_.shift(1)
    open_shift = open_.shift(-1)
    
    dataset = pd.concat([close_shift, open_shift], axis = 1)
    dataset["OvernightReturn"] = np.log(dataset.Open/dataset.Close)
    
    average_overnight = dataset.OvernightReturn.mean()
    dataset["Sigma2_o"] = (dataset.OvernightReturn - average_overnight)**2
    
    
    dataset2 = pd.concat([close_, open_], axis = 1)
    dataset2["CloseOpenRet"] = np.log(dataset2.Close/dataset2.Open)
    
    average_closeopen = dataset2.CloseOpenRet.mean()
    dataset2["Sigma2_c"] = (dataset2.CloseOpenRet - average_closeopen)**2
    
    
    high, low = pd.DataFrame(data.High).set_index(date), pd.DataFrame(data.Low).set_index(date)
    dataset3 = pd.concat([close_, open_, high, low], axis = 1)
    
    dataset3["one"] = np.log(dataset3.High/dataset3.Close)
    dataset3["two"] = np.log(dataset3.High/dataset3.Open)

    dataset3["three"] = np.log(dataset3.Low/dataset3.Close)
    dataset3["four"] = np.log(dataset3.Low/dataset3.Open)
    #print(dataset3)
    dataset3["Sigma2_RS"] = dataset3["one"] * dataset3.two + dataset3.three*dataset3.four  
    
    vol_df = pd.concat([dataset["Sigma2_o"], dataset2["Sigma2_c"] ,dataset3["Sigma2_RS"]], axis = 1)
    
    return np.sqrt( vol_df.Sigma2_o + k*vol_df.Sigma2_c + (1-k)*vol_df.Sigma2_RS  )                    
    
    
    
    
    
    
    
    