from utils import *
import yfinance as yf
import matplotlib.pyplot as plt
from yahoo_fin import stock_info as si 
import numpy as np 
import fbm

name = "AD"
download_daily_data(name)
price = load_daily_raw_data(name)

vol = yang_zhang_vol(price)

plt.plot(vol)


#logreturn = np.log(price/price.shift(1)).dropna()


#you can pass a dataframe of logreturns or only a pd.Series with only one 
#autocorrelation(logreturn.abs(), 50, plot = True, include_zero = False, save = False )

#ecdf calculation 
#abs_ecdf(logreturn)

