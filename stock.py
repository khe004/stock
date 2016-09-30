#!/usr/bin/env python3

from pandas_datareader import data as web
import datetime
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab 
from candlestick import pandas_candlestick_ohlc


start = datetime.datetime(2016,1,1)
end = datetime.date.today()

apple = web.DataReader("AAPL", "yahoo", start, end)
cadence = web.DataReader("CDNS", "yahoo", start, end)
synopsys = web.DataReader("SNPS", "yahoo", start, end)

pylab.rcParams['figure.figsize'] = (15, 9)
apple["Adj Close"].plot(grid = True)

pandas_candlestick_ohlc(apple)
