import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import numpy as np
plt.rcParams['figure.dpi'] = 227

gain = lambda x: x if x > 0 else 0
loss = lambda x: abs(x) if x < 0 else 0

def display_predictions():
    pass

def MACrossOver(data, shortWindow, longWindow):
  signals = pd.DataFrame(index=data.index)
  signals['tradeSignal'] = 0.0
  signals['MA20'] = data['Close'].rolling(window=shortWindow, 
                                                min_periods=1, center=False).mean()
  signals['MA100'] = data['Close'].rolling(window=longWindow, 
                                                         min_periods=1, center=False).mean()
  signals['tradeSignal'][shortWindow:] = np.where(signals['MA20'][shortWindow:] > signals['MA100'][shortWindow:], 1.0, 0.0)
  signals['finalSignal'] = signals['tradeSignal'].diff()
  return signals


def buynsell(newSeries, data):
  newSeries = newSeries['2021-01-01':]
  DATA = data['2021-01-01':]
  fig = plt.figure(figsize= (15,6))
  ax1 = fig.add_subplot(111)
  DATA["Adj Close"].plot(ax=ax1, lw=.6)
  newSeries["MA20"].plot(ax=ax1, lw=2.)
  newSeries["MA100"].plot(ax=ax1,  lw=2.)
  ax1.plot(newSeries.loc[newSeries.finalSignal== 1.0].index, DATA["Adj Close"][newSeries.finalSignal == 1.0],'^', markersize=7, color='green')
  ax1.plot(newSeries.loc[newSeries.finalSignal== -1.0].index, DATA["Adj Close"][newSeries.finalSignal == -1.0],'v', markersize=7, color='red')
  plt.legend(["Adj Close Price","Short mavg","Long mavg","BuySignal","SellSignal"])
  plt.title("Simple Moving Average Crossover Strategy")
  st.pyplot()
  #plt.show()



def bollinger_bands(stock, std=2):    
    
    # Bollinger band plot with EMA and original historical data
    plt.figure(figsize=(16,5))
    plt.style.use('seaborn-whitegrid')
    plt.plot(stock.index, stock.Close, color='#3388cf', label='Price')
    plt.plot(stock.index, stock.MA21, color='#ad6eff', label='Moving Average (21 days)')
    #plt.plot(stock.index, stock.MA7, color='#ff6e9d', label='Moving Average (7 days)')
    plt.plot(stock.index, stock.Upper_band, color='#ffbd74', alpha=0.3)
    plt.plot(stock.index, stock.Lower_band, color='#ffa33f', alpha=0.3)
    plt.fill_between(stock.index, stock.Upper_band, stock.Lower_band, color='#ffa33f', alpha=0.1, label='Bollinger Band ({} STD)'.format(std))
    plt.legend(frameon=True, loc=1, ncol=1, fontsize=10, borderpad=.6)
    plt.title('Bollinger Bands', fontSize=15)
    plt.ylabel('Price', fontSize=12)
    plt.xlim([stock.index.min(), stock.index.max()])
    #plt.show()
    st.pyplot()

def volume(stock):
    # Volume plot
    plt.figure(figsize=(16,2))
    plt.style.use('seaborn-whitegrid')
    plt.title('Volume', fontSize=15)
    plt.ylabel('Volume', fontSize=12)
    plt.plot(stock.index, stock['Volume'].ewm(21).mean())
    plt.xlim([stock.index.min(), stock.index.max()])
    plt.show()    

def macd(stock):
    # MACD
    plt.figure(figsize=(16,2))
    plt.plot(stock.MACD, label='MACD', color = '#b278ff')
    plt.plot(stock.Signal, label='Signal', color='#ffa74a')
    plt.axhline(0, color='#557692')
    plt.legend(frameon=True, loc=1, ncol=1, fontsize=10, borderpad=.6)
    plt.title('MACD', fontSize=15)
    plt.ylabel('Strength', fontSize=12)
    st.pyplot()
    #plt.show()    

def rsi(stock):
    # RSI
    plt.figure(figsize=(16,2)) 
    plt.plot(stock.index, stock.RSI, color='#ad6eff')
    plt.xlim([stock.index.min(), stock.index.max()])
    plt.axhline(20, color='#f9989c')
    plt.axhline(80, color='#60e8ad')
    plt.title('RSI', fontSize=15)
    plt.ylabel('%', fontSize=12)
    plt.ylim([0, 100])
    plt.show()
    
def hist(data, name, bins=50):
    plt.rcParams['figure.dpi'] = 227
    plt.figure(figsize=(16,6))
    plt.style.use('seaborn-whitegrid')
    plt.hist(data, bins=bins)
    plt.title(name, fontSize=16)
    plt.xlabel('Values', fontSize=13)
    plt.ylabel('Quantities', fontSize=13)
    plt.show()
    
def qqplot(data):
    plt.rcParams['figure.dpi'] = 227
    plt.figure(figsize=(16,6))
    plt.style.use('seaborn-whitegrid')
    sm.qqplot(data.dropna(), line='s', scale=1)
    plt.title('Check for Normality', fontSize=16)
    plt.show()
    
def compare_stocks(stocks, value='Close', by='month', scatter=False):
    '''
    Function groups stocks' Close values
    '''
    plt.rcParams['figure.dpi'] = 227
    plt.figure(figsize=(16,6))
    plt.style.use('seaborn-whitegrid')
    group_by_stock = {}
    
    for stock in list(stocks.keys()): 
        
        if by == 'month': group_by = stocks[stock].index.month
        if by == 'day': group_by = stocks[stock].index.day
        if by == 'year': group_by = stocks[stock].index.year
        
        a = stocks[stock].groupby(group_by).mean()[value]
        normalized_price = (a-a.mean())/a.std()
        group_by_stock[stock] = normalized_price
        
        if scatter == False:
            plt.plot(normalized_price, label=stock)
        else:
            plt.scatter(normalized_price.keys(), normalized_price.values, label=stock)       
    
    plt.plot(pd.DataFrame(group_by_stock).mean(axis=1), label='ALL', color='black', linewidth=5, linestyle='--')    
    plt.legend(frameon=True, fancybox=True, framealpha=.9, loc=1, ncol=4, fontsize=12, title='Stocks')
    plt.title(value+' by '+by, fontSize=14)
    plt.xlabel('Period', fontSize=12)
    plt.ylabel(value, fontSize=12)
    plt.show()
    
    
def trading_history(stock, net, std=2):    
    
    # Bollinger band plot with EMA and original historical data
    plt.figure(figsize=(16,5))
    plt.style.use('seaborn-whitegrid')
    plt.plot(stock.index, stock.Close, color='#3388cf', label='Price')
    plt.plot(stock.index, stock.MA21, color='#ad6eff', label='Moving Average (21 days)')
    plt.plot(stock.index, stock.Upper_band, color='#ffbd74', alpha=0.3)
    plt.plot(stock.index, stock.Lower_band, color='#ffa33f', alpha=0.3)
    plt.fill_between(stock.index, stock.Upper_band, stock.Lower_band, color='#ffa33f', alpha=0.1, label='Bollinger Band ({} STD)'.format(std))
    
    plt.title('Trading History', fontSize=15)
    plt.ylabel('Price', fontSize=12)
    plt.xlim([stock.index.min(), stock.index.max()])
    
    for i in net:
        if i[2] == 1: color = '#ff005e'
        else: color = '#4bd81d'
        plt.plot_date(i[0], i[1], color=color)
        
    plt.plot_date([],[],label='Buy', c='#ff005e')
    plt.plot_date([],[],label='Sell', c='#4bd81d')
        
    plt.legend(frameon=True, loc=1, ncol=1, fontsize=10, borderpad=.6)
    plt.show()