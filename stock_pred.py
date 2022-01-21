import streamlit as st
import pandas as pd
import smtplib
import yfinance as yf
import os
import plotting_helper
import prediction

      
def ind_email(email, str1, name, symbol, signal):
  signaltext = ""
  if(signal):
    signaltext = "Buy"
  else:
    signaltext = "Sell"
  s = smtplib.SMTP('smtp.gmail.com', 587)
  s.starttls()
  s.login("sparrowjohn390@gmail.com", "9561836477")
  message = "Hello "+ name +"!\n\nPrediction for " + symbol.upper() +" for the next 7 days: \nRs. " + str1+"\n The predicted signal for this stock is " +  signaltext +"\n\nThanks & Regards." 
  s.sendmail("sparrowjohn390@gmail.com", email, message)
  s.quit()
      
def get_input():
  start_date = st.sidebar.text_input("Start Date","2021-01-01")
  end_date = st.sidebar.text_input("End Date","2022-01-20")
  stock_symbol = st.sidebar.text_input("Stock Symbol","RELIANCE.NS")
  return start_date, end_date, stock_symbol

    
def get_data(symbol, start, end):
  data = yf.download(symbol, start=start, end=end)
  return data

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)  
    st.title("Stock App")
    menu = ["Stock", "Exploratory Analysis", "Prediction","Email Notification"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Stock":
      start, end, symbol = get_input()
      df = get_data(symbol, start, end)
      st.header(symbol+" High\n")
      st.line_chart(df['High'])
      
      st.header(symbol+" Low\n")
      st.line_chart(df['Low'])
      
      st.header(symbol+" Open Price\n")
      st.line_chart(df['Open'])
      
      st.header(symbol+" Close Price\n")
      st.line_chart(df['Close'])
      
      st.header(symbol+" Volume\n")
      st.line_chart(df['Volume'])

    
    elif choice == "Email Notification":
        st.subheader("Email Notification Section")
        name = st.text_input("Name")
        email = st.text_input("Email")
        symbol = st.text_input("Stock Symbol")
        if st.button("Send Email") :
            pred, signal = prediction.predictEmail(symbol)
            str1 =' Rs. '.join([str(round(elem[0],2))+', ' for elem in pred])
            ind_email(email, str1, name, symbol, signal)
            st.info("Email send successfully")
            
    elif choice == "Prediction" :
        st.subheader("Prediction")
        symbol = st.text_input("Stock Symbol")
        if st.button("Predict") :
            prediction.predict(symbol)
            
    elif choice == "Exploratory Analysis":
        st.subheader("Exploratory Analysis")
        symbol = st.text_input("Stock Symbol")
        if st.button("Plot") :
            files = os.listdir('stocks')
            stocks = {}
            for file in files:
                # Include only csv files
                if file.split('.')[1] == 'csv':
                    name = file.split('.')[0]
                    stocks[name] = pd.read_csv('stocks/'+file, index_col='Date')
                    stocks[name].index = pd.to_datetime(stocks[name].index)
                    
            st.header(symbol.upper() + " Bollinger Bands\n")
            plotting_helper.bollinger_bands(stocks[symbol].loc['2018':'2018'])
            
            st.header(symbol.upper() + " MACD\n")
            plotting_helper.macd(stocks[symbol].loc['2018':'2018'])
            
            st.header(symbol.upper() + " Simple Moving Average Crossover Strategy\n")
            newSeries = plotting_helper.MACrossOver(stocks[symbol],20,100)
            plotting_helper.buynsell(newSeries, stocks[symbol])
          
        

if __name__ == '__main__':
	main()
