import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)

def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

def predict(stockName):
  df=pd.read_csv('stocks/'+ stockName + '.csv')
  df.head()

  df1 = df.reset_index()['Close']

  model = load_model('models/' + stockName + '.h5')

  scaler=MinMaxScaler(feature_range=(0,1))
  df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

  training_size=int(len(df1)*0.65)
  test_size=len(df1)-training_size
  train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

  time_step = 100
  X_train, y_train = create_dataset(train_data, time_step)
  X_test, ytest = create_dataset(test_data, time_step)

  X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
  X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

  train_predict=model.predict(X_train)
  test_predict=model.predict(X_test)

  train_predict=scaler.inverse_transform(train_predict)
  test_predict=scaler.inverse_transform(test_predict)

  testAccuracy = 100 - (math.sqrt(mean_squared_error(ytest,test_predict))/100)
  trainAccuracy = 100 - (math.sqrt(mean_squared_error(y_train,train_predict))/100)

  print ("Train Accuracy = ", trainAccuracy, "%")
  print ("Test Accuracy = ", testAccuracy, "%\n")
  

  look_back=100
  trainPredictPlot = np.empty_like(df1)
  trainPredictPlot[:, :] = np.nan
  trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
  # shift test predictions for plotting
  testPredictPlot = np.empty_like(df1)
  testPredictPlot[:, :] = np.nan
  testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
  # plot baseline and predictions
  plt.plot(scaler.inverse_transform(df1), color = "lightblue", label = "All Values")
  plt.plot(trainPredictPlot, color = "darkgreen", label = "Train Predicted")
  plt.plot(testPredictPlot, color = "orange", label = "Test Values")
  plt.xlabel("Days")
  plt.ylabel("Cost of share (in INR)")
  plt.legend()
  #plt.show()
  st.pyplot()
  
  st.write("Train Accuracy = ", trainAccuracy, "%")
  st.write("Test Accuracy = ", testAccuracy, "%\n")

  x_input=test_data[325:].reshape(1,-1)
  x_input.shape

  temp_input=list(x_input)
  temp_input=temp_input[0].tolist()

  lst_output=[]
  n_steps=100
  i=0
  while(i<30):
      
      if(len(temp_input)>100):
          #print(temp_input)
          x_input=np.array(temp_input[1:])
          #print("{} day input {}".format(i,x_input))
          x_input=x_input.reshape(1,-1)
          x_input = x_input.reshape((1, n_steps, 1))
          #print(x_input)
          yhat = model.predict(x_input, verbose=0)
          #print("{} day output {}".format(i,yhat))
          temp_input.extend(yhat[0].tolist())
          temp_input=temp_input[1:]
          #print(temp_input)
          lst_output.extend(yhat.tolist())
          i=i+1
      else:
          x_input = x_input.reshape((1, n_steps,1))
          yhat = model.predict(x_input, verbose=0)
          #print(yhat[0])
          temp_input.extend(yhat[0].tolist())
          #print(len(temp_input))
          lst_output.extend(yhat.tolist())
          i=i+1

  day_new=np.arange(1,101)
  day_pred=np.arange(101,131)

  print("\n")
  plt.plot(day_new,scaler.inverse_transform(df1[1114:]), label = "last 100 days")
  plt.plot(day_pred,scaler.inverse_transform(lst_output), label = "predicted 30 days")
  plt.xlabel("Days")
  plt.ylabel("Cost of share (in INR)")
  plt.legend()
  st.pyplot()

def predictEmail(stockName):
  df=pd.read_csv('stocks/' + stockName + '.csv')
  df.head()

  df1 = df.reset_index()['Close']

  model = load_model('models/' + stockName + '.h5')

  scaler=MinMaxScaler(feature_range=(0,1))
  df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

  training_size=int(len(df1)*0.65)
  test_size=len(df1)-training_size
  train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

  time_step = 100
  X_train, y_train = create_dataset(train_data, time_step)
  X_test, ytest = create_dataset(test_data, time_step)

  X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
  X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

  train_predict=model.predict(X_train)
  test_predict=model.predict(X_test)

  train_predict=scaler.inverse_transform(train_predict)
  test_predict=scaler.inverse_transform(test_predict)

  testAccuracy = 100 - (math.sqrt(mean_squared_error(ytest,test_predict))/100)
  trainAccuracy = 100 - (math.sqrt(mean_squared_error(y_train,train_predict))/100)

  print ("Train Accuracy = ", trainAccuracy, "%")
  print ("Test Accuracy = ", testAccuracy, "%\n")

  
  look_back=100
  x_input=test_data[325:].reshape(1,-1)
  x_input.shape

  temp_input=list(x_input)
  temp_input=temp_input[0].tolist()

  lst_output=[]
  n_steps=100
  i=0
  while(i<30):
      
      if(len(temp_input)>100):
          #print(temp_input)
          x_input=np.array(temp_input[1:])
          #print("{} day input {}".format(i,x_input))
          x_input=x_input.reshape(1,-1)
          x_input = x_input.reshape((1, n_steps, 1))
          #print(x_input)
          yhat = model.predict(x_input, verbose=0)
          #print("{} day output {}".format(i,yhat))
          temp_input.extend(yhat[0].tolist())
          temp_input=temp_input[1:]
          #print(temp_input)
          lst_output.extend(yhat.tolist())
          i=i+1
      else:
          x_input = x_input.reshape((1, n_steps,1))
          yhat = model.predict(x_input, verbose=0)
          #print(yhat[0])
          temp_input.extend(yhat[0].tolist())
          #print(len(temp_input))
          lst_output.extend(yhat.tolist())
          i=i+1

  print("\n")
  return scaler.inverse_transform(lst_output)[:7]

  