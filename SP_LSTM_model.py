import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as data
import yfinance as yfin
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2024-01-20'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

yfin.pdr_override()
df = data.get_data_yahoo(user_input, start, end)

#describe data
st.subheader('DATA FROM 2010-2024')
st.write(df.describe())


#Visualization
st.subheader('closing price vs time Plot')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

#MOving Average
st.subheader('Moving Average vs Time Plot')
mov100 = df.Close.rolling(100).mean()
mov200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(mov100, 'r')
plt.plot(mov200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)


#ML model

#spliting data and scaling into x and y train
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i,0])

x_train = np.array(x_train)
y_train = np.array(y_train)


#Now, lets load the model we previously trained on JN 
model = load_model('keras_model.h5')

#feed data into model and Test

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test = np.array(x_test)
y_test = np.array(y_test)

# MAKING PREDICTIONS
y_predicted = model.predict(x_test)

#scale them up to check it out
scaler = scaler.scale_ #this is the scale by what everything was scaled
scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor


#Final PLot
st.subheader('Prediction vs original stock closing price ')
st.info('Prediction on Test Data, so tail 0.3 of total time')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.plot(y_test, label = 'Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Apple Stock Prediction using LSTM Neural Network')
plt.legend()
st.pyplot(fig2)