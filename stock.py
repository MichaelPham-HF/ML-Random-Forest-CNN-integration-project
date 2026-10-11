import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sb
from datetime import datetime
from tensorflow import keras 

os.environ["TP_CPP_MIN_LOG_LEVEL"] = "2"
microsoft = pd.read_csv("MicrosoftStock.csv")

plt.figure(figsize = (12, 6))
plt.plot(microsoft["date"], microsoft["open"], label = "Open", color = 'blue')
plt.plot(microsoft["date"], microsoft["close"], label = "Close", color = 'red')
plt.title("Open-Close Prices")
plt.legend()
# plt.show()

plt.figure(figsize = (12, 6))
plt.plot(microsoft["date"], microsoft["volume"], label = "Volume", color = 'green')
plt.title("Volume Over Time")
# plt.show()

#convert to datetime
microsoft["date"] = pd.to_datetime(microsoft["date"])

#filter date to focus on year range

prediction = microsoft.loc[
    (microsoft["date"] > datetime(2013, 1, 1)) &
    (microsoft["date"] < datetime(2018, 1, 1))
]

#remove/get the closing price data
msft_close = microsoft.filter(['close'])
dataset = msft_close.values #convert into numpy
training_data_length = int(np.ceil(len(dataset)*0.95))

#preprocessing data with scikit
sc = StandardScaler()
scaled_data = sc.fit_transform(dataset)

# create green flashcards
train_data = scaled_data[:training_data_length]
x_train, y_train = [], []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

#convert into array
x_train, y_train = np.array(x_train), np.array(y_train)

#convert arrat into 3d array
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#build our model

reshaped = x_train.reshape(-1, 1)
print(reshaped)

model = keras.models.Sequential()
#Layer 1
model.add(keras.layers.LSTM(64, return_sequences = True, input_shape = (x_train.shape[1], 1)))

#Layer 2
model.add(keras.layers.LSTM(64, return_sequences = False))

#Layer 3 (dense layer)
model.add(keras.layers.Dense(128, activation = "relu"))

#Layer 4 (Dropout)
model.add(keras.layers.Dropout(0.5))

#Final layer
model.add(keras.layers.Dense(1))

model.compile(optimizer = "adam", loss = "mae", metrics = [keras.metrics.RootMeanSquaredError()])

#Train the model
history = model.fit(x_train, y_train, epochs = 20, batch_size = 32)

test_data = scaled_data[training_data_length - 60:]
x_test, y_test = [], dataset[training_data_length:]

# Generate test sequences using the same sliding window approach
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert to numpy array and reshape for LSTM input
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions using the trained LSTM model
predictions = model.predict(x_test)

# Inverse transform predictions to return to the original scale
predictions = sc.inverse_transform(predictions)

# Separate train and test portions for plotting
train = microsoft[:training_data_length]
test = microsoft[training_data_length:]

# Copy test data to avoid modifying original DataFrame (avoids SettingWithCopyWarning)
test = test.copy()

# Add predictions to the test DataFrame for comparison
test['Predictions'] = predictions

# Plot actual vs predicted stock prices
plt.figure(figsize=(12, 8))
plt.plot(train['date'], train['close'], label='Train (Actual)', color='blue')
plt.plot(test['date'], test['close'], label='Test (Actual)', color='green')
plt.plot(test['date'], test['Predictions'], label='Predictions', color='red')
plt.title('Microsoft Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()