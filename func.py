#imports relevant dependencies
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
plt.style.use('dark_background')

def filterCloseValue(data): #filters out the close pricing value of the general stock data
    tempData = data.filter(['Close']) #get the close key
    dataSet = tempData.values #get the value (pricing)
    return dataSet #return pricing

def graphClose(data): #sketches graph of closing price in data
    plt.figure(figsize=(10, 5)) #scales the window
    plt.title('Close Price History') #declares title
    plt.plot(data['Close']) #get the date and price of closing each day
    plt.xlabel('Date', fontsize=18) #declare x-axis title
    plt.ylabel('Close Price USD ($)', fontsize=18) #declare y-axis title
    plt.show() #print graph

def scaleDataset(dataSet): #scales data and creates a training dataset and a test dataset
    training_data_len = math.ceil(len(dataSet) * .8) #gets a variable with 80% of original dataset length
    scaler = MinMaxScaler(feature_range=(0, 1)) #declares scaler as a num between 0-1
    scaled_data = scaler.fit_transform(dataSet) #fit declared scaler unto dataset
    trainData = scaled_data[0:training_data_len, :] # fits the scaled data within the training data length
    testData = scaled_data[training_data_len - 60:, :] #creates latter testing dataset
    return trainData, training_data_len, testData, scaler #returns needed variables for latter processing

def declareXY(trainData): #remodells data
    x_data = [] #declare x_data
    y_data = [] #declare y_data
    for i in range(60, len(trainData)): #for every number between 60 and trainData length
        x_data.append(trainData[i - 60:i, 0]) #creates new empty list(s)
        y_data.append(trainData[i, 0]) #creates new empty list(s)
    x_data, y_data = np.array(x_data), np.array(y_data) #converts x and y data into np arrays
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1)) #adds third dimension
    return x_data, y_data #returns relevant variables

def modelAi(x_data, y_data): #defines ai model
    model = Sequential() #add sequential layers passed in constructor
    model.add(LSTM(50, return_sequences=True, input_shape=(x_data.shape[1], 1))) #declare neurological nodes, add öayer
    model.add(LSTM(50, return_sequences=False)) #declare more neurological nodes, add layer
    model.add(Dense(25)) #adds dense layer
    model.add(Dense(1)) #adds dense layer
    model.compile(optimizer='adam', loss='mean_squared_error') #compiles model with adam and mean_squared_error loss
    model.fit(x_data, y_data, batch_size=1, epochs=1) #fits model into our x and y data
    return model #returns model

def predictValues(closedData, trainDataLen, testData, model, scaler): #predicts stock market
    x_test = [] #declares x_test
    y_test = closedData[trainDataLen:, :] #declares y_test
    for i in range(60, len(testData)): #for every number in the range 60-length data
        x_test.append(testData[i - 60:i, 0]) #creates new empty list(s)
    x_test = np.array(x_test) #Create an array.
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) #Gives a new shape to an array without changing its data.
    predictions = model.predict(x_test) #Generates output predictions for the input samples
    predictions = scaler.inverse_transform(predictions) #Apply inverse transformations in reverse order
    return x_test, y_test, predictions #return relevant variables

def plotVar(trainDataLen, data, predictions): #declares variables for plotting
    train = data[:trainDataLen] #takes first 0.8 of the data
    valid = data[trainDataLen:] #takes last 0.2 of the data
    valid['Predictions'] = predictions #sets prediction key to prediction value
    return valid, train #returns valid and train variables

def graphPredict(valid, train): #plots graph for predicted stock price
    plt.figure(figsize=(16, 8)) #declares size
    plt.title('Model') #declares title
    plt.xlabel('Date', fontsize=18) #declares x label
    plt.ylabel('Close Price USD ($)', fontsize=18) #declares y label
    plt.plot(train['Close']) #plots first 0.8
    plt.plot(valid['Predictions']) #plots prediction
    plt.show() #plots the graph