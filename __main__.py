#import relevant dependencies
import func
from datetime import date
import pandas_datareader as web

today = date.today() #gets current day
#pulls first parameter microsoft stock data from yahoo finance spanning from start to end date
data = web.DataReader('MSFT', data_source='yahoo', start='2010-01-01', end=today.strftime("%Y-%B-%d"))

func.graphClose(data) #sketch graph of previous data
closedData = func.filterCloseValue(data) #filters through data for stock closing price
trainData, trainDataLen, testData, scaler = func.scaleDataset(closedData) #scales the data for training
x_data, y_data = func.declareXY(trainData) #remodells data
model = func.modelAi(x_data, y_data) #defines model
x_test, y_test, predictions = func.predictValues(closedData, trainDataLen, testData, model, scaler) #calculates prediction
valid, train = func.plotVar(trainDataLen, data, predictions) #plots prediction graph
func.graphPredict(valid, train) #plots predictions graph

# This code predicts the last 20% since 2010-01-01
# the information is readable through graphs
# made by - Hugo Mårdbrink, Axel Söderberg