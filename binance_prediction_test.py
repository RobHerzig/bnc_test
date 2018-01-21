# LSTM for international airline passengers problem with window regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from binance.client import Client
from keras.models import model_from_json
import time

api_key = "sGfyIE0zYoKkZ3M0CtnuZpf070GGkUmXrDimdyavp3FMShQkh7unnrzue9pwGEay"
api_secret = "lz9NXEITdTH6sF8UZ1IPFaWs3MjjScHhjuMvyznS27GMFQviTWqlF1RvFP0D7snl"

client = Client(api_key, api_secret)

info = client.get_account()
print(info)

from binance_startup import get_ratios_1min_24h, get_last_30min


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX), numpy.array(dataY)


def create_model(look_back, name_for_file_load):
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))

    try:
        model.load_weights(name_for_file_load + ".h5")
    except:
        print("NO WEIGHTS FOUND")
    return model


def get_prediction(model, values):
    model.predict(values)


def generate_historical_arrays(names):
    close_arrays = []
    for currency in names:
        close_array = get_ratios_1min_24h(currency)
        print(currency + " close values: " + str(close_array))
        close_arrays.append(close_array)
    return close_arrays


def save_model(name_of_model, model):
    # serialize model to JSON
    model_json = model.to_json()
    with open(name_of_model + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name_of_model + ".h5")
    print("Saved model + " + name_of_model + " to disk")


def load_model(name_of_model):
    json_file = open(name_of_model + ".json", 'r')
    print("ATTEMPTING TO LOAD " + name_of_model + ".json")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(name_of_model + ".h5")
    print("SUCCESSFULLY LOADED MODEL " + name_of_model)
    return loaded_model


scaler = MinMaxScaler(feature_range=(0, 1))


def generate_models_for_data(data, names, num_epochs=3, lookback_steps=5):
    models = []
    for i in range(0, len(data)):
        global scaler
        dataset = data[i]
        cur_name = names[i]
        dataset = dataset.astype('float32')
        dataset = dataset.reshape(-1, 1)
        # scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        train_size = int(len(dataset) * 0.67)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
        # reshape into X=t and Y=t+1
        look_back = lookback_steps
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        # reshape input to be [samples, time steps, features]
        trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        # create and fit the LSTM network
        model = create_model(look_back, cur_name)
        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['mae', 'acc'])
        model.fit(trainX, trainY, epochs=num_epochs, batch_size=1, verbose=2)

        # todo: save model
        save_model(cur_name, model)

        models.append(model)
        # make predictions
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        # invert predictions
        # trainPredict = scaler.inverse_transform(trainPredict)
        # trainY = scaler.inverse_transform(trainY)
        # testPredict = scaler.inverse_transform(testPredict)
        # testY = scaler.inverse_transform(testY)
        # # calculate root mean squared error
        # trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:]))
        # print('Train Score: %.2f RMSE' % (trainScore))
        # testScore = math.sqrt(mean_squared_error(testY, testPredict[:]))
        # print('Test Score: %.2f RMSE' % (testScore))
        # # shift train predictions for plotting
        # trainPredictPlot = numpy.empty_like(dataset)
        # trainPredictPlot[:, :] = numpy.nan
        # trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
        # # shift test predictions for plotting
        # testPredictPlot = numpy.empty_like(dataset)
        # testPredictPlot[:, :] = numpy.nan
        # testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
        # # plot baseline and predictions
        # plt.plot(scaler.inverse_transform(dataset))
        # plt.plot(trainPredictPlot)
        # plt.plot(testPredictPlot)
        # plt.show()
    return models


def load_models_without_training(names):
    models = []
    for name in names:
        print("TRYING TO LOAD " + name)
        model = load_model(name)
        models.append(model)
    return models


def predict_most_current(names, lookback, models):
    ratios = []
    for i in range(0, len(names)):
        global scaler
        new_data = get_last_30min(names[i])
        model = models[i]
        new_data = new_data[(len(new_data) - lookback): len(new_data)]
        new_data = new_data.reshape(-1, 1)
        scaled_data = scaler.fit_transform(new_data)
        scaled_data = scaled_data.reshape(1, 1, lookback)
        # print("PREDICT BASED ON: " + str(new_data))
        prediction = model.predict(scaled_data)
        prediction = scaler.inverse_transform(prediction)
        ratio = prediction / new_data[len(new_data) - 1]
        # print(names[i] + " PREDICTION: " + str(prediction))
        # print("WHICH IS " + str(ratio) + " * most recent value")
        ratios.append(ratio)
    return ratios


def get_newest_ratios(name_list, lookback_steps, models):
    cur_ratios = predict_most_current(names, lookback_steps, models)
    for i in range(0, len(name_list)):
        print(name_list[i] + " " + str(cur_ratios[i]) + " times previous")
    return cur_ratios


num_lookback_steps = 5
continuously_predict = True
train = False

names = ["TRXETH",
         "OMGETH",
         "NEOETH",
         "LRCETH",
         "AMBETH"]

generated_models = []
if (train):
    data = generate_historical_arrays(names)
    generated_models = generate_models_for_data(data, names, lookback_steps=num_lookback_steps, num_epochs=15)
else:
    try:
        generated_models = load_models_without_training(names)
    except:
        print("COULD NOT LOAD MODELS")

# predict_most_current(names, num_lookback_steps, generated_models)

get_newest_ratios(names, num_lookback_steps, generated_models)

time_interval_in_seconds = 30
if continuously_predict:
    seconds = time_interval_in_seconds
    print("PREDICT EVERY " + str(seconds) + " SECONDS")
    while True:
        # print(seconds)
        time.sleep(1)
        seconds = seconds - 1
        if seconds == 0:
            ratios = get_newest_ratios(names, num_lookback_steps, generated_models)
            seconds = time_interval_in_seconds
