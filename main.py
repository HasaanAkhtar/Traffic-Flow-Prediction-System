"""
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).
"""
import math
import warnings

from datetime import *

import numpy as np
import pandas as pd
from numpy import concatenate
from Value import Value
from sklearn.metrics import mean_squared_error
from data.data import process_data
from data.proccess import process_intersection_data, process_all_data_with_lag
from data.proccess import process_all_data
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
warnings.filterwarnings("ignore")


def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """

    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape


def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted result.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """

    # mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance_score:%f' % vs)
    # print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)


def plot_results(y_true, y_preds, names, date='', location="", periods=192):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, true data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    d = '2006-10-1 00:00'
    x = pd.date_range(d, periods=periods, freq='15min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)

    plt.legend()
    plt.title(location + str(date))
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    date_format = mpl.dates.DateFormatter("%Y/%m/%d - %H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()


# round up the time to fill with the data
def ceilDt(dt, delta):
    return datetime.min + math.ceil((dt - datetime.min) / delta) * delta


def floorDt(dt, delta):
    rounded = dt - (dt - datetime.min) % delta
    return rounded


# get current time
def getTime():
    now = datetime.now()

    # now = now - timedelta(minutes=now.minute % 15, seconds = now.second, microseconds= now.microsecond)
    if now.minute in range(0, 9):
        now = floorDt(now, timedelta(minutes=15))
    else:
        now = ceilDt(now, timedelta(minutes=15))
    print(now)

    return now


'''
def main():
    lstm = load_model('model/lstm.h5')
    gru = load_model('model/gru.h5')
    saes = load_model('model/saes.h5')
    models = [lstm, gru, saes]
    names = ['LSTM', 'GRU', 'SAEs']

    lag = 12
    file1 = 'data/train.csv'
    file2 = 'data/test.csv'
    _, _, X_test, y_test, scaler = process_data(file1, file2, lag)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

    y_preds = []
    for name, model in zip(names, models):
        if name == 'SAEs':
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        else:
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        file = 'images/' + name + '.png'
        plot_model(model, to_file=file, show_shapes=True)
        predicted = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
        y_preds.append(predicted[:288])
        print(name)
        eva_regress(y_test, predicted)

    plot_results(y_test[: 288], y_preds, names)
'''


# testing purpose
# one intersection only using the given model
# lag = 12
def intersection():
    lstm = load_model('model/lstm-intersection.h5')
    gru = load_model('model/gru-intersection.h5')
    saes = load_model('model/saes-intersection.h5')
    models = [lstm, gru, saes]
    names = ['LSTM', 'GRU', 'SAEs']
    file1 = 'data/boroondara.xls'
    lag = 12
    _, _, X_test, y_test, scaler = process_intersection_data(file1, lag)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]
    y_preds = []

    for name, model in zip(names, models):
        if name == 'SAEs':
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        else:
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        file = 'images/' + name + '-intersection.png'
        plot_model(model, to_file=file, show_shapes=True)
        predicted = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
        y_preds.append(predicted[:192])
        print(name)
        eva_regress(y_test, predicted)
    plot_results(y_test[:192], y_preds, names)


# get record
# planned to use date for the search as well


def searchRecord(data, location):
    """search
        #intended to search for the previous 15 minute data to predict the next 15 minute
        #but the front end only sent in the location list without time

        # Arguments
        data: dataframe of whole spreed sheet
        location: list of intersection name

        return dataframe for prediction
    """

    location_data = pd.DataFrame(columns=["value", "Location", "Date"])
    test = pd.DataFrame(columns=["value", "Location", "Date"])
    for item in location:
        test = test.append(data.loc[(data['Location'] == item)], ignore_index=True)
        location_data = location_data.append(test.sample())
        test = test.iloc[0:0]

    return location_data


def search(location):
    """prediction given a list of location
        #model used lstm with lag = 1

        #argument:
        location: list of intersection name

        return a dictionary with intersection name and prediction value
    """
    lstm = load_model('model/lstm-all.h5')
    file1 = 'data/boroondara.xls'
    lag = 1
    feature = 3
    _, _, _, _, scaler, location_encoder, date_encoder, data = process_all_data(file1)
    # search through the spreed sheet to get the real data
    test = searchRecord(data, location)
    printout = test

    # date not relavent
    printout = printout.drop(columns=['Date'])

    print("Real data")
    # print out the data
    print(printout)
    #time.sleep(300)
    y_test = Value(test)

    # reshape data for model to predict
    y_test.reshapeData(scaler, location_encoder, date_encoder)

    # predict the flow
    predicted_data = Value(lstm.predict(y_test.data))

    # reshape the data back to comparison with real data
    predicted_data.reverseReshape(y_test.data)

    # reverse scaling for the traffic value
    predicted = scaler.inverse_transform(predicted_data.data)

    # put predicted data in dataframe
    predicted = pd.DataFrame(data=predicted, columns=["value", "Location", "Date"])

    # reverse encoding for date and location
    predicted["Location"] = location_encoder.inverse_transform(predicted["Location"].astype(int))
    predicted["Date"] = date_encoder.inverse_transform(predicted["Date"].astype(int))
    predicted["value"] = predicted["value"].astype(int)
    # eva_regress(cal_test, predicted["value"])

    # priniting out the predicted data
    print("PREDICTED traffic flow ")
    predicted = predicted.drop(columns=['Date'])
    print(predicted)
    #time.sleep(300)
    # put predicted data in a dictionary for the path finding algorithm to use
    location_name = predicted['Location']
    location_name = location_name.tolist()

    value = predicted['value']
    value = value.tolist()
    location_value = {}
    for i in range(0, 4):
        location_value[location_name[i]] = value[i]

    # return the dictionary
    return location_value


def searchOneIntersection():
    """prediction for one intersection
            #for testing purpose
            #lag is 1
            #model used lstm,gru with 3 features location, traffic flow, and date
    """
    lstm = load_model('model/lstm-all.h5')
    gru = load_model('model/gru-all.h5')
    models = [lstm, gru]
    names = ['LSTM', 'GRU']

    file1 = 'data/boroondara.xls'

    lag = 1
    feature = 3
    y_preds = []
    _, _, X_test, y_test, scaler, location_encoder, date_encoder, data = process_all_data(file1, lag, feature)

    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[2]))
    predicted = lstm.predict(X_test)
    predicted = concatenate((predicted, X_test_reshaped[:, 1:]), axis=1)
    predicted = scaler.inverse_transform(predicted)

    location_name = input("Enter the intersection\n")
    y_test = y_test.reshape((len(y_test), 1))
    y_test = concatenate((y_test, X_test_reshaped[:, 1:]), axis=1)
    y_test = scaler.inverse_transform(y_test)
    # unencode the location
    y_test_one_location = pd.DataFrame(data=y_test, columns=["value", "Location", "Date"])
    y_test_one_location["Location"] = location_encoder.inverse_transform(y_test_one_location["Location"].astype(int))
    y_test_one_location["Date"] = date_encoder.inverse_transform(y_test_one_location["Date"].astype(int))
    y_test_one_location["value"] = y_test_one_location["value"].astype(int)
    y_test_one_location = y_test_one_location.loc[y_test_one_location['Location'] == location_name]
    y_test_one_location['Date'] = pd.to_datetime(y_test_one_location['Date'])
    y_test_one_location = y_test_one_location.sort_values(by=['Date'])

    y_test = y_test_one_location['value']
    for name, model in zip(names, models):
        # get predicted data of the intersection
        predicted = model.predict(X_test)
        # invert scaling for forecast
        predicted = concatenate((predicted, X_test_reshaped[:, -2:]), axis=1)
        predicted = scaler.inverse_transform(predicted)
        predicted = pd.DataFrame(data=predicted, columns=["value", "Location", "Date"])
        predicted["Location"] = location_encoder.inverse_transform(predicted["Location"].astype(int))
        predicted["Date"] = date_encoder.inverse_transform(predicted["Date"].astype(int))
        predicted["value"] = predicted["value"].astype(int)
        x_one_location = predicted.loc[predicted['Location'] == location_name]
        x_one_location['Date'] = pd.to_datetime(x_one_location['Date'])
        x_one_location = x_one_location.sort_values(by=['Date'])
        predicted = x_one_location["value"]
        # invert scaling for actual
        y_preds.append(predicted[:672])
        print(name)
        eva_regress(y_test, predicted)
    dates = x_one_location["Date"].iloc[0]
    plot_results(y_test[:672], y_preds, names, dates, location_name, 672)


def searchOneIntersectionLag():
    """prediction for one intersection
            #for testing purpose
            #lag is 12
            #model used lstm,gru with 3 features location, traffic flow, and date
    """
    lstm = load_model('model/lstm-lag-all.h5')
    gru = load_model('model/gru-lag-all.h5')
    models = [lstm, gru]
    names = ['LSTM', 'GRU']

    file1 = 'data/boroondara.xls'

    lag = 12
    feature = 3
    y_preds = []
    _, _, X_test, y_test, scaler, location_encoder, date_encoder, data = process_all_data_with_lag(file1, lag, feature)

    X_test_reshaped = X_test.reshape((X_test.shape[0], lag * feature))

    location_name = input("Enter the intersection\n")

    y_test = y_test.reshape((len(y_test), 1))
    y_test = concatenate((y_test, X_test_reshaped[:, -2:]), axis=1)
    y_test = scaler.inverse_transform(y_test)
    # unencode the location
    y_test_one_location = pd.DataFrame(data=y_test, columns=["value", "Location", "Date"])
    y_test_one_location["Location"] = location_encoder.inverse_transform(y_test_one_location["Location"].astype(int))
    y_test_one_location["Date"] = date_encoder.inverse_transform(y_test_one_location["Date"].astype(int))
    y_test_one_location["value"] = y_test_one_location["value"].astype(int)
    y_test_one_location = y_test_one_location.loc[y_test_one_location['Location'] == location_name]
    y_test_one_location['Date'] = pd.to_datetime(y_test_one_location['Date'])
    y_test_one_location = y_test_one_location.sort_values(by=['Date'])

    y_test = y_test_one_location['value']
    for name, model in zip(names, models):
        # get predicted data of the intersection
        predicted = model.predict(X_test)
        # invert scaling for forecast
        predicted = concatenate((predicted, X_test_reshaped[:, -2:]), axis=1)
        predicted = scaler.inverse_transform(predicted)
        predicted = pd.DataFrame(data=predicted, columns=["value", "Location", "Date"])
        predicted["Location"] = location_encoder.inverse_transform(predicted["Location"].astype(int))
        predicted["Date"] = date_encoder.inverse_transform(predicted["Date"].astype(int))
        predicted["value"] = predicted["value"].astype(int)
        x_one_location = predicted.loc[predicted['Location'] == location_name]
        x_one_location['Date'] = pd.to_datetime(x_one_location['Date'])
        x_one_location = x_one_location.sort_values(by=['Date'])
        predicted = x_one_location["value"]
        # invert scaling for actual
        y_preds.append(predicted[:672])
        print(name)
        eva_regress(y_test, predicted)
    dates = x_one_location["Date"].iloc[0]
    plot_results(y_test[:672], y_preds, names, dates, location_name, 672)


# testing purpose
# lag is 1
# test all intersection
def all():
    """prediction for all intersection
            #testing purpose
            #lag is 1
            #test all intersection
    """
    lstm = load_model('model/lstm-all.h5')
    gru = load_model('model/gru-all.h5')
    models = [lstm, gru]
    names = ['LSTM', 'GRU']
    file1 = 'data/boroondara.xls'

    lag = 1
    feature = 3
    _, _, X_test, y_test, scaler, location_encoder, date_encoder, data = process_all_data(file1, lag, feature)

    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[2]))
    # print(X_test_reshaped.shape)

    y_test = y_test.reshape((len(y_test), 1))
    y_test = concatenate((y_test, X_test_reshaped[:, 1:]), axis=1)
    y_test = scaler.inverse_transform(y_test)
    y_test = y_test[:, 0]
    y_preds = []
    for name, model in zip(names, models):
        predicted = model.predict(X_test)
        # invert scaling for forecast
        predicted = concatenate((predicted, X_test_reshaped[:, 1:]), axis=1)

        predicted = scaler.inverse_transform(predicted)
        predicted = predicted[:, 0]
        # invert scaling for actual
        y_preds.append(predicted[:672])
        print(name)
        eva_regress(y_test, predicted)
    plot_results(y_test[:672], y_preds, names, periods=672)


def all_with_lag():
    """prediction for all intersection
                #testing purpose
                #lag is 12
                #test all intersection
        """
    lstm = load_model('model/lstm-lag-all.h5')
    gru = load_model('model/gru-lag-all.h5')
    models = [lstm, gru]
    names = ['LSTM', 'GRU']
    file1 = 'data/boroondara.xls'

    lag = 12
    feature = 3
    _, _, X_test, y_test, scaler, location_encoder, date_encoder, data = process_all_data_with_lag(file1, lag, feature)

    X_test_reshaped = X_test.reshape((X_test.shape[0], lag * feature))
    # print(X_test_reshaped.shape)
    y_test = y_test.reshape((len(y_test), 1))
    y_test = concatenate((y_test, X_test_reshaped[:, -2:]), axis=1)
    y_test = scaler.inverse_transform(y_test)
    y_test = y_test[:, 0]
    y_preds = []
    for name, model in zip(names, models):
        predicted = model.predict(X_test)
        # invert scaling for forecast
        predicted = concatenate((predicted, X_test_reshaped[:, -2:]), axis=1)
        predicted = scaler.inverse_transform(predicted)
        predicted = predicted[:, 0]
        # invert scaling for actual
        y_preds.append(predicted[:672])
        print(name)
        eva_regress(y_test, predicted)
    plot_results(y_test[:672], y_preds, names, periods=672)


if __name__ == '__main__':
    search(['RIVERSDALE_RD W of BURKE_RD', 'BALWYN_RD S of DONCASTER_RD', 'BALWYN_RD N OF BELMORE_RD',
            'WARRIGAL_RD S OF RIVERSDALE_RD',
            'BURKE_RD S of EASTERN_FWY', 'TRAFALGAR_RD S of RIVERSDALE_RD', 'WHITEHORSE_RD E OF BURKE_RD'])

    # searchOneIntersection()
    # searchOneIntersectionLag()
    # all()
    # all_with_lag()
    # intersection()
