import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values

    if dropnan:
        agg.dropna(inplace=True)
    return agg


def process_intersection_data(train, lags):
    xl = pd.ExcelFile(train)
    xl.sheet_names
    [u'Notes', u'Data', u'Summary Of Data']
    df = xl.parse("Data")

    mylist = []
    for num in range(0, 96):
        if (num < 10):
            mylist.append('V0' + str(num))
        else:
            mylist.append('V' + str(num))

    columns = ['SCATS Number', 'Location', 'CD_MELWAY', 'NB_LATITUDE',
               'NB_LONGITUDE', 'HF VicRoads Internal',
               'VR Internal Stat', 'VR Internal Loc', 'NB_TYPE_SURVEY', 'Date']
    for value in mylist:
        columns.append(value)
    df.columns = columns

    war_rd = df.iloc[1:500, 9:]
    war_rd_location = df.iloc[1:500, 1:2]
    data = war_rd_location.join(war_rd)
    data = data.melt(id_vars=['Location', 'Date'], value_name='value')
    data = data.sort_values(by=['Date', 'variable'])
    data = data.drop(columns=['variable', 'Date'])

    test_war_rd = df.iloc[501:750, 9:]
    test_war_rd_location = df.iloc[501:750, 1:2]
    data2 = test_war_rd_location.join(test_war_rd)
    data2 = data2.melt(id_vars=['Location', 'Date'], value_name='value')
    data2 = data2.sort_values(by=['Date', 'variable'])
    data2 = data2.drop(columns=['variable', 'Date'])

    # scaler = StandardScaler().fit(df1[attr].values)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(data['value'].values.reshape(-1, 1))
    flow1 = scaler.transform(data['value'].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(data2['value'].values.reshape(-1, 1)).reshape(1, -1)[0]

    train, test = [], []
    for i in range(lags, len(flow1)):
        train.append(flow1[i - lags: i + 1])
    for i in range(lags, len(flow2)):
        test.append(flow2[i - lags: i + 1])

    train = np.array(train)
    test = np.array(test)
    np.random.shuffle(train)
    np.random.shuffle(test)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]

    return X_train, y_train, X_test, y_test, scaler

def process_all_data(train):
    xl = pd.ExcelFile(train)
    xl.sheet_names
    [u'Notes', u'Data', u'Summary Of Data']
    df = xl.parse("Data")

    list = df.columns[10:]
    columns = ['SCATS Number', 'Location', 'CD_MELWAY', 'NB_LATITUDE',
               'NB_LONGITUDE', 'HF VicRoads Internal', 'VR Internal Stat',
               'VR Internal Loc', 'NB_TYPE_SURVEY', 'Date']
    for value in list:
        columns.append(value)
    df.columns = columns

    # drop columns that are not used
    data = df.drop(columns={'SCATS Number', 'CD_MELWAY', 'NB_LATITUDE',
                            'NB_LONGITUDE', 'HF VicRoads Internal',
                            'VR Internal Stat',
                            'VR Internal Loc', 'NB_TYPE_SURVEY'})

    # drop first row
    data = data.iloc[1:, :]

    # reformat the data
    locations = data['Location'].unique()
    data = data.melt(id_vars=['Location', 'Date'], var_name='Time',
                     value_name='value')
    data['Location'] = pd.Categorical(data['Location'], categories=locations,
                                      ordered=True)
    data = data.sort_values(by=['Location', 'Date', 'Time'])

    #combine date and time
    data['Date'] = data['Date'].dt.strftime('%d/%m/%Y')
    data.loc[:,'Date'] = data.Date.astype(str)+' '+data.Time.astype(str)

    #change index to date and time
    #data.index = pd.to_datetime(data['Date'])

    #drop date and time column
    data = data.drop(columns={'Time'})
    data = data[['value', 'Location','Date']]


    # load values of the dataset
    values = data.values

    # encoder the name using labelEncoder
    location_encoder = LabelEncoder()
    date_encoder = LabelEncoder()
    values[:, 1] = location_encoder.fit_transform(values[:, 1])
    values[:, 2] = date_encoder.fit_transform(values[:, 2])

    # ensure all data is float
    values = values.astype('float32')

    # normailze features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # frame data as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)

    #drop columns location column
    reframed.drop(reframed.columns[[4,5]], axis=1, inplace=True)
    # split into train and test sets
    values = reframed.values

    np.random.shuffle(values)
    train_split = 0.8

    num_train = int(train_split * values.size)

    train = values[:280000, :]
    test = values[280000:, :]

    X_train = train[:, :-1]

    y_train = train[:, -1]
    X_test = test[:, :-1]

    y_test = test[:, -1]

    # reshape input to be 3D [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    return X_train, y_train, X_test, y_test, scaler, location_encoder, date_encoder, data

def process_all_data_with_lag(train, lags, features):
    xl = pd.ExcelFile(train)
    xl.sheet_names
    [u'Notes', u'Data', u'Summary Of Data']
    df = xl.parse("Data")

    list = df.columns[10:]
    columns = ['SCATS Number', 'Location', 'CD_MELWAY', 'NB_LATITUDE',
               'NB_LONGITUDE', 'HF VicRoads Internal', 'VR Internal Stat',
               'VR Internal Loc', 'NB_TYPE_SURVEY', 'Date']
    for value in list:
        columns.append(value)
    df.columns = columns

    # drop columns that are not used
    data = df.drop(columns={'SCATS Number', 'CD_MELWAY', 'NB_LATITUDE',
                            'NB_LONGITUDE', 'HF VicRoads Internal',
                            'VR Internal Stat',
                            'VR Internal Loc', 'NB_TYPE_SURVEY'})

    # drop first row
    data = data.iloc[1:, :]

    # reformat the data
    locations = data['Location'].unique()
    data = data.melt(id_vars=['Location', 'Date'], var_name='Time',
                     value_name='value')
    data['Location'] = pd.Categorical(data['Location'], categories=locations,
                                      ordered=True)
    data = data.sort_values(by=['Location', 'Date', 'Time'])

    #combine date and time
    data['Date'] = data['Date'].dt.strftime('%d/%m/%Y')
    data.loc[:,'Date'] = data.Date.astype(str)+' '+data.Time.astype(str)

    #change index to date and time
    #data.index = pd.to_datetime(data['Date'])

    #drop date and time column
    data = data.drop(columns={'Time'})
    data = data[['value', 'Location','Date']]


    # load values of the dataset
    values = data.values

    # encoder the name using labelEncoder
    location_encoder = LabelEncoder()
    date_encoder = LabelEncoder()
    values[:, 1] = location_encoder.fit_transform(values[:, 1])
    values[:, 2] = date_encoder.fit_transform(values[:, 2])

    # ensure all data is float
    values = values.astype('float32')

    # normailze features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # frame data as supervised learning
    reframed = series_to_supervised(scaled, lags, 1)

    #drop columns location column
    #reframed.drop(reframed.columns[[4,5]], axis=1, inplace=True)
    # split into train and test sets
    values = reframed.values

    np.random.shuffle(values)
    train = values[:280000, :]
    test = values[280000:, :]

    n_obs = lags * features
    X_train = train[:, :n_obs]

    y_train = train[:, -features]
    X_test = test[:, :n_obs]

    y_test = test[:, -features]

    # reshape input to be 3D [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], lags, features))
    X_test = X_test.reshape((X_test.shape[0], lags, features))

    return X_train, y_train, X_test, y_test, scaler, location_encoder, date_encoder, data

if __name__ == '__main__':
    process_all_data_with_lag("boroondara.xls", 12, 2)
