import argparse
import sys
from data.proccess import process_intersection_data, process_all_data_with_lag
from data.proccess import process_all_data
from train import train_seas
from model import model
from keras.models import Model
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def train_model(model, X_train, y_train, name, config):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.33)
    model.save('model/' + name + '-all.h5')
    df = pd.DataFrame.from_dict(hist.history)
    plt.plot(hist.history['loss'], label='train')
    plt.plot(hist.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    df.to_csv('model/' + name + '-all-loss.csv', encoding='utf-8', index=False)


'''def train_seas(models, X_train, y_train, name, config):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    temp = X_train
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(input=p.input,
                                       output=p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])

        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)

        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer('hidden%d' % (i + 1)).set_weights(weights)

    train_model(saes, X_train, y_train, name, config)'''


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="lstm",
        help="Model to train.")
    args = parser.parse_args()

    lag = 12
    features = 3
    config = {"batch": 1024, "epochs": 400}
    file1 = 'data/boroondara.xls'

    # X_train, y_train, _, _, _ = process_intersection_data(file1, lag)

    if args.model == 'lstm':
        X_train, y_train, _, _, _, _, _, _ = process_all_data(file1, 1, features)
        m = model.get_lstm_three_features([1, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config)
    if args.model == 'gru':
        X_train, y_train, _, _, _, _, _, _ = process_all_data(file1, 1, features)
        m = model.get_gru_three_features([1, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config)
    if args.model == 'lstm-lag':
        X_train, y_train, _, _, _, _, _, _ = process_all_data_with_lag(file1, lag, features)
        m = model.get_lstm_three_features([12, 8, 8, 1])
        train_model(m, X_train, y_train, args.model, config)

    if args.model == 'gru-lag':
        X_train, y_train, _, _, _, _, _, _ = process_all_data_with_lag(file1, lag, features)
        m = model.get_gru_three_features([12, 8, 8, 1])
        train_model(m, X_train, y_train, args.model, config)


if __name__ == '__main__':
    main(sys.argv)
