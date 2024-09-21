import numpy as np
from numpy import concatenate

"""
    #class for reshape data for prediction
    
"""


class Value:

    def __init__(self, data):
        self.data = data

    # encode and reshape data, which the model can take in for prediction
    def reshapeData(self, scaler, location_encoder, date_encoder):
        test_values = self.data.values

        # encode the location
        test_values[:, 1] = location_encoder.transform(test_values[:, 1])

        # encode the date
        test_values[:, 2] = date_encoder.transform(test_values[:, 2])

        # scale the values
        test_values = test_values.astype('float32')
        test_values = scaler.transform(test_values)

        # test_reshaped = test_values[:, :-1]
        test_reshaped = test_values.reshape((test_values.shape[0], 1, test_values.shape[1]))

        self.data = test_reshaped

    # reverse the reshape
    def reverseReshape(self, shape):
        data_reshape = shape.reshape((shape.shape[0], shape.shape[2]))
        predicted = concatenate((self.data, data_reshape[:, 1:]), axis=1)

        self.data = predicted
