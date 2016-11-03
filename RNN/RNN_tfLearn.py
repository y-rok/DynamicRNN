from __future__ import division, print_function, absolute_import

import tflearn
from RNN.RNN_InputGenerator import *
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from RNN.RNN_DataSet import *


class RNN_tfLearn:


    def __init__(self,trainingFilePath,testingFilePath):
        self.trainingData = RNN_DataSet(trainingFilePath)
        self.testingData = RNN_DataSet(testingFilePath)
        self.numClasses = 2

    def learning(self,numHidden=100):

        # Network building
        net = input_data(shape=[None, MAXIMUM_LENGTH_DATA, NUM_FEATURES])
        # net = embedding(net, input_dim=20000, output_dim=128)
        net = tflearn.lstm(net, numHidden, dropout=0.8,return_state=True)
        net = dropout(net, 0.5)
        net = fully_connected(net, 2, activation='softmax')
        net = regression(net, optimizer='adam', loss='categorical_crossentropy')


        # Training
        model = tflearn.DNN(net,tensorboard_dir="./log/", tensorboard_verbose=0)
        model.fit(self.trainingData.data, self.trainingData.label, snapshot_epoch=True,validation_set=0.1, show_metric=False, batch_size=100,n_epoch=2,run_id="fdd009")