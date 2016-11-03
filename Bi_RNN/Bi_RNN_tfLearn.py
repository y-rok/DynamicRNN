from __future__ import division, print_function, absolute_import

import tflearn
from Bi_RNN.Bi_RNN_InputGenerator import *
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.data_utils import to_categorical, pad_sequences
from Bi_RNN.Bi_RNN_DataSet import *
import tensorflow as tf

class Bi_RNN:


    def __init__(self,trainingFilePath,testingFilePath):
        self.trainingData = Bi_RNN_DataSet(trainingFilePath)
        self.testingData = Bi_RNN_DataSet(testingFilePath)
        self.numClasses = 2


    def learning(self):

        # Network building
        net = input_data(shape=[None, MAXIMUM_LENGTH_DATA, 1])
        index = tf.placeholder(shape=[None], dtype=tf.int32)

        # net = bidirectional_rnn(net, BasicLSTMCell(200), BasicLSTMCell(200),return_seq=True)
        # net = dropout(net, 0.5)
        # net = tflearn.time_distributed(net, tflearn.fully_connected, [1,'softmax'])
        # net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='binary_crossentropy')
        # model = tflearn.DNN(net, tensorboard_verbose=3)
        # model.fit(self.trainingData.data, self.trainingData.label, validation_set=0.1, show_metric=True, batch_size=1000,n_epoch=100)

        net = bidirectional_rnn(net, BasicLSTMCell(200), BasicLSTMCell(200), return_seq=True,dynamic=True)
        net = fully_connected(net[index], 2, activation='softmax')
        net = regression(net, optimizer='adam', loss='categorical_crossentropy')

        # # Training
        model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=2)
        model.fit(self.trainingData.data, self.trainingData.label, validation_set=0.1, show_metric=True, batch_size=64)