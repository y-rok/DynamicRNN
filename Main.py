from RNN.RNN_tfLearn import *
from Bi_RNN.Bi_RNN_tfLearn import *
import RNN.RNN_InputGenerator as RNN_Input


# ====================
#  File Path
#  ====================

# generate input data and write it to file
# data - binary data which of length is from 3~20
# label - if the number of 1s is more than 7, 1 / else, 0
# RNN_trainingFilePath = "./input/RNN_trainingData.csv"
# RNN_testingFilePath = "./input/RNN_testingData.csv"


# generate input data and write it to file
# data - binary data which of length is from 3~20
# label - if a number is larger than the average of all numbers, 1 / else, 0
Bi_RNN_trainingFilePath = "./input/Bi_RNN_trainingData.csv"
Bi_RNN_testingFilePath = "./input/Bi_RNN_testingData.csv"

# ====================
#  Generate Input
#  ====================
#
# inputLength,inputData,label=RNN_Input.genrateInputData(10000)
# RNN_Input.writeInputData(inputLength, inputData, label, RNN_trainingFilePath)
# inputLength,inputData,label=RNN_Input.genrateInputData(10000)
# RNN_Input.writeInputData(inputLength, inputData, label, RNN_testingFilePath)


# generate input data and write it to file
# data - binary data which of length is from 3~20
# label - if a number is larger than the average of all numbers, 1 / else, 0
# Bi_RNN_trainingFilePath = "./input/Bi_RNN_trainingData.csv"
# Bi_RNN_testingFilePath = "./input/Bi_RNN_testingData.csv"

# inputLength,inputData,label=Bi_RNN_Input.genrateInputData(10000)
# Bi_RNN_Input.writeInputData(inputLength, inputData, label, Bi_RNN_trainingFilePath)
# inputLength,inputData,label=Bi_RNN_Input.genrateInputData(10000)
# Bi_RNN_Input.writeInputData(inputLength, inputData, label, Bi_RNN_testingFilePath)

# ====================
#  Example for Dynamic RNN
# ====================


# training data and store it
# dynamicRnn = DynamicRNN(RNN_trainingFilePath,RNN_testingFilePath)
# dynamicRnn.learning(epoch=11,batchSize=100,storePath="./model/temp")

# ====================
#  Example for RNN tfLearn
# ====================
#
# rnn_tfLearn = RNN_tfLearn(RNN_trainingFilePath, RNN_testingFilePath)
# rnn_tfLearn.learning()

# ====================
#  Example for Bidirectional RNN tfLearn
# ====================

rnn_tfLearn = Bi_RNN(Bi_RNN_trainingFilePath, Bi_RNN_testingFilePath)
rnn_tfLearn.learning()