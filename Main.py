from InputGenerator import *
from DynamicRNN import *
from Bi_RNN_tfLearn import *
from RNN_tfLearn import *

# ====================
#  Generate Input
#  ====================

# generate input data and write it to file
# data - binary data which of length is from 3~20
# label - if the number of 1s is more than 7, 1 / else, 0
# trainingFilePath = "./input/trainingData.csv"
# testingFilePath = "./input/testingData.csv"
#
# inputLength,inputData,label=genrateInputData(10000)
# writeInputData(inputLength,inputData,label,trainingFilePath)
# inputLength,inputData,label=genrateInputData(10000)
# writeInputData(inputLength,inputData,label,testingFilePath)

# ====================
#  Example for Dynamic RNN
# ====================

# # training data and store it
# dynamicRnn = DynamicRNN(trainingFilePath,testingFilePath)
# dynamicRnn.learning(batchSize=100,storePath="./model/temp")

# ====================
#  Example for Bidirectional RNN tfLearn
# ====================
trainingFilePath = "./input/trainingData.csv"
testingFilePath = "./input/testingData.csv"

rnn_tfLearn = RNN(trainingFilePath,testingFilePath)
rnn_tfLearn.learning()