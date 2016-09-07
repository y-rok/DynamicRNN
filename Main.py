from InputGenerator import *
from DynamicRNN import *

# ====================
#  Example
# ====================

# generate input data and write it to file
# data - binary data which of length is from 3~20
# label - if the number of 1s is more than 7, 1 / else, 0
trainingFilePath = "./input/trainingData.csv"
testingFilePath = "./input/testingData.csv"

inputLength,inputData,label=genrateInputData(1000)
writeInputData(inputLength,inputData,label,trainingFilePath)
inputLength,inputData,label=genrateInputData(100)
writeInputData(inputLength,inputData,label,testingFilePath)

# training data and store it
dynamicRnn = DynamicRNN(trainingFilePath,testingFilePath)
dynamicRnn.learning(batchSize=100,storePath="./model/temp")

