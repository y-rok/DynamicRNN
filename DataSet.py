'''
DataSet class

used for reading training and testing data
'''
import numpy as np

class DataSet:
    def __init__(self, filePath):
        self.data = [] # 3d - batch size * time steps * the number of features
        self.dataLength = [] # 1d - batch size
        self.label = [] # 2d - batch size * the number of classes
        self.batchOffset = 0  # next index of data for learning with specific batch size
        # self.numData
        # self.maxDataLength
        # self.numFeatrues

        self.readData(filePath)

    def readData(self, filePath):
        f = open(filePath, "r")

        # get the number of input data and features
        line = f.readline()
        line = line.replace('\n', '')
        item = line.split(',', 2)

        self.numData = int(item[0])
        self.maxDataLength = int(item[1])
        self.numFeatures = int(item[2])

        # data = tf.placeholder(tf.float32, [None, maxDataLength, numFeatures])  # batch size * time steps * features

        while True:
            # read one data and get length, data and label
            line = f.readline()
            line = line.replace('\n', '')
            if not line: break
            item = line.split(',')

            # set dataLength
            self.dataLength.append(float(item[0]))
            del item[0]

            # set label
            if item[len(item) - 1] == "1":
                self.label.append([1.,0.])
            else:
                self.label.append([0.,1.])
            del item[len(item) - 1]

            # set data
            timeSteps = []
            while len(item)>0:
                timeSteps.append([float(x) for x in item[0:self.numFeatures]])
                del item[0:self.numFeatures]

            # pad zero
            if len(timeSteps)<self.maxDataLength:
                for i in range(self.maxDataLength-len(timeSteps)):
                    timeSteps.append([0.]*self.numFeatures)

            self.data.append(timeSteps)

        # self.data = np.array(self.data)
        # self.dataLength = np.array(self.dataLength)
        # self.label = np.array(self.label)

    # get next batch
    def nextBatch(self, batchSize):
        if self.batchOffset == len(self.data):
            self.batchOffset = 0

        batchLength = self.dataLength[self.batchOffset:min(self.batchOffset + batchSize, len(self.data))]
        batchData = self.data[self.batchOffset:min(self.batchOffset + batchSize, len(self.data))]
        batchLabel = self.label[self.batchOffset:min(self.batchOffset + batchSize, len(self.data))]

        self.batchOffset = min(self.batchOffset + batchSize, len(self.data))

        return batchLength, batchData, batchLabel

    # return True, if current batch is the last one
    def isLastBatch(self, batchSize):
        if self.batchOffset == len(self.data):
            return True
        else:
            return False