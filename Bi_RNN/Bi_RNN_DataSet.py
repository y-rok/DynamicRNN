'''
DataSet class

used for reading training and testing data
'''
import numpy as np

class Bi_RNN_DataSet:
    def __init__(self, filePath):
        self.data = [] # 3d - batch size * time steps * the number of features
        self.dataLength = [] # 1d - batch size
        self.index=[] # 1d - the index of time steps used as output
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
            '''
            # set dataLength
            dataLength=int(float(item[0]))
            self.dataLength.append(float(item[0]))
            del item[0]

            timeSteps_label = []
            # set label
            for i in range(dataLength):
                if item[dataLength] == "1":
                    timeSteps_label.append(1.)
                else:
                    timeSteps_label.append(0.)
                del item[dataLength]
            self.label.append(timeSteps_label)

            # set data
            timeSteps_data = []
            while len(item) > 0:
                timeSteps_data.append([float(x) for x in item[0:self.numFeatures]])
                del item[0:self.numFeatures]

            # pad zero
            if len(timeSteps_data) < self.maxDataLength:
                for i in range(self.maxDataLength - len(timeSteps_data)):
                    timeSteps_data.append([0.] * self.numFeatures)
                    timeSteps_label.append(0.)

            self.data.append(timeSteps_data)
            '''

            dataLength = int(float(item[0]))
            del item[0]

            # set data
            timeSteps_data = []
            for i in range(dataLength):
                timeSteps_data.append([float(item[i])])
                # del item[0:self.numFeatures]

            # pad zero
            if len(timeSteps_data) < self.maxDataLength:
                for i in range(self.maxDataLength - len(timeSteps_data)):
                    timeSteps_data.append([0.] )

            index=0
            # set label
            for i in range(dataLength):
                # set dataLength
                self.dataLength.append(dataLength)
                if item[dataLength+i] == "1":
                    self.label.append([1.,0.])
                else:
                    self.label.append([0.,1.])
                self.index.append(index)
                index+=1
                self.data.append(timeSteps_data)
