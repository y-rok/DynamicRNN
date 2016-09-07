'''
written by Claude Jang

# RNN Model classifying whether the number of 1s in input data is more than specific number or not

reference - # http://danijar.com/variable-sequence-lengths-in-tensorflow/
'''
import tensorflow as tf
from DataSet import *

class DynamicRNN:


    def __init__(self,trainingFilePath,testingFilePath):
        self.trainingData = DataSet(trainingFilePath)
        self.testingData = DataSet(testingFilePath)
        self.numClasses = 2


    # learning / store model / restore model
    def learning(self, epoch=1000, batchSize=None, numHidden=200, learningRate=0.003, storePath=None, restorePath=None):

        if batchSize ==None:
            batchSize=len(self.trainingData.data)


        sess = tf.Session()


        length = tf.placeholder(tf.int32,[None])
        data = tf.placeholder(tf.float32, [None, self.trainingData.maxDataLength, self.trainingData.numFeatures])  # batch size * time steps * features
        label = tf.placeholder(tf.float32, [None, self.numClasses])
        # numHidden = tf.Variable(numHidden,dtype=tf.int32)

        prediction = self.prediction(data, numHidden, length)
        cost = -tf.reduce_sum(label * tf.log(prediction))
        optimizer = tf.train.RMSPropOptimizer(learningRate).minimize(cost)
        accuracy = self.accuracy(prediction,label)

        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())

        if restorePath!=None:
            saver.restore(sess,restorePath+".ckpt")
            print("model restored from "+restorePath)


        for i in range(epoch):

            # print accuracy for testing data and loss
            if (i) % 10 == 1:
                testingDataAccuracy = sess.run(accuracy, {data: self.testingData.data, label: self.testingData.label,length: self.testingData.dataLength})
                loss = sess.run(cost, {data: self.trainingData.data, label: self.trainingData.label, length: self.trainingData.dataLength})
                print('Epoch {:2d} accuracy {:3.1f}%'.format(i, 100 * testingDataAccuracy) + ' loss {:.6f}'.format(loss))

            # save model
            if (i + 1) % 100 == 0 and storePath != None:
                savePath = saver.save(sess, storePath + ".ckpt")
                print("Model saved in file : " + savePath)

            self.trainingData.batchOffset=0

            while not self.trainingData.isLastBatch(batchSize):

                # print("epoch = "+str(i+1)+"batch offset = "+str(self.trainingData.batchOffset))
                batchLength, batchData, batchLabel = self.trainingData.nextBatch(batchSize)
                sess.run(optimizer,{data:batchData,label:batchLabel,length:batchLength})


                # sess.run(optimizer, {data:self.testingData.data,label:self.testingData.label,length:self.testingData.dataLength})
                # sess.run(optimizer, {data: [[[0.],[1.]],[[0.],[1.],[0.]]], label: [[0.,1.],[0.,1.]], length: [5.,5.]})
                # sess.run(optimizer, {data: [[[1.], [0.], [1.], [1.], [1.]], [[1.], [0.], [1.], [1.], [1.]]],
                #                      label: [[0., 1.], [0., 1.]], length: batchLength})


    # get last output from rnn model
    def prediction(self,data,numHidden,length):

        # output
        # last - return last hidden state and internal state of memory cell
        output, last = tf.nn.dynamic_rnn(

            tf.nn.rnn_cell.LSTMCell(numHidden,state_is_tuple=True),
            data,
            dtype=tf.float32,
            sequence_length=length,  # sequence length option
                                     # When running the model later, TensorFlow will return zero vectors for states and outputs after these sequence lengths.
                                     # Therefore, weights will not affect those outputs and don’t get trained on them.
        )

        last=last.h # set last hidden state
        # last = self.last_relevant(output, length)

        # Softmax layer.
        weight, bias = self.weight_and_bias(
            numHidden, self.numClasses)
        tf.matmul(last, weight)
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

        return prediction

    # get accuracy by comparing between output of rnn model and label
    def accuracy(self,prediction,label):
        mistakes = tf.equal(
            tf.argmax(label, 1), tf.argmax(prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    def weight_and_bias(self,in_size, out_size):
        weight =tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.01),dtype=tf.float32)
        bias = tf.Variable(tf.constant(0.1, shape=[out_size]),dtype=tf.float32)
        return weight, bias

    # restore model and predict label
    # def predict(self, restorePath,numHidden=200):
    #
    #     sess = tf.Session()
    #
    #     length = tf.placeholder(tf.int32, [None])
    #     data = tf.placeholder(tf.float32, [None, self.trainingData.maxDataLength,self.trainingData.numFeatures])  # batch size * time steps * features
    #     label = tf.placeholder(tf.float32, [None, self.numClasses])
    #
    #
    #     prediction = self.prediction(data, numHidden, length)
    #
    #     saver = tf.train.Saver()
    #     saver.restore(sess, restorePath + ".ckpt")
    #
    #     result=sess.run(prediction,{data: [[[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.]]], length: [10.]})
    #     print (result)