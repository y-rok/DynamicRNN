'''
written by Claude Jang

# Input Data
#   - [first line]the number of data , the maximum length of data , the number of featrues
#   - [following line]the length of one data, data(the items comprising data), label
#       - data is comprised of 1 or 0 and has variable length
#           ex) [ 1 , 0 , 1 , 0 , 0 .... ]
#       - each item in the data will be fed to each time step of RNN
#
# RNN Model
#   classify the number of 1s in input data is more than MAXIMUM_NUMBER_OF_ONES

'''

import random

NUM_DATA = 100
MAIMUM_NUMBER_OF_ONES = 7 # The label will be 1, if the number of 1s is more than MAIMUM_NUMBER_OF_ONES
MAXIMUM_LENGTH_DATA = 20
MINIMUM_LENGTH_DATA = 3
NUM_FEATURES = 1


# generate input data
# parameter
#   numData - the number of input data
#   minLength - the minimum length of input data
#   maxLength - the maximum length of input data
#   maxNumOnes - the maximum number of ones , if data has 1s more than this variable, label will be 1
# return
#   the length of data list, data list, label list
def genrateInputData(numData = NUM_DATA, numFeatures=NUM_FEATURES, minLength=MINIMUM_LENGTH_DATA, maxLength=MAXIMUM_LENGTH_DATA, maxNumOnes=MAIMUM_NUMBER_OF_ONES):

    data=[] # input data list
    dataLength=[] # the length of input data list
    label=[] # the label of input data list

    for i in range(numData):
        # Random sequence length
        length = random.randint(minLength, maxLength)
        if length>length+numFeatures-length%numFeatures:
            length+=numFeatures-length%numFeatures
        else:
            length-=length%numFeatures
        dataLength.append(length)

        inputDataItem=[]
        numOnes=0 # the number of ones
        for j in range(length):
            binaryNum=random.randint(0,1)
            if binaryNum==1:
                numOnes+=1

            inputDataItem.append(binaryNum)

        data.append(inputDataItem)
        if numOnes>maxNumOnes:
            label.append(1)
        else:
            label.append(0)


    return dataLength, data, label


# write generated input data to text file
def writeInputData(numData, data, label, filePath, maxDataLength=MAXIMUM_LENGTH_DATA, numFeatures=NUM_FEATURES):

    f = open(filePath,"w")

    # first line => the number of data , the maximum length of data , the number of featrues
    f.write(str(len(data))+","+str(maxDataLength)+","+str(numFeatures)+"\n")

    for i in range(len(data)):
        out = str(numData[i])
        for temp in data[i]:
            out+= ","+str(temp)

        out+=","+str(label[i])+'\n'

        f.write(out)

    f.close()





