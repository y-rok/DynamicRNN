'''
written by Claude Jang

# Input Data
#   - [first line]the number of data , the maximum length of data , the number of featrues
#   - [following line]the length of one data, data(the items comprising data), label
#       - data is comprised of the numbers between 1 and 9 and has variable length
#           ex) [ 1 , 2 , 1 , 4 , 6 .... ]
#       - each item in the data will be fed to each time step of RNN
#
# RNN Model
#   classify whether each number is larger than tha average of all numbers

'''

import random

NUM_DATA = 100
MAXIMUM_LENGTH_DATA = 20
MINIMUM_LENGTH_DATA = 3



# generate input data
# parameter
#   numData - the number of input data
#   minLength - the minimum length of input data
#   maxLength - the maximum length of input data
# return
#   the length of data list, data list, label list
def genrateInputData(numData = NUM_DATA, minLength=MINIMUM_LENGTH_DATA, maxLength=MAXIMUM_LENGTH_DATA):

    data=[] # input data list
    dataLength=[] # the length of input data list
    label=[] # the label of input data list

    for i in range(numData):
        # Random sequence length
        length = random.randint(minLength, maxLength)
        dataLength.append(length)

        inputDataItem=[]
        sum=0
        for j in range(length):
            number=random.randint(1,9)
            sum+=number
            inputDataItem.append(number)

        avg = sum/length
        data.append(inputDataItem)

        labelItem=[]
        for numData in inputDataItem:
            if numData>=avg:
                labelItem.append(1)
            else :
                labelItem.append(0)
        label.append(labelItem)


    return dataLength, data, label


# write generated input data to text file
def writeInputData(numData, data, label, filePath, maxDataLength=MAXIMUM_LENGTH_DATA):

    f = open(filePath,"w")

    # first line => the number of data , the maximum length of data , the number of featrues
    f.write(str(len(data))+","+str(maxDataLength)+",1\n")

    for i in range(len(data)):
        out = str(numData[i])
        for temp in data[i]:
            out+= ","+str(temp)
        for temp in label[i]:
            out+= ","+str(temp)
        out+='\n'

        f.write(out)

    f.close()





