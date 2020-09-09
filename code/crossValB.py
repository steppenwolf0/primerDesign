# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import os
import sys
import tarfile
import math
import random
import sys
from pandas import read_csv
import pandas as pd 

def oneHot (array, size):
	output=[]
	for i in range (len(array)):
		temp=np.zeros(size)
		temp[int(array[i])]=1
		output.append(temp)
	print(len(output))
	print(len(output[0]))
	return np.array(output)
	
def getBatch(data, labels, size, sampleSize):
	index=[]
	for i in range (len(data)):
		index.append(i)
	batch=random.sample(index,  size)
	outData=[]
	outLabels=[]
	for i in range (len(batch)):
		sample=np.zeros(sampleSize)
		for j in range (0, len(data[batch[i]])):
			if(data[batch[i]][j]=='C'):
				sample[j]=0.25
			elif(data[batch[i]][j]=='T'):
				sample[j]=0.50
			elif(data[batch[i]][j]=='G'):
				sample[j]=0.75
			elif(data[batch[i]][j]=='A'):
				sample[j]=1.0
			else:
				sample[j]=0.0
		outData.append(sample)
		outLabels.append(labels[batch[i]])
	return np.array(outData), np.array(outLabels)

def getBatch_run(data, labels, size,run,vector, sampleSize):
	infLimit=run*size
	supLimit=infLimit+size
	if supLimit > len(data):
		supLimit=len(data)
	batch=[]
	for i in range (infLimit,supLimit):
		batch.append(vector[i])
	outData=[]
	outLabels=[]
	for i in range (len(batch)):
		sample=np.zeros(sampleSize)
		for j in range (0, len(data[batch[i]])):
			if(data[batch[i]][j]=='C'):
				sample[j]=0.25
			elif(data[batch[i]][j]=='T'):
				sample[j]=0.50
			elif(data[batch[i]][j]=='G'):
				sample[j]=0.75
			elif(data[batch[i]][j]=='A'):
				sample[j]=1.0
			else:
				sample[j]=0.0
		outData.append(sample)
		outLabels.append(labels[batch[i]])
	return np.array(outData), np.array(outLabels)

def get_Info(indexVar,dataFolder):
	#examples=8129

	data = read_csv(dataFolder+'sequences.csv', header=None).values.ravel()
	print('data set', data.shape)

	
	labels=read_csv(dataFolder+'labels.csv', header=None).values.ravel()
	
	labelSize=int(np.max(labels)+1)
	
	max=0
	for i in range (0,len(data)):
		if (len(data[i])>max):
			max=len(data[i])
	
	vectorSize=max
	print('vectorSize', vectorSize)
	
	#print(labels)
	testIndex=read_csv(dataFolder+'index/'+str(indexVar)+'test_index.txt', header=None).values.ravel()
	valIndex=read_csv(dataFolder+'index/'+str(indexVar)+'val_index.txt', header=None).values.ravel()
	trainIndex=read_csv(dataFolder+'index/'+str(indexVar)+'train_index.txt', header=None).values.ravel()
	
	testIndex=testIndex.astype(int)
	valIndex=valIndex.astype(int)
	trainIndex=trainIndex.astype(int)
		
	train=[]
	test=[]
	valid=[]
	
	
	trainLabels=[]
	testLabels=[]
	validLabels=[]
	#test***************************************************************************
	for i in range (0,len(testIndex)):
		testLabels.append(labels[testIndex[i]])
		test.append(data[testIndex[i]])
	#valid***************************************************************************
	for i in range (0,len(valIndex)):
		validLabels.append(labels[valIndex[i]])
		valid.append(data[valIndex[i]])
	#train***************************************************************************
	for i in range (0,len(trainIndex)):
		trainLabels.append(labels[trainIndex[i]])
		train.append(data[trainIndex[i]])
	

	testLabels=np.array(testLabels)
	validLabels=np.array(validLabels)
	trainLabels=np.array(trainLabels)

	print(len(train))
	print(len(train[0]))
	print(trainLabels.shape)
	print(len(valid))
	print(validLabels.shape)
	print(len(test))
	print(testLabels.shape)
	
	print(labelSize)
	oneHot_train_labels=oneHot(trainLabels,labelSize)
	print(oneHot_train_labels.shape)

	oneHot_valid_labels=oneHot(validLabels,labelSize)
	print(oneHot_valid_labels.shape)

	oneHot_test_labels=oneHot(testLabels,labelSize)
	print(oneHot_test_labels.shape)


	return(test,oneHot_test_labels,valid,oneHot_valid_labels,train,oneHot_train_labels,labelSize,vectorSize)
def get_InfoTotal(dataFolder):
	#examples=8129

	data = read_csv(dataFolder+'sequences.csv', header=None).values.ravel()
	print('data set', data.shape)

	
	labels=read_csv(dataFolder+'labels.csv', header=None).values.ravel()
	
	labelSize=int(np.max(labels)+1)
	
	max=0
	for i in range (0,len(data)):
		if (len(data[i])>max):
			max=len(data[i])
	
	vectorSize=max
	print('vectorSize', vectorSize)
	
	
	print(labelSize)
	oneHot_labels=oneHot(labels,labelSize)
	print(oneHot_labels.shape)


	return(data,oneHot_labels)



if __name__ == "__main__" :
	sys.exit( get_Info(0,"./") )