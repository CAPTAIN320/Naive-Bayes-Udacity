#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(kernel='rbf', C=10000)
t0 = time()
#features_train = features_train[:int(len(features_train)/100)] #take 1% of train data
#labels_train = labels_train[:int(len(labels_train)/100)] #take 1% of train labels
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")
t0 = time()
pred = clf.predict(features_test)
print("prediction time:", round(time()-t0, 3), "s")
accuracy = accuracy_score(pred, labels_test)
print("Accuracy is ", accuracy)
print("Prediction for element 10 is ", pred[10])
print("Prediction for element 26 is ", pred[26])
print("Prediction for element 50 is ", pred[50])
no_of_0 = 0
no_of_1 = 0
for i in pred:
	if i == 0:
		no_of_0 += 1
	else:
		no_of_1 += 1

print("Number of elements in class 0 = ", no_of_0)
print("Number of elements in class 1 = ", no_of_1)
#########################################################


