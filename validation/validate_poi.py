#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle as cPickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

#data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

data_dict_path = "../final_project/final_project_dataset.pkl"

with open(data_dict_path, 'rb') as file:
	    data_dict = cPickle.load(file)

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X = features
Y = labels

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
						    random_state=42)

clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf = clf.fit(X_train, Y_train)

pred = clf.predict(X_test)
acc = accuracy_score(pred, Y_test)

print("Accuracy is ", acc)

