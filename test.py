# following tutorial found on YT, using scikit learn
import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("train.csv").as_matrix()
clf = DecisionTreeClassifier()

# training data
xtrain = data[0:21000, 1:]    # takes data from rows 0 to 21000, starting from column 1 (ignoring first column)
train_label = data[0:21000, 0] # takes only column 0, which is the training label

clf.fit(xtrain,train_label)

# testing data
xtest = data[21000: , 1:]
actual_label = data[21000:, 0]

# d = xtest[8]
# d.shape = (28, 28)
# pt.imshow(255 - d, cmap = 'gray')
# print(clf.predict([xtest[8]]))

# pt.show()

p = clf.predict(xtest)

count = 0
for i in range(0,21000):
    if p[i] == actual_label[i]:
        count +=1  

print("Accuracy = ", (count/21000) * 100)