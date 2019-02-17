# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 00:08:01 2018

@author: Hikmet Ugur Akgul

"""
from sklearn.neighbors import KNeighborsClassifier
import numpy as numpy 
import pandas as pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix   
import matplotlib.pyplot as plt

names = ['X1' , 'X2', 'Class']
dataset = pandas.read_csv("Data-KNN.csv",names=names)

# X for dataset and y for labels
X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 2].values  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01) 

print("Please enter X1 value : ")
x1 = input()
x1 = int(x1)
print("Please enter X2 value : ")
x2 = input()
x2 = int(x2)

X_test = numpy.array([[x1,x2]]) # Load user-given inputs to a test variable

classifier = KNeighborsClassifier(n_neighbors=5) #K = 5  
classifier.fit(X_train, y_train)  

y_pred = classifier.predict(X_test) #Make the class prediction


#print(confusion_matrix(y_test, y_pred))  
#print(classification_report(y_test, y_pred)) 
"""
If we print these results , then our prediction according to these results will fail
because it will select a random data from dataset. -- From train_test_split function.(test_size=0.01)
And because of this random selection , if it selects a different label from our prediction,
the classification report will show 0% precision. And if it selects the same label as we 
predict , it will 100% precision. There is no other value that precision or f-1 score can 
take according to this codes work.  
"""
print(y_pred)

"""
 In the error calculation section , we are trying various K values on the created model.
 And we find error values for each K value in that model. 
 We are trying to find which K value would be more effective on the created model and 
 plotting the result.
"""

# Calculating error for K values between 1 and 40
error = []
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(numpy.mean(pred_i != y_test))
    
    plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=7)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')  