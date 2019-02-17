# -*- coding: utf-8 -*-
"""
@author: Hikmet Ugur Akgul

In this code we are not adding an extra line , we are not waiting for user to input some 
values. All values are coming from a CSV file which is read by pandas csv reader.
"""

import numpy as np
import pandas as pandas  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix   


names = ['X1' , 'X2', 'Class']
dataset = pandas.read_csv("Data-KNN.csv",names=names)


X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 2].values  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 





classifier = KNeighborsClassifier(n_neighbors=3) #K = 5  
classifier.fit(X_train, y_train)  

y_pred = classifier.predict(X_test)  


print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 

print(y_test) # Class test values
print(y_pred) # Class prediction values

error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    
    plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='blue', linestyle='dashed', marker='o',  
         markerfacecolor='red', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')  