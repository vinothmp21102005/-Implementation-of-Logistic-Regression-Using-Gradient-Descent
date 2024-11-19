# Implementation-of-Logistic-Regression-Using-Gradient-Descent
## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1. Set weights, bias, learning rate, and iterations.
2. Load data, split into X and y, normalize features.
3.  Apply sigmoid: ℎ(𝑥)
4. Use gradient descent to update weights and bias iteratively.
5.  Classify based on sigmoid output (>0.5→1,≤0.5→0).
6.  Measure performance with accuracy or other metrics.
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: VINOTH M P
RegisterNumber:  212223240182
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("Placement_Data.csv")
data=data.drop(["sl_no","salary"],axis=1)

data["gender"]=(data["gender"]).astype('category')
data["ssc_b"]=(data["ssc_b"]).astype('category')
data["hsc_b"]=(data["hsc_b"]).astype('category')
data["hsc_s"]=(data["hsc_s"]).astype('category')
data["degree_t"]=(data["degree_t"]).astype('category')
data["workex"]=(data["workex"]).astype('category')
data["specialisation"]=(data["specialisation"]).astype('category')
data["status"]=(data["status"]).astype('category')
data.dtypes

data["gender"]=(data["gender"]).cat.codes
data["ssc_b"]=(data["ssc_b"]).cat.codes
data["hsc_b"]=(data["hsc_b"]).cat.codes
data["hsc_s"]=(data["hsc_s"]).cat.codes
data["degree_t"]=(data["degree_t"]).cat.codes
data["workex"]=(data["workex"]).cat.codes
data["specialisation"]=(data["specialisation"]).cat.codes
data["status"]=(data["status"]).cat.codes

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
theta = np.random.randn(x.shape[1])
Y=y
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def loss(theta,x,Y):
    h= sigmoid(x.dot(theta))
    return -np.sum(Y * np.log(h) + (1-Y) * np.log(1-h))
def gradient_descent(theta,x,Y,alpha,num_iterations):
    m=len(Y)
    for i in range(num_iterations):
        h = sigmoid(x.dot(theta))
        gradient = x.T.dot(h-Y)/m
        theta -= alpha * gradient
    return theta
theta = gradient_descent(theta,x,Y,alpha=0.01,num_iterations=1000)
def predict(theta,x):
    h = sigmoid(x.dot(theta))
    y_pred = np.where(h >= 0.5 ,1 ,0)
    return y_pred

y_pred = predict(theta,x)
accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print("prediction\n",y_pred)
xnew = np.array(([0,87,0,95,0,0,1,0,2,3,2,3]))
yprednew=predict(theta,xnew)
print("New prediction :",yprednew)
```
## OUTPUT:
Data Type:

![image](https://github.com/user-attachments/assets/9c81e85e-0fd5-42a0-843c-7f5324e3a935)

![image](https://github.com/user-attachments/assets/1176a866-af39-4afe-a159-9247562fa52a)

![image](https://github.com/user-attachments/assets/255c5e34-971e-4938-bd2b-c10f85a1f04d)

![image](https://github.com/user-attachments/assets/bf0c4268-d042-47b8-b15c-2b9b21e4ce4e)
## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

