# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by:V.CHANTHRU 
RegisterNumber:24900997

code:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv(r"C:\Users\admin\Downloads\student_scores.csv")
print(df)
df.head()

df.tail()
print(df.head())
print(df.tail())

X=df.iloc[:,:-1].values
print(X)

Y=df.iloc[:,1].values
print(Y)

#spilitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
print(Y_pred)

print(Y_test)

#graph plot for training data
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)



## Output:
![Screenshot 2024-10-18 093523](https://github.com/user-attachments/assets/1e8a932e-5dcb-4c73-b9b1-64b9e829b188)
![Screenshot 2024-10-18 093604](https://github.com/user-attachments/assets/f6c658e5-85ea-405a-9fa6-f7adad6111a1)
![Screenshot 2024-10-18 093615](https://github.com/user-attachments/assets/7302827d-ec4e-432c-87a6-db511b25ec26)






## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
