# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step1. Start the Program.

Step 2. Import the necessary packages.

Step 3. Read the given csv file and display the few contents of the data.

Step 4. Assign the features for x and y respectively.

Step 5. Split the x and y sets into train and test sets.

Step 6. Convert the Alphabetical data to numeric using CountVectorizer.

Step 7. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

Step 8. Find the accuracy of the model.

Step 9. Close the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Joel John Jobinse
RegisterNumber:  212223240062
*/

#import packages
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("spam.csv",encoding="latin-1")
df.head()

#checking the data information and null presence
df.info()
df.isnull().sum()

#assigning x and y array
x=df["v1"].values
y=df["v2"].values

#splitting the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#converting to numerical count in train and test set
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

#predicting y- i.e detecting spam
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

#checking the accuracy of the model
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### data.head():
![image](https://github.com/joeljohnjobinse/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138955488/54aa85a1-cf9a-4557-b0cf-65fecb224737)

### data.info():
![image](https://github.com/joeljohnjobinse/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138955488/fce16b1b-858d-4eb7-8815-e4e8e916e58e)

### data.isnull().sum():
![image](https://github.com/joeljohnjobinse/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138955488/d335a210-3774-405c-9aed-94a66d728a19)

### Detected spam:
![image](https://github.com/joeljohnjobinse/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138955488/295a9b6c-9bfc-4a76-8a9a-28be1a1db0bd)

### Accuracy score of the model:
![image](https://github.com/joeljohnjobinse/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138955488/992f9557-7f2e-4c2d-a73f-1867dbad1b11)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
