# ******************************** IMPORT MODULES ***************************

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn import preprocessing

# ************************************* READING CSV FILE **************************

url = r"C:\Users\Ayush jain\Desktop\car_evaluation.csv"
car = pd.read_csv(url)
print("The First Five Datas are : ",car.head())
print("Total (Rows,Columns) in the Dataset : ",car.shape)


# ************************************* FEATURE ENGINEERING.************************* 

#1  CONVERT STRING TO NUMBER
# a) Using Map function
car['Lug_Boot'] = car['Lug_Boot'].map({'small':0,'med':1,'big':2})
car['Safety'] = car['Safety'].map({'low':0,'med':1,'high':2})
car['Maintenance Cost'] = car['Maintenance Cost'].map({'low':0,'med':1,'high':2,'vhigh':3})
car['Buying Price'] = car['Buying Price'].map({'low':0,'med':1,'high':2,'vhigh':3})

# b) Using Label-Encoder
label_encoder = preprocessing.LabelEncoder()
car['Number of Doors'] = label_encoder.fit_transform(car['Number of Doors'])        # 2=0, 3=1, 4=2, 5more=3
car['Number of Persons'] = label_encoder.fit_transform(car['Number of Persons'])    # 2=0, 4=1, more=2
car['Decision'] = label_encoder.fit_transform(car['Decision'])

print("After Encoding, the data will look like as follows :", car.head())

#2  CHECKING FOR MISSING VALUES 

print(car.isnull().sum())
print("No Missing Values are Found in any Column")


# ************************************* ANALYZING THE DATA ************************

# For Detail Information like Quartile Range, Mean etc...
car.describe()
car.info()

# Types of Graph Plotting
#1
print("The Histogram for Each Column are :")
for column in car:
    plt.hist(car[column], color="red")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.title(f"Histogram for {column}")
    plt.show()

#2
print("The BoxPlot for Each Column are :")
fig, axs = plt.subplots(ncols=7,nrows=1,figsize=(20,10))
index=0
axs = axs.flatten() 
for k,v in car.items():
    sb.boxplot(y=v, data=car, ax=axs[index]) 
    index = index + 1
plt.tight_layout(pad=0.4, w_pad=0.1, h_pad=5.0)
plt.show()

#3
print("The DistPlot for Each Column are :")
fig, axs = plt.subplots(ncols=7,nrows=1,figsize=(20,10))
index=0
axs = axs.flatten()   
for k,v in car.items():
    sb.distplot(v,ax=axs[index])  
    index = index + 1
plt.tight_layout(pad=0.4, w_pad=0.1, h_pad=5.0)
plt.show()

#4
print("The RegPlot for Each Column are :")
for column in car:
    sb.regplot(x=column, y='Decision', data=car)
    plt.show()

#5
print("The CountPlot for Each Column are :")
for column in car:
    sb.countplot(car[column])
    plt.show()

#6
print("The BarPlot for Each Column are :")
for column in car:
    sb.barplot(car[column])
    plt.show()

# Value Count for Each Column
car['Buying Price'].value_counts()
car['Maintenance Cost'].value_counts()
car['Number of Doors'].value_counts()
car['Number of Persons'].value_counts()
car['Lug_Boot'].value_counts()
car['Safety'].value_counts()
car['Decision'].value_counts()

# Relationship of one column with target column
car[['Buying Price', 'Decision']].groupby(['Buying Price'], as_index=False).mean()
car[['Maintenance Cost', 'Decision']].groupby(['Maintenance Cost'], as_index=False).mean()
car[['Number of Doors', 'Decision']].groupby(['Number of Doors'], as_index=False).mean()
car[['Number of Persons', 'Decision']].groupby(['Number of Persons'], as_index=False).mean()
car[['Lug_Boot', 'Decision']].groupby(['Lug_Boot'], as_index=False).mean()
car[['Safety', 'Decision']].groupby(['Safety'], as_index=False).mean()

# Correlation Matrix
sb.heatmap(car.corr().abs(), annot=True)

# ************************************* ALGORITHMS *************************

X = car.iloc[:, 0:6]
Y = car.iloc[:, 6]

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

#1 KNeighborsClassifier Model
kn = KNeighborsClassifier(n_neighbors=7)
kn.fit(X_train,y_train)
y_pred = kn.predict(X_test)


print("The Accuracy is : ",accuracy_score(y_pred,y_test))
print("The Trained Score is : ",kn.score(X_train,y_train))

print("The Confusion Matrix is : ",confusion_matrix(y_pred, y_test))
print("The Classification Report is :",classification_report(y_pred, y_test))


#2 RandomForestClassifier Model
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)

print("The Accuracy is : ",accuracy_score(y_pred,y_test))
print("The Trained Score is : ",rf.score(X_train,y_train))

print("The Confusion Matrix is : ",confusion_matrix(y_pred, y_test))
print("The Classification Report is :",classification_report(y_pred, y_test))