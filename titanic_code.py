from __future__ import division, print_function, absolute_import
import csv
import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import seaborn as sns


newdata = pd.read_csv('train.csv')
print(newdata.head(5))


newdata.drop(['Name'],axis=1,inplace=True)
newdata.drop(['Cabin'],axis=1,inplace=True)
newdata.drop(['Ticket'],axis=1,inplace=True)



sns.factorplot('Pclass','Survived', data=newdata,size=4,aspect=3)
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
sns.countplot(x='Embarked', data=newdata, ax=axis1)

sns.countplot(x='Survived', hue="Embarked", data=newdata, order=[1,0], ax=axis2)

# group by embarked, and get the mean for survived passengers for each value in Embarked
embark_perc = newdata[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
print(embark_perc)
sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)


fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age values - Titanic')
axis2.set_title('New Age values - Titanic')

newdata["Age"].fillna(newdata["Age"].median(), inplace=True)

newdata['Age']=newdata['Age'].astype(int)
newdata['Age'].hist(bins=70, ax=axis2)

facet = sns.FacetGrid(newdata, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, newdata['Age'].max()))
facet.add_legend()

# average survived passengers by age
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
average_age = newdata[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)
plt.show()



newdata['Embarked'].replace(
    to_replace=['S'],
    value=0,
    inplace=True
)

newdata['Embarked'].replace(
    to_replace=['C'],
    value=1,
    inplace=True
)

newdata['Embarked'].replace(
    to_replace=['Q'],
    value=2,
    inplace=True
)

newdata['Sex'].replace(
    to_replace=['male'],
    value=0,
    inplace=True
)


newdata['Sex'].replace(
    to_replace=['female'],
    value=1,
    inplace=True
)

print(newdata.head(5))

data = newdata.sum(axis=0)
print(data)

newdata=newdata.fillna(newdata.mean())
survive=newdata['Survived']
print(survive.head(5))
newdata.drop(['Survived'],axis=1,inplace=True)

newdata.drop(['PassengerId'],axis=1,inplace=True)

print(newdata.head(5))


##for test data , repeat the same


testdata = pd.read_csv('test.csv')
print(testdata.head(5))


testdata.drop(['Name'],axis=1,inplace=True)
testdata.drop(['Cabin'],axis=1,inplace=True)
testdata.drop(['Ticket'],axis=1,inplace=True)
print(testdata.head(5))

testdata['Embarked'].replace(
    to_replace=['S'],
    value=0,
    inplace=True
)

testdata['Embarked'].replace(
    to_replace=['C'],
    value=1,
    inplace=True
)

testdata['Embarked'].replace(
    to_replace=['Q'],
    value=2,
    inplace=True
)

testdata['Sex'].replace(
    to_replace=['male'],
    value=0,
    inplace=True
)


testdata['Sex'].replace(
    to_replace=['female'],
    value=1,
    inplace=True
)

print(testdata.head(5))
testdata.drop(['PassengerId'],axis=1,inplace=True)

data = testdata.sum(axis=0)
print(data)

testdata=testdata.fillna(testdata.mean())


print(testdata.head(5))

#normalize the data
testdata1=(testdata-testdata.mean())/testdata.std()
newdata1=(newdata-newdata.mean())/newdata.std()


#using logistic regression


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets



X = newdata1  # we only take the first two features.
Y = survive


logreg = linear_model.LogisticRegression(C=1e5)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, Y)

accuracy=list()
ans=logreg.predict(testdata1)
print(ans)

#write the data to csv file

# data columns of csv file
fields = ['PassengerId','Survived']




# name of csv file
filename = "log_reg.csv"

# writing to csv file
start=892
with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(fields)

    # writing the data rows
    for i in range(len(ans)):
        csvwriter.writerow([start,ans[i]])
        start=start+1




#using svm

from sklearn import svm


y = Y
clf = svm.SVC()
clf.fit(X, y)

answer=clf.predict(testdata1)
print(answer)



# name of csv file
filename = "svm.csv"

# writing to csv file
start=892
with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(fields)

    # writing the data rows
    for i in range(len(ans)):
        csvwriter.writerow([start,answer[i]])
        start=start+1
