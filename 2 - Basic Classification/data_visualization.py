import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



trainData = pd.read_csv("./titanic/train.csv") 

testData = pd.read_csv("./titanic/test.csv")

features = ["Pclass","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]
features2 = ["Survived","Ticket","Fare","Cabin","Embarked"]
features3 = ["Ticket","Fare","Cabin","Embarked"]
dfFeatures = ["PassengerId","Sex","Pclass","Survived","Age","SibSp","Parch","Fare","Embarked"]

df = trainData[dfFeatures]
print(trainData["Ticket"])
print(df)


X = pd.get_dummies(trainData[features])
X_test = pd.get_dummies(testData[features])

l = trainData.isnull().sum()
print(l, trainData.shape)

all_data = pd.concat([trainData, testData], ignore_index=True)
all_data.info()
all_data.value_counts() 


"""
fig, axes = plt.subplots(1,2, figsize=(20,5))
sns.countplot(ax=axes[0], data=df,x='Sex')
sns.countplot(ax=axes[1],data=df,x='Sex',hue='Survived')
plt.show()

sns.set_palette('Paired')
fig, axes = plt.subplots(1,2, figsize=(20,5))
sns.countplot(ax=axes[0], data=df,x='Embarked',hue='Survived')
sns.barplot(ax=axes[1],data=df,x='Embarked',y='Survived')
plt.show()


fig, axes = plt.subplots(1,2, figsize=(20,5))
sns.countplot(ax=axes[0], data=df,x='Pclass')
sns.barplot(ax=axes[1], data=df,x='Sex',y='Survived', hue="Pclass")
plt.show()




sns.histplot(data=df,x='Age')
plt.show()

Age_group=[]
for c in df.Age:
    if c<11:
        Age_group.append("0-10")
    elif 10<c<21:
        Age_group.append("11-20")
    elif 20<c<31:
        Age_group.append("21-30")
    elif 30<c<41:
        Age_group.append("31-40")
    elif 40<c<51:
        Age_group.append("41-50")
    elif 50<c<61:
        Age_group.append("51-60")
    elif 60<c<71:
        Age_group.append("61-70")
    else:
        Age_group.append("71-80")      

#Created an age group tag for all of the passengers, and then sorted tem in ascending order wrt ages

df['age_group']=Age_group
fig, axes = plt.subplots(1,2, figsize=(20,5))
sns.set_palette('Paired')
df1=df.sort_values('Age', ascending=True)
sns.countplot(ax=axes[0], data=df1,x='age_group', hue="Survived")
sns.barplot(ax=axes[1], x='age_group', hue='Sex', data=df1, y='Survived')
plt.show()

#In the 0-10 age group being a male or a female didn't make much of a difference


fig, axes = plt.subplots(1,2, figsize=(20,5))
sns.countplot(ax=axes[0], data=df,x='SibSp')
sns.countplot(data=df, x='SibSp', hue='Survived')
plt.show()



sns.set_palette('Paired')
fig, axes = plt.subplots(1,2, figsize=(20,5))
sns.countplot(ax=axes[0], data=df,x='Parch')
sns.barplot(data=df, x='Parch', y='Survived')
plt.show()






fare_group=[]
for c in df.Fare:
    if c<11:
        fare_group.append("0-10")
    elif 10<c<21:
        fare_group.append("11-20")
    elif 20<c<31:
        fare_group.append("21-30")
    elif 30<c<41:
        fare_group.append("31-40")
    elif 40<c<51:
        fare_group.append("41-50")
    elif 50<c<101:
        fare_group.append("50-100")
    elif 100<c<201:
        fare_group.append("101-200")
    elif 200<c<301:
        fare_group.append("201-300")
    elif 300<c<401:
        fare_group.append("301-400")
    elif 400<c<501:
        fare_group.append("401-500")
    else:
        fare_group.append("501-550")      
 
df['Fare_group']=fare_group
df['Fare_group'].value_counts()

fig, axes = plt.subplots(1,2, figsize=(20,5))
sns.set_palette('Paired')
df1=df.sort_values('Fare', ascending=True)
sns.countplot(ax=axes[0], data=df1,x='Fare_group', hue="Survived")
sns.barplot(ax=axes[1], x='Fare_group', hue='Sex', data=df1, y='Survived')
plt.show()
"""

# Females had the highest survival rate
# C embarkers were most likely to survive
# PClass 1 has the highest survival rate with almost all females surviving. 
# PClass 3 had the most passenger count although most of them didnt' survive.
# Surprisingly there were a lot of people in the 71-80 age group and almost half of them surived.
# People with 0 sibling/spouse were the most likely to survive
# People with 2 or 3 parents/children had the highest survival rate


