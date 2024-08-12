import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer



trainData = pd.read_csv("./titanic/train.csv") 

testData = pd.read_csv("./titanic/test.csv")

features = ["Pclass","Sex","Age","SibSp","Parch","Embarked"]

X = pd.get_dummies(trainData[features])
X_test = pd.get_dummies(testData[features])
print(X_test, X)

imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp.fit(X)
X = imp.transform(X)
print(X.shape)

imp2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp2.fit(X_test)
X_test = imp2.transform(X_test)

print(X_test.shape, X.shape)

y = trainData["Survived"]

model = LogisticRegression(C=0.1, penalty='l1', solver='liblinear')

model.fit(X,y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': testData.PassengerId, 'Survived': predictions})

output.to_csv('submission.csv', index=False)