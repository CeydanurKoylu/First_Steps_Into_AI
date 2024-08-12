import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer


trainData = pd.read_csv("./titanic/train.csv") 

testData = pd.read_csv("./titanic/test.csv")
# "PassengerId","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"  
features = ["Pclass","Sex","Age","SibSp","Parch","Ticket","Fare","Embarked"]


print(trainData.columns)


print(trainData.shape)

imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
trainData["Embarked"] = imp.fit_transform(trainData[["Embarked"]]).ravel()
print(trainData.isnull().sum())
print(trainData.shape)

print(trainData.Embarked)
"""

imp = SimpleImputer(missing_values=np.nan, strategy="mean")
testData["Fare"] = imp.fit_transform(testData[["Fare"]]).ravel()
print(testData.isnull().sum())"""

"""
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp.fit(trainData["Embarked"])
trainData["Embarked"] = imp.transform(trainData["Embarked"])
print(trainData.shape)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(trainData["Fare"])
trainData["Fare"] = imp.transform(trainData["Fare"])"""

X = pd.get_dummies(trainData[features])
X_test = pd.get_dummies(testData[features])


X_test = X_test.reindex(columns=X.columns, fill_value=0)

imp = SimpleImputer(missing_values=np.nan, strategy="mean")
X_test["Fare"] = imp.fit_transform(X_test[["Fare"]]).ravel()

imp = SimpleImputer(missing_values=np.nan, strategy="mean")
X_test["Age"] = imp.fit_transform(X_test[["Age"]]).ravel()

imp = SimpleImputer(missing_values=np.nan, strategy="mean")
X["Age"] = imp.fit_transform(X[["Age"]]).ravel()



print(X_test.shape, X.shape)

y = trainData["Survived"]

model = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', max_iter = 500)

model.fit(X,y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': testData.PassengerId, 'Survived': predictions})

output.to_csv('submission_feature_engineering.csv', index=False)

# With simple imputation over most frequent values in all columns:
# For "Pclass","Sex","Age","SibSp","Parch","Embarked" set with L1 I got 0.772
# When I added "Ticket","Fare","Cabin" and filled the missing feature columns of test_data with 0's, I got 0.76
# Switching to L2 gave 0.770
# I will try more feature enginnering after checking the correlations between the data and the survival rates
