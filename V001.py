import pandas as pd
import numpy as np
import seaborn as sns
import sklearn.metrics as mt
from tqdm.auto import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix as cm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import statistics

df = pd.read_excel("C:/Users/Zehra/Desktop/Python/HR_Employee.xlsx")
df.head(5)

sns.countplot(x='Attrition', data=df)

Label Encoder
for columns in tqdm(df.columns):
    if dict(df.dtypes)[columns] == 'object':        
        label_encoder = preprocessing.LabelEncoder()
        df[columns] = label_encoder.fit_transform(df[columns])
        
        
Train Test Split
X = df.drop("Attrition", axis=1)
y = df["Attrition"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

Random Forest Classifier
R_F = RandomForestClassifier()
R_F.fit(X_train, y_train)
y_pred=R_F.predict(X_test)
print("Confusion Matrix")
print (confusion_matrix(y_test, y_pred))
print("Accuracy Score")
print(mt.accuracy_score(y_test, y_pred))


Cross Validation
kfold = KFold(n_splits=12, random_state=0, shuffle=True)
cvs = cross_val_score(R_F, X, y, cv=kfold)
statistics.mean(cvs)


Gradient Boosting Classifier
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)
y_pred1 = gb_model.predict(X_test)
print("Confusion Matrix")
print (confusion_matrix(y_test, y_pred1))
print("Accuracy Score")
print(mt.accuracy_score(y_test, y_pred1))

Cross Validation
cvs1= cross_val_score(gb_model, X, y, cv=kfold)
statistics.mean(cvs1)
