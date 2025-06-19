import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load dataset
data = pd.read_csv('Dataset/f2.csv')

# Rename columns for consistency
data.rename(columns={'Humidity ':'Humidity','Soil Type':'Soil_Type','Crop Type':'Crop_Type','Fertilizer Name':'Fertilizer'}, inplace=True)

# Encode categorical variables
encode_soil = LabelEncoder()
data.Soil_Type = encode_soil.fit_transform(data.Soil_Type)

encode_crop = LabelEncoder()
data.Crop_Type = encode_crop.fit_transform(data.Crop_Type)

encode_ferti = LabelEncoder()
data.Fertilizer = encode_ferti.fit_transform(data.Fertilizer)

# Split data into features and target
X = data.drop('Fertilizer', axis=1)
y = data.Fertilizer

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Initialize lists to store results
acc = []
model_names = []
acc_train = []

# Decision Tree
ds = DecisionTreeClassifier(criterion="entropy", random_state=2, max_depth=5)
ds.fit(x_train, y_train)
y_pred_test = ds.predict(x_test)
y_pred_train = ds.predict(x_train)
acc.append(accuracy_score(y_test, y_pred_test))
acc_train.append(accuracy_score(y_train, y_pred_train))
model_names.append('Decision Tree')
print("Decision Tree Accuracy: Test = {:.2f}%, Train = {:.2f}%".format(acc[-1]*100, acc_train[-1]*100))
print(classification_report(y_test, y_pred_test))

# Naive Bayes
nb = GaussianNB()
nb.fit(x_train, y_train)
y_pred_test = nb.predict(x_test)
y_pred_train = nb.predict(x_train)
acc.append(accuracy_score(y_test, y_pred_test))
acc_train.append(accuracy_score(y_train, y_pred_train))
model_names.append('Naive Bayes')
print("Naive Bayes Accuracy: Test = {:.2f}%, Train = {:.2f}%".format(acc[-1]*100, acc_train[-1]*100))
print(classification_report(y_test, y_pred_test))

# SVM with polynomial kernel
norm = MinMaxScaler().fit(x_train)
X_train_norm = norm.transform(x_train)
X_test_norm = norm.transform(x_test)
svm = SVC(kernel='poly', degree=3, C=1)
svm.fit(X_train_norm, y_train)
y_pred_test = svm.predict(X_test_norm)
y_pred_train = svm.predict(X_train_norm)
acc.append(accuracy_score(y_test, y_pred_test))
acc_train.append(accuracy_score(y_train, y_pred_train))
model_names.append('SVM')
print("SVM Accuracy: Test = {:.2f}%, Train = {:.2f}%".format(acc[-1]*100, acc_train[-1]*100))
print(classification_report(y_test, y_pred_test))

# Logistic Regression
logreg = LogisticRegression(random_state=2)
logreg.fit(x_train, y_train)
y_pred_test = logreg.predict(x_test)
y_pred_train = logreg.predict(x_train)
acc.append(accuracy_score(y_test, y_pred_test))
acc_train.append(accuracy_score(y_train, y_pred_train))
model_names.append('Logistic Regression')
print("Logistic Regression Accuracy: Test = {:.2f}%, Train = {:.2f}%".format(acc[-1]*100, acc_train[-1]*100))
print(classification_report(y_test, y_pred_test))

# Random Forest
rf = RandomForestClassifier(n_estimators=20, random_state=0)
rf.fit(x_train, y_train)
y_pred_test = rf.predict(x_test)
y_pred_train = rf.predict(x_train)
acc.append(accuracy_score(y_test, y_pred_test))
acc_train.append(accuracy_score(y_train, y_pred_train))
model_names.append('Random Forest')
print("Random Forest Accuracy: Test = {:.2f}%, Train = {:.2f}%".format(acc[-1]*100, acc_train[-1]*100))
print(classification_report(y_test, y_pred_test))

# Cross-validation scores for Random Forest
cv_scores = cross_val_score(rf, X, y, cv=5)
print("Random Forest Cross-validation scores:", cv_scores)

# Pickle the best model (Random Forest) and label encoder
with open('classifier.pkl', 'wb') as f:
    pickle.dump(rf, f)

with open('fertilizer.pkl', 'wb') as f:
    pickle.dump(encode_ferti, f)

print("Model and label encoder saved as classifier.pkl and fertilizer.pkl")
