#!/usr/bin/env python
# coding: utf-8

# In[41]:


#importing libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from yellowbrick.classifier import ROCAUC
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


# In[42]:


# Read the excel file into a DataFrame
df = pd.read_csv("C:\\Users\\Vishnu P Kumar\\OneDrive\\Documents\\BreastTissue.csv")
df


# In[43]:


#Dropping Unnecessary Columns

df = df.drop(columns=['Case #'])


# In[44]:


df


# In[45]:


#Checking for Null values
df.isnull().sum()


# In[46]:


df.info()


# In[47]:


#changing the categorical column to label

labels,counts = pd.factorize(df["Class"])
df['Class']=labels


# In[48]:


labels


# In[49]:


counts


# In[50]:


#checking the correlation
df.corr()


# In[52]:


# Split the data into training and test sets
X = df.drop('Class',axis=1).values
Y = df['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,random_state=42)


# In[56]:


#import warnings filters
from warnings import simplefilter
#ignore all future warnings
simplefilter(action='ignore', category = FutureWarning)


# In[127]:


#import warning filters
import warnings
warnings.filterwarnings("ignore")


# LOGISTIC REGRESSION

# In[128]:


# Initialize the model
model = LogisticRegression(max_iter=10000)

# Fit the model to the training data
model.fit(X_train, y_train)

# Use the model to predict on the test data
y_pred_LR = model.predict(X_test)


# In[54]:


# Print the accuracy score of the model
Acc_scores={}
accuracy = accuracy_score(y_test, y_pred_LR)*100
print("Accuracy: {:.2f}%".format(accuracy))
LR={'LR':accuracy}
Acc_scores.update(LR)


# In[129]:


#Cross-Validation 10-Fold
kfold = KFold(n_splits=10, shuffle=True , random_state=10) 
score = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy').mean()
print(score)


# Classification Report

# In[57]:


from sklearn.metrics import classification_report
report = classification_report(y_test,y_pred_LR) 

print(report)


# Confusion Matrix

# In[58]:


conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_LR)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix of Logistic Regression', fontsize=18)
plt.show()


# Visualizing the ROC Curve

# In[62]:


visualizer = ROCAUC(model, classes=['car', 'fad', 'mas', 'gla', 'con', 'adi'])
visualizer.fit(X_train, y_train) 
visualizer.score(X_test, y_test)        
visualizer.show()   


# DECISION TREE
# 

# In[63]:


# Initialize the model
clf = DecisionTreeClassifier()

# Fit the model to the training data
clf.fit(X_train, y_train)

# Use the model to predict on the test data
y_pred_df = clf.predict(X_test)


# In[64]:


# Print the accuracy score of the model
accuracy = accuracy_score(y_test, y_pred_df)
print("Accuracy: {:.2f}%".format(accuracy*100))


# Using Grid Search to find best Parameters

# In[65]:


param_grid = {'max_depth': [3,4, 5, 10],
              'min_samples_leaf': [5, 10,15],
              'random_state' :range(20)}


# In[66]:


grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs = -1, verbose = 2)


# In[67]:


grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)


# In[68]:


# Use the best parameters to create a decision tree classifier
best_df = grid_search.best_estimator_

# Fit the new classifier to the training data
best_df.fit(X_train, y_train)

# Get predictions on the test data
y_pred_df = best_df.predict(X_test)

# Print accuracy score
accuracy = accuracy_score(y_test, y_pred_df)*100
print("Accuracy: {:.2f}%".format(accuracy))
DT={'Decision_Tree':accuracy}
Acc_scores.update(DT)


# In[69]:


#Cross-Validation 10-Fold
kfold = KFold(n_splits=10, shuffle=True , random_state=10) 
score = cross_val_score(best_df, X, Y, cv=kfold, scoring='accuracy').mean()
print(score)


# Classification Report

# In[130]:


from sklearn.metrics import classification_report
report = classification_report(y_test,y_pred_df) 

print(report)


# Confusion Matrix

# In[71]:


conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_df)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix of Decision Tree Classifier', fontsize=18)
plt.show()


# Visualizing the ROC Curve

# In[94]:


visualizer = ROCAUC(best_df, classes=['car', 'fad', 'mas', 'gla', 'con', 'adi'])
visualizer.fit(X_train, y_train) 
visualizer.score(X_test, y_test)        
visualizer.show()


# RANDOM FOREST

# In[95]:


# Initialize the model
rfr = RandomForestClassifier()

# Fit the model to the training data
rfr.fit(X_train, y_train)

# Use the model to predict on the test data
y_pred_rf = rfr.predict(X_test)


# In[96]:


accuracy = accuracy_score(y_test, y_pred_rf)
print("Accuracy: {:.2f}%".format(accuracy*100))


# Using Grid Search to find best Parameters
# 
# 

# In[97]:


param_grid = {'n_estimators': [50, 100, 200], 
              'max_depth': [5, 10, 15, 20], 
              'min_samples_leaf': [1, 2, 4,6,8]}


# In[98]:


# Create a GridSearchCV object
grid_search = GridSearchCV(rfr, param_grid, cv=5, n_jobs = -1, verbose = 2)


# In[100]:


# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)


# In[101]:


# Use the best parameters to create a new random forest classifier
best_rf = grid_search.best_estimator_

# Fit the new classifier to the training data
best_rf.fit(X_train, y_train)

# Get predictions on the test data
y_pred_rf = best_rf.predict(X_test)

# Print accuracy score
accuracy = accuracy_score(y_test, y_pred_rf)*100
print("Accuracy: {:.2f}%".format(accuracy))
RF={'Random_Forest':accuracy}
Acc_scores.update(RF)


# In[102]:


#Cross-Validation 10-Fold
kfold = KFold(n_splits=10, shuffle=True , random_state=10) 
score = cross_val_score(best_rf, X, Y, cv=kfold, scoring='accuracy').mean()
print(score)


# Classification Report

# In[103]:


from sklearn.metrics import classification_report
report = classification_report(y_test,y_pred_rf) 

print(report)


# Confusion Matrix

# In[104]:


conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_rf)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix of Random Forest Classifier', fontsize=18)
plt.show()


# Visualizing the ROC Curve

# In[105]:


visualizer = ROCAUC(best_rf, classes=['car', 'fad', 'mas', 'gla', 'con', 'adi'])
visualizer.fit(X_train, y_train) 
visualizer.score(X_test, y_test)        
visualizer.show()   


# SUPPORT VECTOR MACHINE

# In[131]:


# Initialize the model
svc = svm.SVC(kernel='linear',max_iter=1000)

# Fit the model to the training data
svc.fit(X_train, y_train)

# Use the model to predict on the test data
y_pred_svc = svc.predict(X_test)


# In[107]:


accuracy = accuracy_score(y_test, y_pred_svc)
print("Accuracy: {:.2f}%".format(accuracy*100))


# Using Grid Search to find best Parameters
# 
# 

# In[132]:


param_grid = {'C': [1, 10, 100, 1000],
              'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
              'degree': [2, 3, 4, 5],
              'gamma': [0.1, 0.01, 0.001, 0.0001]}


# In[133]:


# Create a GridSearchCV object
grid_search = GridSearchCV(svc, param_grid, cv=5, n_jobs = -1, verbose = 2)


# In[134]:


# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)


# In[135]:


# Use the best parameters to create a new random forest classifier
best_svc = grid_search.best_estimator_

# Fit the new classifier to the training data
best_svc.fit(X_train, y_train)

# Get predictions on the test data
y_pred_svc = best_svc.predict(X_test)

# Print accuracy score
accuracy = accuracy_score(y_test, y_pred_df)*100
print("Accuracy: {:.2f}%".format(accuracy))
SVC_={'SVC':accuracy}
Acc_scores.update(SVC_)


# In[136]:


#Cross-Validation 10-Fold
kfold = KFold(n_splits=10, shuffle=True , random_state=10) 
score = cross_val_score(best_svc, X, Y, cv=kfold, scoring='accuracy').mean()
print(score)


# Classification Report

# In[113]:


from sklearn.metrics import classification_report
report = classification_report(y_test,y_pred_svc) 

print(report)


# Confusion Matrix

# In[114]:


conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_svc)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix of Support Vector Machine', fontsize=18)
plt.show()


# K-NEAREST NEIGHBORS

# In[115]:


# Initialize the model
knn = KNeighborsClassifier()

# Fit the model to the training data
knn.fit(X_train, y_train)

# Use the model to predict on the test data
y_pred_knn = knn.predict(X_test)


# In[116]:


accuracy = accuracy_score(y_test, y_pred_knn)
print("Accuracy: {:.2f}%".format(accuracy*100))


# Using Grid Search to find best Parameters
# 
# 

# In[117]:


param_grid = {'n_neighbors': [3,5,7,9,11],
              'weights': ['uniform', 'distance'],'metric':['minkowski']}


# In[118]:


# Create a GridSearchCV object
grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs = -1, verbose = 2)


# In[119]:


# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)


# In[120]:


# Use the best parameters to create a new random forest classifier
best_knn = grid_search.best_estimator_

# Fit the new classifier to the training data
best_knn.fit(X_train, y_train)

# Get predictions on the test data
y_pred_knn = best_knn.predict(X_test)

# Print accuracy score
accuracy = accuracy_score(y_test, y_pred_knn)*100
print("Accuracy: {:.2f}%".format(accuracy))
knn_={'KNN':accuracy}
Acc_scores.update(knn_)


# In[121]:


#Cross-Validation 10-Fold
kfold = KFold(n_splits=10, shuffle=True , random_state=10) 
score = cross_val_score(best_knn, X, Y, cv=kfold, scoring='accuracy').mean()
print(score)


# Classification Report

# In[122]:


from sklearn.metrics import classification_report
report = classification_report(y_test,y_pred_knn) 

print(report)


# Confusion Matrix

# In[123]:


conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_knn)

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix of KNeighborsClassifier', fontsize=18)
plt.show()


# Visualizing the ROC Curve

# In[124]:


visualizer = ROCAUC(best_knn, classes=['car', 'fad', 'mas', 'gla', 'con', 'adi'])
visualizer.fit(X_train, y_train) 
visualizer.score(X_test, y_test)        
visualizer.show()  


# In[126]:


Classifier = list(Acc_scores.keys())
Accuracy = list(Acc_scores.values())
  
fig = plt.figure(figsize = (15, 8))
 
plt.bar(Classifier, Accuracy, color ='green',
        width = 0.4)
 
plt.xlabel("Classifier")
plt.ylabel("Accuracy")
plt.title("Comparison between Classifiers and Accuracy")
plt.show()


# In[ ]:




