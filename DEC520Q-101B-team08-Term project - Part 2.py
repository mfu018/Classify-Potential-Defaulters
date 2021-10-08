
# import libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,precision_score,recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from six import StringIO
import graphviz 
from sklearn import tree, metrics
import scikitplot as skplt
from plot_metric.functions import BinaryClassification


# load dataset
data = pd.read_csv("PythonData.csv") #PythonData.csv is written by the R script
data.head()
data.describe()
data.info()

# remove unmeaningful variables
data = data.iloc[:,1:]
data.head()

# encode the non-numeric variables
en = LabelEncoder()
catCols = ['Married.Single','House_Ownership','Car_Ownership','Profession']
for cols in catCols:
    data[cols] = en.fit_transform(data[cols])
data.info()

# split the dataset to independent variables (X) and dependent variable (Y)
X = data.drop("Risk_Flag",axis = 1)
Y = data['Risk_Flag']

feature_names = ['Income', 
               'Age',
               'Experience', 
               'Married.Single',
               'House_Ownership',
               'Car_Ownership',
               'Profession', 
               'CURRENT_JOB_YRS',
               'CURRENT_HOUSE_YRS',
               'IncomeRatio']

# split the Xs and Ys into train set and test set
train_x, test_x, train_y, test_y=train_test_split(X, Y, test_size=0.2, random_state=0)
print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

# Modelling
print("****** Model Performance *****")
# 1) Decision Tree
dtc = tree.DecisionTreeClassifier()
dtc.fit(train_x, train_y)
predsdtc = dtc.predict(test_x)
print("DT accuracy_score: " + str(accuracy_score(test_y, predsdtc)))
print("DT ROC AUC score: " + str(roc_auc_score(test_y, predsdtc)))
print("DT Classification report")
print(classification_report(test_y, predsdtc))

# Plot the decision tree
text_representation = tree.export_text(dtc, feature_names=feature_names, max_depth =3)
print(text_representation)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dtc, max_depth=2,
                   feature_names=feature_names,
                   impurity = False,
                   filled=True)

# Plot confusion matrix
conf_matrix = confusion_matrix (test_y, predsdtc)

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

sns.heatmap(conf_matrix,ax=axes[0],annot=True, cmap='Blues', cbar=False, fmt='d')
axes[0].set_xlabel('\nPrediction', size = 14)
axes[0].set_ylabel('Actual\n', size = 14)

sns.heatmap(conf_matrix/np.sum(conf_matrix),ax=axes[1], annot=True, 
            fmt='.2%', cmap='Blues', cbar=False)
axes[1].set_xlabel('\nPrediction', size = 14)
axes[1].set_ylabel('Actual\n', size = 14)
axes[1].yaxis.tick_left()
plt.show()

# Plot ROC curve
fpr, tpr, thresholds = metrics.roc_curve(test_y, predsdtc)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name='ROC for Decision Tree', 
                                  )
display.plot()
plt.show()




# 2）Random Forest
rf = RandomForestClassifier(n_estimators = 400, criterion = 'entropy')
rf.fit(train_x, train_y)
rf.score(train_x, train_y)

predictions = rf.predict(test_x)

print("RF accuracy_score: " + str(accuracy_score(test_y, predictions)))
print("RF ROC AUC score: " + str(roc_auc_score(test_y, predictions)))
print("RF Classification report")
print(classification_report(test_y, predictions))

# Plot confusion matrix for the decision tree model
conf_matrix = confusion_matrix (test_y, predictions)

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

sns.heatmap(conf_matrix,ax=axes[0],annot=True, cmap='Blues', cbar=False, fmt='d')
axes[0].set_xlabel('\nPrediction', size = 14)
axes[0].set_ylabel('Actual\n', size = 14)

sns.heatmap(conf_matrix/np.sum(conf_matrix),ax=axes[1], annot=True, 
            fmt='.2%', cmap='Blues', cbar=False)
axes[1].set_xlabel('\nPrediction', size = 14)
axes[1].set_ylabel('Actual\n', size = 14)
axes[1].yaxis.tick_left()
plt.show()

# Plot ROC curve
fpr, tpr, thresholds = metrics.roc_curve(test_y, predictions)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name='ROC for Random Forest', 
                                  )
display.plot()
plt.show()


# Evaluation
# K-Fold Cross Validation
print("****** Cross Validation Model Performance *****")
k = 5
kf = KFold(n_splits=k, random_state=None)

# Decision Tree
model = tree.DecisionTreeClassifier()
 
dtacc_score = []
dtauc_score = []
 
for train_index , test_index in kf.split(X):
    X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
    y_train , y_test = Y[train_index] , Y[test_index]
     
    model.fit(X_train,y_train)
    pred_values = model.predict(X_test)
     
    dtacc = accuracy_score(pred_values , y_test)
    dtacc_score.append(dtacc)
    
    dtauc = roc_auc_score(pred_values , y_test)
    dtauc_score.append(dtauc)
     
avg_dtacc_score = sum(dtacc_score)/k
avg_dtauc_score = sum(dtauc_score)/k
 
#print('accuracy of each fold - {}'.format(acc_score))
print('DT Avg accuracy : {}'.format(avg_dtacc_score))
#print('AUC of each fold - {}'.format(auc_score))
print('DT Avg AUC : {}'.format(avg_dtauc_score))


# Random Forest
model = RandomForestClassifier(n_estimators = 400, criterion = 'entropy')
 
rfacc_score = []
rfauc_score = []
 
for train_index , test_index in kf.split(X):
    X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
    y_train , y_test = Y[train_index] , Y[test_index]
     
    model.fit(X_train,y_train)
    pred_values = model.predict(X_test)
     
    rfacc = accuracy_score(pred_values , y_test)
    rfacc_score.append(rfacc)
    
    rfauc = roc_auc_score(pred_values , y_test)
    rfauc_score.append(rfauc)
     
avg_rfacc_score = sum(rfacc_score)/k
avg_rfauc_score = sum(rfauc_score)/k
 
#print('accuracy of each fold - {}'.format(acc_score))
print('RF Avg accuracy : {}'.format(avg_rfacc_score))
#print('AUC of each fold - {}'.format(auc_score))
print('RF Avg AUC : {}'.format(avg_rfauc_score))


# Plot the reuslts from cross validation
# bar plot
X = ['1','2','3','4','5']
DT = dtauc
RF = rfauc
  
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, DT, 0.4, label = 'Decision Tree')
plt.bar(X_axis + 0.2, RF, 0.4, label = 'Random Forest')
  
plt.xticks(X_axis, X)
plt.xlabel("Folds")
plt.ylabel("OOS AUC")
plt.title("K-Fold Cross Validation")
plt.legend()
plt.ylim([0.7,1.0])
plt.show()
