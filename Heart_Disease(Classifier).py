# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset(Train)
dataset = pd.read_csv('Data.csv')
dataset.isnull().sum()
dataset.describe()
dataset.head()
dataset.info()

X = dataset.iloc[:, :-1].values


X_train = dataset.iloc[:623, :-1].values
y_train = dataset.iloc[:623, 13].values

X_test = dataset.iloc[623:, :-1].values
y_test = dataset.iloc[623:, 13].values



# Missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:, 0:13])
X_train[:, 0:13] = imputer.transform(X_train[:, 0:13])
#TEST
imputer_test = imputer.fit(X_test[:,:])
X_test[:,:] = imputer_test.transform(X_test[:,:])

#Backward elimination
#import statsmodels.formula.api as sm
#X = np.append(arr = np.ones((779,1)).astype(int), values = X, axis=1)
#X_opt = X[:,[0,1,2,3,5,9,11,12,13]]
#reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#reg_OLS.summary()

#Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler as ss
sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#################################
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)


y_pred_train = classifier.predict(X_train)
y_pred_test = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(y_train,y_pred_train)
##Test
cm_test = confusion_matrix(y_test,y_pred_test)

print('Accuracy of SVC for train set = {}'.format((cm_train[0][0] + cm_train[1][1])/(len(X_train))))
print('Accuracy of SVC for test set = {}'.format((cm_test[0][0] + cm_test[1][1])/(len(X_test))))
####################################

#####################################
#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10)
classifier.fit(X_train, y_train)

y_pred_train = classifier.predict(X_train)
y_pred_test = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(y_train,y_pred_train)
##Test
cm_test = confusion_matrix(y_test,y_pred_test)


print('Accuracy of Random Forest for train set = {}'.format((cm_train[0][0] + cm_train[1][1])/(len(X_train))))
print('Accuracy of Random Forest for test set = {}'.format((cm_test[0][0] + cm_test[1][1])/(len(X_test))))

########################################


#####################################
# Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred_train = classifier.predict(X_train)
y_pred_test = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(y_train,y_pred_train)
##Test
cm_test = confusion_matrix(y_test,y_pred_test)


print('Accuracy of Logistic Regression for train set = {}'.format((cm_train[0][0] + cm_train[1][1])/(len(X_train))))
print('Accuracy of Logistic Regression for test set = {}'.format((cm_test[0][0] + cm_test[1][1])/(len(X_test))))

########################################

#####################################
#Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred_train = classifier.predict(X_train)
y_pred_test = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(y_train,y_pred_train)
##Test
cm_test = confusion_matrix(y_test,y_pred_test)


print('Accuracy of Naive Bayes for train set = {}'.format((cm_train[0][0] + cm_train[1][1])/(len(X_train))))
print('Accuracy of Naive Bayes for test set = {}'.format((cm_test[0][0] + cm_test[1][1])/(len(X_test))))

########################################



#####################################
# KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)

y_pred_train = classifier.predict(X_train)
y_pred_test = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(y_train,y_pred_train)
##Test
cm_test = confusion_matrix(y_test,y_pred_test)


print('Accuracy of KNN for train set = {}'.format((cm_train[0][0] + cm_train[1][1])/(len(X_train))))
print('Accuracy of KNN for test set = {}'.format((cm_test[0][0] + cm_test[1][1])/(len(X_test))))

########################################





