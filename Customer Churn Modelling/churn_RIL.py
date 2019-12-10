
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('/Users/shreyas_rl/Desktop/Research/R4/churn_dataset2.csv')
X = dataset.iloc[:, 4:-2].values
y = dataset.iloc[:, -1].values


from sklearn.preprocessing import LabelEncoder , OneHotEncoder
label_encoder = LabelEncoder()
X[:,0] = label_encoder.fit_transform(X[:,0])
X[:,1] = label_encoder.fit_transform(X[:,1])


onehotencoder = OneHotEncoder(categorical_features = [0,1])
X = onehotencoder.fit_transform(X).toarray()

y = label_encoder.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""X_train = np.delete(X_train, 2, 1)
X_test = np.delete(X_test, 2, 1)

X_train = np.delete(X_train, 4, 1)
X_test = np.delete(X_test, 4, 1)

X_train = np.delete(X_train, 2, 1)
X_test = np.delete(X_test, 2, 1)

X_train = np.delete(X_train, 4, 1)
X_test = np.delete(X_test, 4, 1)

X_train = np.delete(X_train, 2, 1)
X_test = np.delete(X_test, 2, 1)

X_train = np.delete(X_train, 0, 1)
X_test = np.delete(X_test, 0, 1)"""



from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
import math
from statistics import mean



def sensit(cm):
    ans = cm[0,0]/(cm[0,0]+cm[1,0])
    return ans

def prec(cm):
    ans = cm[0,0]/(cm[0,0]+cm[0,1])
    return ans

#MACHINE LEARNING

from sklearn.ensemble import ExtraTreesClassifier
classifier = ExtraTreesClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

acc = cross_val_score(classifier, X_test, y_test, scoring="accuracy")
sensitivity = sensit(cm)
rec = cross_val_score(classifier, X_test, y_test, scoring="recall")
bcr = math.sqrt(sensitivity*mean(rec))
p = prec(cm)
f1 = cross_val_score(classifier, X_test, y_test, scoring="f1")
mcc = matthews_corrcoef(y_test, y_pred)

false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_pred)
auc = metrics.auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

acc = cross_val_score(classifier, X_test, y_test, scoring="accuracy")
sensitivity = sensit(cm)
rec = cross_val_score(classifier, X_test, y_test, scoring="recall")
bcr = math.sqrt(sensitivity*mean(rec))
p = prec(cm)
f1 = cross_val_score(classifier, X_test, y_test, scoring="f1")
mcc = matthews_corrcoef(y_test, y_pred)



false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_pred)
auc = metrics.auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

acc = cross_val_score(classifier, X_test, y_test, scoring="accuracy")
sensitivity = sensit(cm)
rec = cross_val_score(classifier, X_test, y_test, scoring="recall")
bcr = math.sqrt(sensitivity*mean(rec))
p = prec(cm)
f1 = cross_val_score(classifier, X_test, y_test, scoring="f1")
mcc = matthews_corrcoef(y_test, y_pred)



false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_pred)
metrics.auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


from sklearn.ensemble import AdaBoostClassifier
classifier = AdaBoostClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

acc = cross_val_score(classifier, X_test, y_test, scoring="accuracy")
sensitivity = sensit(cm)
rec = cross_val_score(classifier, X_test, y_test, scoring="recall")
bcr = math.sqrt(sensitivity*mean(rec))
p = prec(cm)
f1 = cross_val_score(classifier, X_test, y_test, scoring="f1")
mcc = matthews_corrcoef(y_test, y_pred)



false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_pred)
metrics.auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


from xgboost import XGBClassifier


classifier = XGBClassifier(random_state=1,learning_rate=0.01)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

acc = cross_val_score(classifier, X_test, y_test, scoring="accuracy")
sensitivity = sensit(cm)
rec = cross_val_score(classifier, X_test, y_test, scoring="recall")
bcr = math.sqrt(sensitivity*mean(rec))
p = prec(cm)
f1 = cross_val_score(classifier, X_test, y_test, scoring="f1")
mcc = matthews_corrcoef(y_test, y_pred)


false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_pred)
metrics.auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


from sklearn import svm
classifier = svm.SVC(gamma = 0.001, C = 100)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

acc = cross_val_score(classifier, X_test, y_test, scoring="accuracy")
sensitivity = sensit(cm)
rec = cross_val_score(classifier, X_test, y_test, scoring="recall")
bcr = math.sqrt(sensitivity*mean(rec))
p = prec(cm)
f1 = cross_val_score(classifier, X_test, y_test, scoring="f1")
mcc = matthews_corrcoef(y_test, y_pred)


false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_pred)
metrics.auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

acc = cross_val_score(classifier, X_test, y_test, scoring="accuracy")
sensitivity = sensit(cm)
rec = cross_val_score(classifier, X_test, y_test, scoring="recall")
bcr = math.sqrt(sensitivity*mean(rec))
p = prec(cm)
f1 = cross_val_score(classifier, X_test, y_test, scoring="f1")
mcc = matthews_corrcoef(y_test, y_pred)


false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_pred)
metrics.auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()





from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train) 

importances = classifier.feature_importances_

y_pred = classifier.predict(X_test)

acc = cross_val_score(classifier, X_test, y_test, scoring="accuracy")
f1 = cross_val_score(classifier, X_test, y_test, scoring="f1")
rec = cross_val_score(classifier, X_test, y_test, scoring="recall")
cm = confusion_matrix(y_test, y_pred)


false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_pred)
metrics.auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show() 

#DEEP LEARNING

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization


classifier = Sequential()


classifier.add(Dense(units = 516, kernel_initializer = 'uniform', activation = 'relu', input_dim = 17))


classifier.add(Dense(units = 1024, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(BatchNormalization())

classifier.add(Dense(units = 2048, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

classifier.add(Dense(units = 1024, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 512, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


classifier.fit(X_train, y_train, batch_size = 500, epochs = 25)


y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

cm8 = confusion_matrix(y_test, y_pred)
