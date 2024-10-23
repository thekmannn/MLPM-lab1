import time

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

from config import *

startTime = time.time()

print('Initialize Modeling')
print('    Loading training and test datasets')
X_train = np.loadtxt(X_TRAIN_PATH, delimiter = ",")
X_test = np.loadtxt(X_TEST_PATH, delimiter = ",")
y_train = np.loadtxt(Y_TRAIN_PATH, delimiter = ",")
y_test = np.loadtxt(Y_TEST_PATH, delimiter = ",")

print('    Running Support Vector Machine')
vect_m = svm.SVC()
vect_m.fit(X_train, y_train)

y_pred = vect_m.predict(X_test)
print(f'    Finished SVM with accuracy score', accuracy_score(y_test, y_pred))

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
