import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from config import *


startTime = time.time()


print('Initialize Modeling')
print('    Loading training and test datasets')
X_train = np.loadtxt(X_TRAIN_PATH, delimiter = ",")
X_test = np.loadtxt(X_TEST_PATH, delimiter = ",")
y_train = np.loadtxt(Y_TRAIN_PATH, delimiter = ",")
y_test = np.loadtxt(Y_TEST_PATH, delimiter = ",")


print('    Running logistic regression')
log_reg_classifier = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
log_reg_classifier.fit(X_train, y_train)
y_pred = log_reg_classifier.predict(X_test)
print('    Finished modeling with accuracy score', accuracy_score(y_test, y_pred))


executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
