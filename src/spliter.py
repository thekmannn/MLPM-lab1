import time

import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import os

from config import *

startTime = time.time()

# Loading the extracted features
with open(X_PATH, 'r', newline='') as file:
    X = [[float(item) for item in row] for row in csv.reader(file)]

# Loading the labels
with open(Y_PATH, 'r', newline='') as file:
    y = [float(row[0]) for row in csv.reader(file)]

print('    Splitting dataset into train and test datasets')
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.2, random_state=42)


print('    Saving data to file')
np.savetxt(X_TRAIN_PATH, X_train, delimiter=",")
np.savetxt(X_TEST_PATH, X_test, delimiter=",")
np.savetxt(Y_TRAIN_PATH, y_train, delimiter=",")
np.savetxt(Y_TEST_PATH, y_test, delimiter=",")


executionTime = (time.time() - startTime)
print('Execution time for the splitting process in seconds: ' + str(executionTime))
