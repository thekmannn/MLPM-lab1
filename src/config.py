import os

PREFIX = ".." if os.path.dirname(os.getcwd()) == "src" else ""

DATA_PATH = os.path.join(PREFIX, 'data/')
X_TRAIN_PATH = os.path.join(DATA_PATH, 'X_train.csv')
X_TEST_PATH = os.path.join(DATA_PATH, 'X_test.csv')
Y_TRAIN_PATH = os.path.join(DATA_PATH, 'y_train.csv')
Y_TEST_PATH = os.path.join(DATA_PATH, 'y_test.csv')