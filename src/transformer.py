import glob
import time

import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import os
import csv

from config import *


startTime = time.time()
SCALE_AUDIO = True


def getPitch(x,fs,winLen=0.02):
  p = winLen*fs
  frame_length = int(2**int(p-1).bit_length())
  hop_length = frame_length//2
  f0, voiced_flag, voiced_probs = librosa.pyin(y=x, fmin=80, fmax=450, sr=fs,
                                                 frame_length=frame_length,hop_length=hop_length)
  return f0,voiced_flag


print('Preprocessing raw data')
print('    Loading raw data')
files = glob.glob(os.path.join(DATA_PATH, '*.wav'))


print('    Extracting labels')
MLENDHW_table = [] 
for file in files:
    file_name = file.split('/')[-1]
    participant_ID = file.split('/')[-1].split('_')[0]
    interpretation_type = file.split('/')[-1].split('_')[1]
    song = file.split('/')[-1].split('_')[2].split('.')[0]
    MLENDHW_table.append([file_name,participant_ID,interpretation_type, song])
MLENDHW_df = pd.DataFrame(MLENDHW_table,columns=['file_id','participant','interpretation','song']).set_index('file_id') 


print('    Extracting audio features')
X,y =[],[]
for fileID in tqdm(MLENDHW_df.index):
    yi = MLENDHW_df.loc[fileID]['interpretation']=='hum'
    fs = None 

    x, fs = librosa.load(fileID,sr=fs)
    if SCALE_AUDIO: x = x/np.max(np.abs(x))
    f0, voiced_flag = getPitch(x,fs,winLen=0.02)
      
    power = np.sum(x**2)/len(x)
    voiced_fr = np.mean(voiced_flag)    

    xi = [power, voiced_fr]
    X.append(xi)
    y.append(yi*1)


# Saving the extracted features locally
with open(X_PATH, 'w', newline='') as file:
    csv.writer(file).writerows(X)

# Saving the labels locally
with open(Y_PATH, 'w', newline='') as file:
    csv.writer(file).writerows([[value] for value in y])


executionTime = (time.time() - startTime)
print('Execution time for the transformation process in seconds: ' + str(executionTime))
