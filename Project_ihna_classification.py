#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import mne
import glob
import os
import re
import numpy as np
from matplotlib import pyplot as plt
from mne.time_frequency import psd_multitaper
from mne.time_frequency import psd_welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import pandas as pd


# In[ ]:


def eeg_power_band(epochs):
    fr_bands = {
                  "theta": [4, 8],
                  "alpha": [8, 12],
                  "beta": [12, 30]}
    
    psds, freqs =  psds, freqs = psd_multitaper(epochs)
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []
    for fmin, fmax in fr_bands.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)
def predict(a,b):  
    k=[]
    for t in range(len(last_list)):
        y = ['n']*last_list[t][a].shape[0] + ['m']*last_list[t][b].shape[0]
        X = np.concatenate((last_list[t][a],last_list[t][b]),axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        ppn.fit(X_train, y_train)
        y_pred = ppn.predict(X_test)
        m=accuracy_score(y_test, y_pred)
        k.append(m)
    return k


# In[ ]:


montage = mne.channels.read_montage('GSN-HydroCel-128')
events_list = [241,242,244]
#os.makedirs(path_res)
last_list = []
for f,h in enumerate(files):
    paths = glob.glob(path + '/{0}/Reals/*.edf'.format(files[f]))
    epochs_list = []
    #os.makedirs(subj_path)
    for k,m in enumerate(events_list):
        for j,i in enumerate([x for x in paths if '{0}'.format(m) in x]): 
            event_id = dict(a=m)
            raw = mne.io.read_raw_edf(i)
            if  len(raw.times)//500 < 7:
                continue  
            new_events = mne.make_fixed_length_events(raw, id=m,start=5, duration=2, overlap=1)
            if j>=1:
                try:
                    epochs = mne.concatenate_epochs([mne.Epochs(raw, new_events, event_id = event_id, tmin=0, tmax=2, baseline=(None, 0),preload=True), epochs])
                except: 
                    pass
            else:
                epochs = mne.Epochs(raw, new_events, event_id = event_id , tmin=0, tmax=2, baseline=(None, 0),preload=True)
        epochs_list.append(epochs.copy())
    for teta in range(len(epochs_list)):
        try:
            new_names = {}
            new_names = dict(
                    (ch_name,
                     ch_name.replace('-', '').replace('Chan ', 'E').replace('CAR', '').replace('EEG ', '').replace('CA', '').replace(' ', ''))
                     for ch_name in epochs_list[teta].ch_names)
            epochs_list[teta].rename_channels(new_names)
            epochs_list[teta].set_montage(montage)
            epochs_list[teta].drop_channels(['E8','E14','E21','E25','E43','E48','E49','E56','E57','E63','E64','E65','E68','E69','E73','E74','E81','E82','E88','E89','E90','E94','E95','E99','E100','E107','E113','E119','E120','E125','E126','E127','E128'])
        except:
            pass
    prelast_list = []
    for beta in range(len(epochs_list)):
        dd = eeg_power_band(epochs_list[beta])
        prelast_list.append(dd)
    last_list.append(prelast_list)            
                


# In[ ]:


df = pd.DataFrame(columns=['241/242','242/244','241/244'], index=indexes)
df['241/242'] = predict(0,1)
df['242/244'] = predict(1,2)
df['241/244'] = predict(0,2)
writer = pd.ExcelWriter('subjects_results_classification.xlsx', engine='xlsxwriter')
df.to_excel(writer)
writer.save()

