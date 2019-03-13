
from scipy.io import savemat
from sklearn.externals import joblib
from braininvaders.dataset import bi2013a
import numpy as np
import mne
import pandas as pd

"""
=====================================
Conversion to different data formats
=====================================

This scripts gives some options of how to extract the epochs from the .gdf
files and then saving them into files with standard formats

"""
# Authors: Pedro Rodrigues <pedro.rodrigues01@gmail.com>
#
# License: BSD (3-clause)

import warnings
warnings.filterwarnings("ignore")

# define the dataset instance
dataset = bi2013a(NonAdaptive=True, Adaptive=False, Training=True, Online=False)

# get the data from subject of interest
subject = dataset.subject_list[0]
data = dataset._get_single_subject_data(subject)

# specify which session and run we want
session = 2
run = 3 # we load only run 3 when choosing NonAdaptive + Training 
raw = data['session_' + str(session)]['run_' + str(run)]

# filter data and resample from 512 Hz to 128 Hz
fmin = 1
fmax = 24
raw.filter(fmin, fmax, verbose=False)
raw.resample(sfreq=128, verbose=False)

# saving filtered signals into a csv file
path = './subject_' + str(subject) + '_session_' + str(session) + '_run_' + str(run) + '_signals'
extension = '.csv'
filename = path + extension
d, t = raw[:, :]
df = pd.DataFrame(d.T, columns=raw.ch_names)
df.to_csv(filename)

# detect the events and cut the signal into epochs
events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
event_id = {'NonTarget': 33286, 'Target': 33285}
epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=1.0, baseline=None, verbose=False)

# get epochs and their labels
X = epochs.get_data()
y = events[:, -1]
y[y == 33286] = 0
y[y == 33285] = 1

# saving epochs into a pickle file
path = './subject_' + str(subject) + '_session_' + str(session) + '_run_' + str(run) + '_epochs'
extension = '.pkl'
filename = path + extension
data = {}
data['epochs'] = X
data['labels'] = y
joblib.dump(data, filename)

# saving epochs into a mat file
path = './subject_' + str(subject) + '_session_' + str(session) + '_run_' + str(run) + '_epochs'
extension = '.mat'
filename = path + extension
data = {}
data['epochs'] = X
data['labels'] = y
savemat(filename, data)





