
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from pyriemann.classification import MDM
from pyriemann.estimation import XdawnCovariances
from braininvaders.dataset import BrainInvaders2013
import numpy as np
import mne
"""
=============================
Classification of the trials
=============================

This example shows how to extract the epochs from the dataset of a given
subject and then classify them using Machine Learning techniques using
Riemannian Geometry. 

"""
# Authors: Pedro Rodrigues <pedro.rodrigues01@gmail.com>
#
# License: BSD (3-clause)

import warnings
warnings.filterwarnings("ignore")

# define the dataset instance
dataset = BrainInvaders2013(NonAdaptive=True, Adaptive=False, Training=True, Online=False)

# get the data from subject of interest
subject = dataset.subject_list[0]
data = dataset._get_single_subject_data(subject)
session = 1
raw = data['session_' + str(session)]['run_3']

# filter data and resample
fmin = 1
fmax = 24
raw.filter(fmin, fmax, verbose=False)
raw.resample(sfreq=128, verbose=False)

# detect the events and cut the signal into epochs
events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
event_id = {'NonTarget': 33286, 'Target': 33285}
epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=1.0, baseline=None, verbose=False)

# get trials and labels
X = epochs.get_data()
y = events[:, -1]
y[y == 33286] = 0
y[y == 33285] = 1

# cross validation
skf = StratifiedKFold(n_splits=5)
clf = make_pipeline(XdawnCovariances(estimator='lwf', classes=[1]), MDM())
scr = cross_val_score(clf, X, y, cv=skf, scoring='roc_auc')

# print results of classification
print('subject', subject, 'session', session)
print('mean AUC :', scr.mean())

