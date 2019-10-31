
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from pyriemann.classification import MDM
from pyriemann.estimation import ERPCovariances
from braininvaders2013.dataset import BrainInvaders2013
from sklearn.externals import joblib
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

scores = {}

# get the data from subject of interest
for subject in dataset.subject_list:
	
	scores[subject] = {}

	data = dataset._get_single_subject_data(subject)

	for session in data.keys():			

		raw = data[session]['run_3']

		# filter data and resample
		fmin = 1
		fmax = 24
		raw.filter(fmin, fmax, verbose=False)

		# detect the events and cut the signal into epochs
		events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
		event_id = {'NonTarget': 33286, 'Target': 33285}
		epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=1.0, baseline=None, verbose=False, preload=True)
		epochs.pick_types(eeg=True)

		# get trials and labels
		X = epochs.get_data()
		y = events[:, -1]
		y[y == 33286] = 0
		y[y == 33285] = 1

		# cross validation
		skf = StratifiedKFold(n_splits=5)
		clf = make_pipeline(ERPCovariances(estimator='lwf', classes=[1]), MDM())
		scr = cross_val_score(clf, X, y, cv=skf, scoring='roc_auc')

		# print results of classification
		scores[subject][session] = scr.mean()

		print('subject', subject, session)
		print(scr.mean())

filename = './classification_scores.pkl'		
joblib.dump(scores, filename)

with open('classification_scores.txt', 'w') as the_file:
    for subject in scores.keys():
    	for session in scores[subject].keys():
        	the_file.write('subject ' + str(subject).zfill(2) + ', ' + session + ' :' + ' {:.2f}'.format(scores[subject][session]) + '\n')

