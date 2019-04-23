
from scipy.io import savemat
from sklearn.externals import joblib
from braininvaders2013.dataset import BrainInvaders2013
import numpy as np
import mne
import pandas as pd

# define the dataset instance
dataset = BrainInvaders2013(NonAdaptive=True, Adaptive=False, Training=True, Online=False)

# get the data from subject of interest
for subject in dataset.subject_list:

	sessions = dataset._get_single_subject_data(subject)

	for session in sessions.keys():

		for run in sessions[session].keys():

			raw = sessions[session][run]

			# filter data and resample from 512 Hz to 128 Hz
			fmin = 1
			fmax = 24
			raw.filter(fmin, fmax, verbose=False)
			raw.resample(sfreq=128, verbose=False)

			# detect the events and cut the signal into epochs
			events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
			event_id = {'NonTarget': 33286, 'Target': 33285}
			epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=0.6, baseline=None, verbose=False, preload=True)
			epochs.pick_types(eeg=True)

			# get epochs and their labels
			X = epochs.get_data()
			y = events[:, -1]
			y[y == 33286] = 1
			y[y == 33285] = 2

			# saving epochs into a pickle file
			path_folder = '/research/vibs/Pedro/datasets/BrainInvaders2013/'		
			path_file = path_folder + 'subject_' + str(subject).zfill(2) + '_' + session + '.pkl'
			data = {}
			data['signals'] = X
			data['labels'] = y

			print(path_file)
			joblib.dump(data, path_file)

