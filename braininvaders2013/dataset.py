#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import mne
import numpy as np
from braininvaders2013 import download as dl
import os
import glob
import zipfile
import yaml
from scipy.io import loadmat

BI2013a_URL = 'https://zenodo.org/record/2645882/files/'

class BrainInvaders2013():
    '''
    We describe the experimental procedures for a dataset that we have made publicly available at 
    https://doi.org/10.5281/zenodo.1494163 in mat and csv formats. This dataset contains 
    electroencephalographic (EEG) recordings of 24 subjects doing a visual P300 Brain-Computer 
    Interface experiment on PC. The visual P300 is an event-related potential elicited by visual
    stimulation, peaking 240-600 ms after stimulus onset. The experiment was designed in order to 
    compare the use of a P300-based brain-computer interface on a PC with and without adaptive 
    calibration using Riemannian geometry. The brain-computer interface is based on 
    electroencephalography (EEG). EEG data were recorded thanks to 16 electrodes. A full 
    description of the experiment is available at https://hal.archives-ouvertes.fr/hal-02103098
    Data were recorded during an experiment taking place in the GIPSA-lab, Grenoble, France, in
    2013(Congedo, 2013). Python code for manipulating the data is available at 
    https://github.com/plcrodrigues/py.BI.EEG.2013-GIPSA. The ID of this dataset is BI.EEG.2013-GIPSA.

    **Full description of the experiment and dataset**
    https://hal.archives-ouvertes.fr/hal-02103098

    **Link to the data**
    https://doi.org/10.5281/zenodo.1494163
 
    **Authors**
    Principal Investigator: B.Sc. Erwan Vaineau, Ph.D. Alexandre Barachant
    Technical Supervisors: Eng. Anton Andreev, Eng. Pedro. L. C. Rodrigues, Eng. Grégoire Cattan
    Scientific Supervisor: Ph.D. Marco Congedo

    **ID of the dataset**
    BI.EEG.2013-GIPSA
    '''

    def __init__(self, NonAdaptive=True, Adaptive=False, Training=True, Online=False):

        self.adaptive = Adaptive
        self.nonadaptive = NonAdaptive
        self.training = Training
        self.online = Online
        self.subject_list = list(range(1, 24 + 1))

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""

        file_path_list = self.data_path(subject)
        sessions = {}
        for file_path in file_path_list:

            session_number = file_path.split(os.sep)[-2].strip('Session')
            session_name = 'session_' + session_number
            if session_name not in sessions.keys():
                sessions[session_name] = {}

            run_number = file_path.split(os.sep)[-1]
            run_number = run_number.split('_')[-1]
            run_number = run_number.split('.mat')[0]
            run_name = 'run_' + run_number

            chnames = ['FP1',
                        'FP2',
                        'F5',
                        'AFz',
                        'F6',
                        'T7',
                        'Cz',
                        'T8',
                        'P7',
                        'P3',
                        'Pz',
                        'P4',
                        'P8',
                        'O1',
                        'Oz',
                        'O2',
                        'STI 014']
            chtypes = ['eeg'] * 16 + ['stim']               

            X = loadmat(file_path)['data'].T
            info = mne.create_info(ch_names=chnames, sfreq=512,
                                   ch_types=chtypes, montage='standard_1020',
                                   verbose=False)
            raw = mne.io.RawArray(data=X, info=info, verbose=False)    

            sessions[session_name][run_name] = raw

        return sessions

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):

        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        # check if has the .zip
        url = '{:s}subject{:d}.zip'.format(BI2013a_URL, subject)
        path_zip = dl.data_path(url, 'BRAININVADERS2013')
        path_folder = path_zip.strip('subject{:d}.zip'.format(subject))

        # check if has to unzip
        if not(os.path.isdir(path_folder + 'subject{:d}/'.format(subject))):
            print('unzip', path_zip)
            zip_ref = zipfile.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder)

        # filter the data regarding the experimental conditions
        meta_file = 'subject{:d}/meta.yml'.format(subject)
        meta_path = path_folder + meta_file
        with open(meta_path, 'r') as stream:
            meta = yaml.load(stream)
        conditions = []
        if self.adaptive:
            conditions = conditions + ['adaptive']
        if self.nonadaptive:
            conditions = conditions + ['nonadaptive']
        types = []
        if self.training:
            types = types + ['training']
        if self.online:
            types = types + ['online']
        filenames = []
        for run in meta['runs']:
            run_condition = run['experimental_condition']
            run_type = run['type']
            if (run_condition in conditions) and (run_type in types):
                filenames = filenames + [run['filename']]

        # list the filepaths for this subject
        subject_paths = []
        for filename in filenames:
            subject_paths = subject_paths + \
                glob.glob(path_folder + 'subject{:d}/Session*/'.format(subject) + filename.replace('.gdf','.mat')) # noqa

        return subject_paths
