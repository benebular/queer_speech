## Forced Alignment (textless) using Charsiu
# Author: Ben Lang
# e: blang@ucsd.edu
# use playground environment
# steps copied and modified from Google CoLab: https://colab.research.google.com/github/lingjzhu/charsiu/blob/development/charsiu_tutorial.ipynb#scrollTo=7HMM0nHe6zZG

import sys
import torch
import os
from itertools import groupby
import glob
import librosa
import soundfile as sf
sys.path.insert(0,'src')

charsiu_dir = '/Users/bcl/Documents/GitHub/charsiu/'
audio_dir = "/Volumes/GoogleDrive/My Drive/Comps/sounds_survey/pilot_jan21/sounds/"
audio_resample_dir = "/Volumes/GoogleDrive/My Drive/Comps/sounds_survey/pilot_jan21/sounds_resample/"
grid_dir = "/Volumes/GoogleDrive/My Drive/Comps/sounds_survey/pilot_jan21/grids/"

## resample
os.chdir(audio_dir)
all_files = glob.glob("*.wav")
n_files = len(all_files) # Number of files
for file in all_files:
    audio, sr = librosa.core.load(audio_dir + str(file), sr=16000)
    sf.write(audio_resample_dir + str(file), audio, sr)

# import selected model from Charsiu and initialize model
os.chdir(charsiu_dir)
from Charsiu import charsiu_predictive_aligner
# initialize model
charsiu = charsiu_predictive_aligner(aligner='charsiu/en_w2v2_fc_10ms') # a much larger model with higher accuracy

# perform textless alignment iteratively
for file in os.listdir(audio_resample_dir):
    print(file)
    # grab the audio file
    # print the Alignment
    # out put the results into grids folder
    # save with matching name
