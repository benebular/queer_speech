## Forced Alignment (textless) using Charsiu
# Author: Ben Lang
# e: blang@ucsd.edu
# use playground environment
# steps copied and modified from Google CoLab: https://colab.research.google.com/github/lingjzhu/charsiu/blob/development/charsiu_tutorial.ipynb#scrollTo=7HMM0nHe6zZG

print("Importing libraries...")
import sys
import torch
import os
from os.path import exists, join, expanduser
from itertools import groupby
import glob
import librosa
import soundfile as sf

#make sure the clone is up to date
os.chdir(expanduser("~"))
charsiu_dir = 'charsiu'
if exists(charsiu_dir):
  !rm -rf /root/charsiu
if not exists(charsiu_dir):
  ! git clone -b development https://github.com/lingjzhu/$charsiu_dir
  ! cd charsiu && git checkout && cd -
os.chdir(charsiu_dir)

### do this one separately from import above
sys.path.insert(0,'/Users/bcl/charsiu/src')

print("Filling directory variables...")
charsiu_dir = '/Users/bcl/charsiu/'
audio_dir = "/Volumes/GoogleDrive/My Drive/Comps/sounds_survey/mark1_jan28/mark1_jan28_normalized/"
audio_resample_dir = "/Volumes/GoogleDrive/My Drive/Comps/sounds_survey/mark1_jan28/resample/"
grid_dir = "/Volumes/GoogleDrive/My Drive/Comps/sounds_survey/mark1_jan28/grids/"

## resample
print("Resampling audio at 16kHz.")
os.chdir(audio_resample_dir)
all_files = glob.glob("*.wav")
n_files = len(all_files) # Number of files
for file in all_files:
    audio, sr = librosa.core.load(audio_dir + str(file), sr=16000)
    sf.write(audio_resample_dir + str(file), audio, sr)

# import selected model from Charsiu and initialize model
# if it says it can't import, try running the source code in line 28 again
print ("Importing Charsiu model.")
os.chdir(charsiu_dir)
from Charsiu import charsiu_predictive_aligner
# initialize model
charsiu = charsiu_predictive_aligner(aligner='charsiu/en_w2v2_fc_10ms') # a much larger model with higher accuracy

# perform textless alignment iteratively
os.chdir(audio_resample_dir)
print("Finding audio files and aligning...")
for file in os.listdir(audio_resample_dir):
    # print()
    f_name = os.path.abspath(str(file))
    # grab the audio file
    alignment = charsiu.align(audio=f_name)
    # print the Alignment
    print(alignment)
    # out put the results into grids folder
    charsiu.serve(audio=f_name, save_to=grid_dir + os.path.splitext(file)[0] + ".TextGrid")
    # save with matching name
    print(str(file) + " " + "is complete.")
