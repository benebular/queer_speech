import re
import csv
import os
import pandas as pd
import numpy as np
from pathlib import Path

#### WARNING: pilot has different log files from the main experiment and needs to be processed separately. pos1, pos2, and pos3 are missing.
#### SECOND WARNIONG: ensure that stimlist.txt has been cleaned according to experiment standards.
# if not identical in shape, clean by hand in Excel first to remove duplicate participant responses and mark reject trials.

# script start
# subjects = ['A0280','A0318','A0417']
subjects = ['A0392','A0396','A0416'] # pilot
dialect = 'ameng'

for subject in subjects:

    dir = '/Volumes/hecate/biphon/meg/%s/'%subject
    os.chdir(dir)
    string = Path('%s-stimlist1.txt'%subject).read_text()
    df_orig = pd.read_csv('%s-stimlist1.txt'%subject,sep='\t',index_col=None)

    # the following items should be 912 for each stimulus item presented (304 trials total x 3 = 912)

    trial_pattern = r'A0.*?\t(.*?)\tSound' # change the subject's ID in the string
    trial_matches = re.findall(trial_pattern,string)
    print(len(trial_matches))
    # print(trial_matches)

    vowel_pattern = r'Sound\t(.*?)\t'
    vowel_matches = re.findall(vowel_pattern,string)
    print(len(vowel_matches))
    # print(vowel_matches)

    # the following items should be 304 items (actual number of ABX trials)

    time_pattern = r'Sound\t.*?\t(.*?)\t.*\n.*\tResponse'# change the subject's ID in the string
    time_matches = re.findall(time_pattern,string,flags=re.MULTILINE)
    print(len(time_matches))
    # print(time_matches)

    response_pattern = r'Response\t(.*?)\t'
    response_matches = re.findall(response_pattern,string)
    print(len(response_matches))
    # print(response_matches)

    RT_pattern = r'Response\t.*?\t(.*?)\t'
    RT_matches = re.findall(RT_pattern,string)
    print(len(RT_matches))
    # print(RT_matches)

    reject_pattern = r'Response\t.*?\t(.?)\n'
    reject_matches = re.findall(reject_pattern,string)
    print(len(reject_matches))
    # print(reject_matches)


    # response_pattern = r'Picture\tresponse.*\n.*\tResponse\t(.*?)\t'
    # response_matches = re.findall(response_pattern,string,flags=re.MULTILINE)
    # print(len(re.findall(response_pattern,string)))
    # print(response_matches)
    #
    # RT_pattern = r'Picture\tresponse.*\n.*\tResponse\t.*?\t(.*?)\t'
    # RT_matches = re.findall(response_pattern,string,flags=re.MULTILINE)
    # print(len(re.findall(RT_pattern,string)))
    # print(RT_matches)

    b = np.array([[i]*3 for i in time_matches]).flatten()
    print(len(b))
    c = np.array([[i]*3 for i in response_matches]).flatten()
    print(len(c))
    d = np.array([[i]*3 for i in RT_matches]).flatten()
    print(len(d))
    e = np.array([1,2,3])
    print(len(e))
    f = np.tile(e,304)
    print(len(f))
    g = np.array([[i]*3 for i in reject_matches]).flatten()
    print(len(g))

    # phon_entry = []
    # for i in phon_matches:
    #   if i not in phon_entry:
    #     phon_entry.append(i)


    df_matches = pd.DataFrame({'subject': subject, 'trial_presentation': trial_matches, 'vowel': vowel_matches, 'participant_response': c, 'time_trial': b, 'time_response':d, 'position':f, 'reject':g, 'dialect':dialect})
    df_matches['trial_order'] = range(1, 1+len(df_matches))
    # df_matches[['vowel','block','vowel_id','order','type','trialid','correct_response','nativeness','trigger','F1','F2','F3','pos1','pos2','pos3']] = df_matches.vowel.str.split(' ', expand=True) # post-pilot 3
    df_matches[['vowel','block','vowel_id','order','type','trialid','correct_response','nativeness','trigger','F1','F2','F3']] = df_matches.vowel.str.split(' ', expand=True) # pilot 3
    df_matches['speaker'] = df_matches['vowel_id'].str[-1:]
    df_matches['correct'] = np.where(( ( (df_matches['participant_response'] == '2') & (df_matches['correct_response'] == '1') ) | ( (df_matches['participant_response'] == '4') & (df_matches['correct_response'] == '2' ) ) ), 1, 0)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(df_matches)

    # add in dummy coding for individual vowels and vowel groups for decoding analysis
    trial_info = df_matches
    trial_info['vowel_id'] = trial_info['vowel_id'].astype('str')
    trial_info['vowel_iso'] = trial_info['vowel_id'].str.extract(r'(.*)\d')
    trial_info_dummies = pd.get_dummies(trial_info['vowel_iso'])
    trial_info = pd.concat([trial_info, trial_info_dummies], axis=1)
    di_1={"i":"1","y":"1","yih":"1"}
    di_2={"u":"1","uu":"1","ob":"1"}
    di_3={"ae":"1","ah":"1"}
    trial_info['vowel_grp1']=trial_info['vowel_iso'].map(di_1).fillna("0")
    trial_info['vowel_grp2']=trial_info['vowel_iso'].map(di_2).fillna("0")
    trial_info['vowel_grp3']=trial_info['vowel_iso'].map(di_3).fillna("0")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(trial_info)

    # add in pos1, pos2, and pos3 for pilot
    text_reference = pd.read_csv('/Volumes/hecate/biphon/meg/A0280/A0280_log_cleaned.csv') # using logfile from A280 as reference to pull pos1, pos2, pos3
    trial_info[['trialid','trigger']]=trial_info[['trialid','trigger']].astype(int) # change data types so it's all the same
    res = trial_info.merge(text_reference, on=['trialid','trigger'], how='inner', suffixes=('','_y')) # merge based on two conditions at the same time so it is unique, make left new df without suffix
    res.drop(res.filter(regex='_y$').columns.tolist(), axis=1,inplace=True) # regex filter and remove _y columns to create new df with just pos1, pos2, pos3 added
    trial_info = res # overwrite trial_info with new data
    trial_info['vowel_spk'] = trial_info['vowel'].str[6:]
    print(len(trial_info)) # sanity check
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(trial_info)

    # add in more dummy coding so it's easier to split trials by vowel groups
    # create columns with just the vowel w/ spk and pos
    # slice based on pos3
    # filter based on specific vowels in A and B

    #### Below section is if you need to run brand new pilot data and the text_reference above is not from a subejct that already has the pos1, pos2, pos3 cleaned up
    # trial_info['pos1'] = trial_info['pos1'].str[6:]
    # trial_info['pos2'] = trial_info['pos2'].str[6:]
    # trial_info['pos3'] = trial_info['pos3'].str[6:]
    # trial_info['pos1_nospk'] = trial_info['pos1'].str.strip('12')
    # trial_info['pos2_nospk'] = trial_info['pos2'].str.strip('12')
    # trial_info['pos3_nospk'] = trial_info['pos3'].str.strip('12')
    # trial_info = trial_info.drop('Unnamed: 0', axis=1) # pilot
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #         print(trial_info)

    trial_info.to_csv('%s_log_cleaned.csv'%subject, index=True, encoding='utf-8')
    # trial_info.to_csv('%s_log_cleaned_test.csv'%subject, index=True, encoding='utf-8')
