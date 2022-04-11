
import mne, surfer, os
import matplotlib.pyplot as plt

root = '/Volumes/bin_battuta/biphon'
subjects_dir = os.path.join(root,'MRI')
os.chdir(root) # set up current directory

subjects = ['A0280','A0318','A0392','A0396','A0416','A0417']
src_fname = '/Volumes/bin_battuta/biphon/mri/fsaverage/bem/fsaverage-ico-4-src.fif'

for subj in subjects:
    print ('\n-----------------------------')
    print ('Plotting and saving visual responses in subject %s...' %subj)
    stc_fname = '/Volumes/bin_battuta/biphon/meg/A0280/A0280_evoked-lh.stc'
    native_fname = '/Volumes/bin_battuta/biphon/meg/A0280/A0280_evoked_native-lh.stc'
    nonnative_fname = '/Volumes/bin_battuta/biphon/meg/A0280/A0280_evoked_nonnative-lh.stc'
    fig_fname = '/Volumes/bin_battuta/biphon/meg/A0280/A0280_time_series.png'
    # stc_low_list_fname = os.path.join(root,'STC','%s/%s_full_low_list_dSPM-lh.stc'%(subj,subj))
    # stc_high_sent_fname = os.path.join(root,'STC','%s/%s_full_high_sent_dSPM-lh.stc'%(subj,subj))
    # stc_high_list_fname = os.path.join(root,'STC','%s/%s_full_high_list_dSPM-lh.stc'%(subj,subj))
    # fig_fname = os.path.join(root, 'MEG', '%s/%s_visual.png' %(subj,subj))


    '''..................... read in source space & load stcs ...................'''

    src = mne.read_source_spaces(src_fname)
    stc_all = mne.read_source_estimate(stc_fname)
    stc_nat = mne.read_source_estimate(native_fname)
    stc_non = mne.read_source_estimate(nonnative_fname)

    '''.......................2. create a plot with subplots .....................'''

    fig, axes = plt.subplots(2,1)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('Acitivty in the left superior temporal cortex', fontsize=16)
    fig.set_size_inches(16, 6)

    '''.................... plot sentence condition ................'''

    # read annotation files
    parc = mne.read_labels_from_annot('fsaverage','aparc', subjects_dir=subjects_dir,hemi='lh')

    # set the right label
    label = [i for i in parc if i.name.startswith('superiortemporal-lh')][0] #

    # extract time series by region label
    native = stc_nat.extract_label_time_course(label,src=src,mode='mean')
    nonnative = stc_non.extract_label_time_course(label,src=src,mode='mean')

    # plot time series
    axes[0].plot(stc_nat.times,native[0],color='#eb1a0c',alpha=0.6,label='Native', linewidth=2.0)
    axes[0].plot(stc_non.times,nonnative[0],color='#ffbe30',alpha=0.9,label='Nonnative', linewidth=2.0)

    # move legend box out of the plot
    axes[0].legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1)

    # set relevant time window
    axes[0].set_xlim(-0.2,1.2)

    # # set x-axis tick labels
    # events = [0.,0.6,1.2,1.8,2.4,3.0,3.6,4.2,4.8,5.4,6.0]
    # event_index = 0
    # while event_index < len(events):
    #     event = events[event_index]
    #     axes[0].axvline(event,color='red',alpha=0.5,linewidth=2)
    #     axes[0].set_xticks([])
    #     event_index += 1

    axes[0].axvline(0,color='red',alpha=0.5,linewidth=2,linestyle=':')
    axes[0].axvline(.5,color='red',alpha=0.5,linewidth=2,linestyle=':')
    axes[0].axvline(1,color='red',alpha=0.5,linewidth=2,linestyle=':')
    axes[0].set_xticks([0,.1,.5,.6,1,1.1,1.5])

    axes[0].set_title('native vs. non-native')
    axes[0].set_ylabel('Activation (dSPM)')


    '''.................... plot list condition ................'''

    # read annotation files
    # parc = mne.read_labels_from_annot('fsaverage','aparc',subjects_dir=subjects_dir,hemi='lh')

    # set the right label
    # label = parc[11]

    # extract time series by region label
    both = stc_all.extract_label_time_course(label,src=src,mode='mean')
    # low_list = stc_low_list.extract_label_time_course(label,src=src,mode='mean')

    # plot time series
    axes[1].plot(stc_all.times,both[0],color='#005421',alpha=0.9,label='both', linewidth=2.0)
    # axes[1].plot(stc_low_list.times,low_list[0],color='#00e6ac',alpha=0.6,label='Low_List', linewidth=2.0, linestyle='--')

    # move legend box out of the plot
    axes[1].legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1)

    # set relevant time window
    axes[1].set_xlim(-0.2,1.5)

    # # set x-axis tick labels
    # events = [0.,0.6,1.2,1.8,2.4,3.0,3.6,4.2,4.8,5.4,6.0]
    # event_index = 0
    # while event_index < len(events):
    #     event = events[event_index]
    #     axes[1].axvline(event,color='red',alpha=0.6,linewidth=1.5,linestyle=':')
    #     axes[1].set_xticks(events)
    #     event_index += 1

    axes[1].axvline(0,color='red',alpha=0.5,linewidth=2,linestyle=':')
    axes[1].axvline(.5,color='red',alpha=0.5,linewidth=2,linestyle=':')
    axes[1].axvline(1,color='red',alpha=0.5,linewidth=2,linestyle=':')
    axes[1].set_xticks([0,.1,.5,.6,1,1.1,1.5])


    axes[1].set_title('')
    axes[1].set_ylabel('Activation (dSPM)')
    axes[1].set_xlabel('Time (s)')

    fig.savefig(fig_fname, dpi=300)
