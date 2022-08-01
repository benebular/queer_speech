% looping through to detect creak using the creaky detector
% outputs 1 or 0 for every 10ms of a file where creaky is detected
% requires resampled files at 16kHz
% author: Ben Lang, blang@ucsd.edu

% clone the COVAREP repository into your MATLAB directory, navigate to that
% directory, add two files: 'audio' and 'output'

% make a list of the resampled files, it's easiest if you just resample
% elsewhere through praat or python, and then place all of the resampled
% files in 'audio' in the COVAREP directory
filePattern = fullfile('./input/', '*.wav');
files = dir(filePattern);

% run creaky detection on each file in directory
% creaky detection loop is edited to write a file with the same filename
% for all output types

%% NEWER VERSION from COVAREP % https://github.com/covarep/covarep
% same instruction apply from above: make two folders in your MATLAB
% directory titled 'audio' and 'output,' and then put all the files you
% want to analyze in 'audio' and run this script from the covarep
% repository

for i=1:length(files)
    filename = files(i);
    filelocation = ['input/',filename.name]
    % Load soundfile
    [x,fs] = audioread(filelocation);

    % Check the speech signal polarity
    polarity = polarity_reskew(x,fs);
    x=polarity*x;

    warning off
    try
        [creak_pp,creak_bin] = detect_creaky_voice(x,fs); % Detect creaky voice
        creak=interp1(creak_bin(:,2),creak_bin(:,1),1:length(x));
        creak(creak<0.5)=0; creak(creak>=0.5)=1;
        do_creak=1;
    catch
        disp('Version or toolboxes do not support neural network object used in creaky voice detection. Creaky detection skipped.')
        creak=zeros(length(x),1);
        creak_pp=zeros(length(x),2);
        creak_pp(:,2)=1:length(x);
        do_creak=0;
    end
    warning on
    new_str = erase(filename.name,'.wav')
    new_file = [new_str, '.csv']
    new_filename = sprintf('%s','output/output_creak_',new_file)
    dlmwrite(new_filename,creak,'delimiter',',','-append');
end

%% OLD VERSION from Voice_Analysis_Toolkit % https://github.com/jckane/Voice_Analysis_Toolkit
% same instruction apply from above, except you need the old repository from the URL just above in this section: 
% make two folders in your MATLAB directory titled 'audio' and 'output,' and then put all the files you
% want to analyze in 'audio' and run this script from the Voice Analysis Toolkit repository

for i=1:length(files)
    filename = files(i);
    filelocation = ['audio/',filename.name]
    [wave,Fs]=audioread(filelocation); % grabs the name of the file that was inserted

    [Outs,Decs,t,H2H1,res_p] = CreakyDetection_CompleteDetection(wave,Fs);

    new_str = erase(filename.name,'.wav')
    new_file = [new_str, '.csv']
    new_filename_decision = sprintf('%s','output/output_decision_',new_file)
    new_filename_time = sprintf('%s','output/output_time_',new_file)
    new_filename_H2H1 = sprintf('%s','output/output_H2H1_',new_file)
    new_filename_resp = sprintf('%s','output/output_resp_',new_file)


    dlmwrite(new_filename_decision,Decs,'delimiter',',','-append');
    dlmwrite(new_filename_time,t,'delimiter',',','-append');
    dlmwrite(new_filename_H2H1,H2H1,'delimiter',',','-append');
    dlmwrite(new_filename_resp,res_p,'delimiter',',','-append');
end