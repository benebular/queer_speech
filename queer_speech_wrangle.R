## Queer Speech Data Analysis and Visualization Treasure Trove
## Author: Ben Lang, blang@ucsd.edu

# library(lmerTest)
library(plyr)
library(dplyr) # this checks for normality
# library(ggpubr) # this plots normality
library(magrittr)
# library(effects)
# library(ggplot2)
library(tidyr)
# library(scales)
library(reshape2)
# library(lme4)
# library(emmeans)
library(forcats)
# library(psycho)
library(tibble)
library(janitor)
library(data.table)

setwd('/Users/bcl/Documents/GitHub/queer_speech/qualtrics_data/mark1_jan28/')
options(stringsAsFactors=F)
queer = read.csv('/Volumes/GoogleDrive/My Drive/Comps/qualtrics_data/mark1_jan28/queer-speech_March 24, 2022_18.19.csv')

#cleaning data down to just ratings, trial type, and stimulus ID
# queer <- subset(queer, Status == 0 | Status == "Response Type") # just for if there's people that don't complete the survey
queer <- select(queer, -contains(c("First.Click","Last.Click","Page.Submit","Click.Count"))) # remove all the time counts, can add in later if needed
# queer$Participant <- 1:nrow(queer)-1
queer <- select(queer, -c(StartDate, EndDate, ResponseId, UserLanguage, Consent, HC2, RecipientLastName, RecipientFirstName, RecipientEmail, ExternalReference, LocationLatitude, LocationLongitude, DistributionChannel, IPAddress, Progress, Status, Finished, RecordedDate, Duration..in.seconds.))
queer <- select(queer, -c(P1_1:P5_1)) # eliminate practice questions
# queer <- select(queer, -c(D1:id)) # eliminate practice questions
# queer <- queer %>% relocate(D2:Q47, .before = 1)
# queer <- queer %>% relocate(Q56, .before = 1)
queer <- queer[-2,]
queer[1,] <- gsub(".*- (.+) -.*", "\\1", queer[1,]) # works on iMac, for some reason not on laptop, supposed to just grab URLs
queer <- add_column(queer, Participant = 1:nrow(queer)-1, .before = 1)


# splitting dfs to then concatenate
queer_qid <- queer[-1,]
ncol(queer_qid)

queer_qid_long <- queer_qid %>% gather(Qualtrics_Trial_Num, Rating, X1_Q36_1:X33_Q62_1)

# sanity check, should come out to 66 so that it's how many trials were responded to, divided by the number of conditions (3)
trial_counts <- queer_qid_long %>% group_by(Qualtrics_Trial_Num) %>% count(Qualtrics_Trial_Num)
nrow(trial_counts)/3

#queer[1,] <- queer %>% separate(X1_Q36_1:X33_Q62_1, Question, Token)
## second match, must remove extra columns to match long data above
queer_stimname <- select(queer, -c(D1:id, HC1)) # eliminate practice questions
queer_stimname <- row_to_names(queer_stimname, row_number = 1)
ncol(queer_stimname)

names(queer_stimname)[names(queer_stimname) == '0'] <- 'Participant'
names(queer_stimname) <- make.unique(names(queer_stimname), sep="*")
queer_stimname_long <- queer_stimname %>% gather(Trial_Type, Rating, -Participant)

# must be same length
nrow(queer_qid_long)
nrow(queer_stimname_long)

# remove duplicates columns from one, then merge together
queer_qid_long <- select(queer_qid_long, -c(Participant, Rating))
merged_queer <- bind_cols(queer_stimname_long, queer_qid_long)

## rename conditions
merged_queer$Qualtrics_Trial_Num[merged_queer$Qualtrics_Trial_Num %like% "Q36"] <- 'gender_id'
merged_queer$Qualtrics_Trial_Num[merged_queer$Qualtrics_Trial_Num %like% "Q60"] <- 'gender_id'
merged_queer$Qualtrics_Trial_Num[merged_queer$Qualtrics_Trial_Num %like% "Q29"] <- 'sexual_orientation'
merged_queer$Qualtrics_Trial_Num[merged_queer$Qualtrics_Trial_Num %like% "Q61"] <- 'sexual_orientation'
merged_queer$Qualtrics_Trial_Num[merged_queer$Qualtrics_Trial_Num %like% "Q30"] <- 'voice_id'
merged_queer$Qualtrics_Trial_Num[merged_queer$Qualtrics_Trial_Num %like% "Q62"] <- 'voice_id'
names(merged_queer)[names(merged_queer) == 'Qualtrics_Trial_Num'] <- 'Condition'

## remove uniqueness
matches = read.csv('matches_queer_speech.csv') # from regex
merged_queer <- merged_queer %>% mutate(Trial_Type = gsub("\\*1", "", Trial_Type))
merged_queer <- merged_queer %>% mutate(Trial_Type = gsub("\\*2", "", Trial_Type))

## merged the two dfs into big boi
matched_queer <- join(merged_queer, matches, by = "Trial_Type")
matched_queer <- select(matched_queer, -c(url, id, X, Trial_Type))
names(matched_queer)[names(matched_queer) == 'phon'] <- 'WAV'
matched_queer$Rating <- as.numeric(matched_queer$Rating)

### filter out D1, D2, HC1, D5, rename everything

# do some subsetting to find out how many people are excluded based on D1, D2, D5, HC1

# now do the actual filtering
matched_queer <- subset(matched_queer, HC1 == "Yes")
matched_queer <- select(matched_queer, -c(HC1))

# matched_queer <- subset(matched_queer, D1 == "Yes")
# matched_queer <- subset(matched_queer, D2 == "Yes")
# matched_queer <- subset(matched_queer, D5 == "No")
# matched_queer <- select(matched_queer, -c(F1, F2, D1, D2, D5))

# remove the blanks, Rating column is best because we don't want any missing ratings anyway
sum(!complete.cases(matched_queer$Rating[-1]))/66/3 # should be 13, corresponding to the 13 participants that have blank responses
matched_queer <- subset(matched_queer, Rating != "NA")

## just change names, no filtering
# names(matched_queer)[names(matched_queer) == 'HC1'] <- 'headphone_check'
names(matched_queer)[names(matched_queer) == 'D1'] <- 'eng_primary_early'
names(matched_queer)[names(matched_queer) == 'D2'] <- 'eng_primary_current'
names(matched_queer)[names(matched_queer) == 'D5'] <- 'deaf_hoh'
names(matched_queer)[names(matched_queer) == 'F1'] <- 'survey_experience'
names(matched_queer)[names(matched_queer) == 'F2'] <- 'survey_feedback'


names(matched_queer)[names(matched_queer) == 'D9_1'] <- 'participant_gender_id'
names(matched_queer)[names(matched_queer) == 'D10_1'] <- 'participant_sexual_orientation'
names(matched_queer)[names(matched_queer) == 'D11_1'] <- 'participant_voice_id'
names(matched_queer)[names(matched_queer) == 'D12_1'] <- 'participant_cis_trans'
names(matched_queer)[names(matched_queer) == 'D13'] <- 'participant_gender_pso_free_response'
names(matched_queer)[names(matched_queer) == 'PQ1_1'] <- 'participant_prox_social'
names(matched_queer)[names(matched_queer) == 'PQ2_1'] <- 'participant_prox_affiliation'
names(matched_queer)[names(matched_queer) == 'PQ3_1'] <- 'participant_prox_media'
names(matched_queer)[names(matched_queer) == 'D3'] <- 'participant_other_langs'
names(matched_queer)[names(matched_queer) == 'D4'] <- 'participant_age'
names(matched_queer)[names(matched_queer) == 'D6'] <- 'participant_race'
names(matched_queer)[names(matched_queer) == 'D7'] <- 'participant_race_hispanic'
names(matched_queer)[names(matched_queer) == 'D8'] <- 'participant_race_free_response'

### same as csv for analysis or plotting elsewhere
write.csv(matched_queer, "/Users/bcl/Documents/GitHub/queer_speech/qualtrics_data/mark1_jan28/matched_queer.csv", row.names=FALSE)
# write.csv(queer_stimname, "/Users/bcl/Documents/GitHub/queer_speech/qualtrics_data/mark1_jan28/queer_stimname.csv", row.names=FALSE)



### transform VoiceSauce features
vs = read.csv('/Users/bcl/Documents/GitHub/queer_speech/feature_extraction/vs_ms_output.csv')
vs_wide = vs %>% spread(Label, strF0_mean)

