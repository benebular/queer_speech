## Queer Speech Data Analysis and Visualization Treasure Trove
## Author: Ben Lang, blang@ucsd.edu

library(lmerTest)
library(dplyr) # this checks for normality 
library(ggpubr) # this plots normality
library(magrittr)
library(effects)
#library(ggplot2)
library(tidyr)
#library(scales)
library(reshape2)
library(lme4)
library(emmeans)
library(forcats)
library(psycho)
library(tibble)
library(janitor)

setwd('/Users/bcl/Documents/GitHub/queer_speech/qualtrics_data/mark1_jan28/')

queer = read.csv('queer-speech_February 1, 2022_22.51.csv')

#cleaning data down to just ratings, trial type, and stimulus ID
# queer <- subset(queer, Status == 0 | Status == "Response Type") # just for if there's people that don't complete the survey
queer <- select(queer, -contains(c("First.Click","Last.Click","Page.Submit","Click.Count"))) # remove all the time counts, can add in later if needed
# queer$Participant <- 1:nrow(queer)-1
queer <- select(queer, -c(StartDate, EndDate, ResponseId, UserLanguage, Consent, HC1, HC2, RecipientLastName, RecipientFirstName, RecipientEmail, ExternalReference, LocationLatitude, LocationLongitude, DistributionChannel, IPAddress, Progress, Status, Finished, RecordedDate, Duration..in.seconds.))
queer <- select(queer, -c(P1_1:P5_1)) # eliminate practice questions
queer <- select(queer, -c(D1:id)) # eliminate practice questions
# queer <- queer %>% relocate(D2:Q47, .before = 1)
# queer <- queer %>% relocate(Q56, .before = 1)
queer <- queer[-2,]
queer <- add_column(queer, Participant = 1:nrow(queer)-1, .before = 1)
# queer[1,] <- gsub(".*- (.+) -.*", "\\1", queer[1,]) # works on iMac for some reason, supposed to just grab URLs


# splitting dfs to then concatenate
queer_qid <- queer[-1,]
queer_qid_long <- queer_qid %>% gather(Qualtrics_Trial_Num, Rating, X1_Q36_1:X33_Q62_1)

#queer[1,] <- queer %>% separate(X1_Q36_1:X33_Q62_1, Question, Token)
queer_stimname <- row_to_names(queer, row_number = 1)
names(queer_stimname)[names(queer_stimname) == '0'] <- 'Participant'
queer_stimname_long <- queer_stimname %>% gather(Trial_Type, Rating, -Participant)



# colnames(queer)[grepl('Q36',colnames(queer))] <- 'gender_id'
# colnames(queer)[grepl('Q60',colnames(queer))] <- 'gender_id'
# colnames(queer)[grepl('Q29',colnames(queer))] <- 'sexual_orientation'
# colnames(queer)[grepl('Q61',colnames(queer))] <- 'sexual_orientation'
# colnames(queer)[grepl('Q30',colnames(queer))] <- 'voice_effect'
# colnames(queer)[grepl('Q62',colnames(queer))] <- 'voice_effect'
# colnames(queer)[grepl('Q26',colnames(queer))] <- 'cis_trans'
# colnames(queer)[grepl('Q63',colnames(queer))] <- 'cis_trans'


queer_long$Rating <- gsub(".*- (.+) -.*", "\\1", queer_long$Rating)
queer_long <- queer_long %>% spread(Qualtrics_Trial_Num, Participant)

