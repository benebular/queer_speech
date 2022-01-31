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

setwd('/Users/bcl/Documents/GitHub/queer_speech/qualtrics_data/pilot_jan21')
queer = read.csv('queer-speech_January 22, 2022_22.36.csv')

queer <- subset(queer, Status == 0 | Status == "Response Type")
queer <- select(queer, -contains(c("First.Click","Last.Click","Page.Submit","Click.Count")))
# queer$Participant <- 1:nrow(queer)-1
queer <- select(queer, -c(RecipientLastName, RecipientFirstName, RecipientEmail, ExternalReference, LocationLatitude, LocationLongitude, DistributionChannel, IPAddress))
queer <- select(queer, -c(P1_1:Q31_1))
queer <- queer %>% relocate(D2:Q47, .before = 1)
queer <- queer %>% relocate(Q56, .before = 1)
queer <- add_column(queer, Participant = 1:nrow(queer)-1, .before = 1)

queer[1,] <- gsub(".*- (.+) -.*", "\\1", queer[1,])
queer[1,] <- queer %>% separate(X1_Q36_1:X26_Q63_1, Question, Token)

queer_qid <- queer[-1,]
queer_stimname <- row_to_names(queer, row_number = 1)


# colnames(queer)[grepl('Q36',colnames(queer))] <- 'gender_id'
# colnames(queer)[grepl('Q60',colnames(queer))] <- 'gender_id'
# colnames(queer)[grepl('Q29',colnames(queer))] <- 'sexual_orientation'
# colnames(queer)[grepl('Q61',colnames(queer))] <- 'sexual_orientation'
# colnames(queer)[grepl('Q30',colnames(queer))] <- 'voice_effect'
# colnames(queer)[grepl('Q62',colnames(queer))] <- 'voice_effect'
# colnames(queer)[grepl('Q26',colnames(queer))] <- 'cis_trans'
# colnames(queer)[grepl('Q63',colnames(queer))] <- 'cis_trans'

queer_long <- queer %>% gather(Qualtrics_Trial_Num, Rating, X1_Q36_1:X26_Q63_1)
queer_long$Rating <- gsub(".*- (.+) -.*", "\\1", queer_long$Rating)
queer_long <- queer_long %>% spread(Qualtrics_Trial_Num, Participant)

