## BIPHON Data Analysis and Visualization Treasure Trove
## Author: Ben Lang, blang@ucsd.edu

library(lmerTest)
library(dplyr) # this checks for normality 
library(ggpubr) # this plots normality
library(magrittr)
library(effects)
#library(ggplot2)
library(tidyr)
#library(scales)
#library(reshape2)
library(lme4)
library(emmeans)
library(forcats)
library(psycho)

setwd('/Users/bcl/Desktop/lab_stuff/SNL2020')
biphon_nb = read.csv('biphon_combined_bhv_dec_20201004.csv')
biphon_nb$correct <- as.factor(biphon_nb$correct)
biphon_nb$vowel_id <- as.factor(biphon_nb$vowel_id)
# biphon_nb$trial_order <- as.factor(biphon_nb$trial_order)
biphon_nb$RT <- abs(biphon_nb$time_trial-biphon_nb$time_response)
biphon_nb <- mutate(biphon_nb, LogRT = log(RT))

# biphon_neuro_bhv$vowel_id <- as.factor(biphon_neuro_bhv$vowel_position)
# biphon_nb <- biphon_nb[,-1] # run for as many times as you need to remove the first unnamed columns
# biphon_neuro_bhv$RT = biphon_neuro_bhv$RT * 1000
# biphon_neuro_bhv = biphon_neuro_bhv[biphon_neuro_bhv$hit == 1, ]
# # first check normality
# set.seed(1234)
# dplyr::sample_n(biphon_nb, 15)
# ggqqplot(biphon_nb$dSPM) # make qq plot
# ggdensity(biphon_nb$dSPM,  # make density plot
#           main = "Density plot of dSPM",
#           xlab = "dSPM")
# shapiro.test(biphon_nb$dSPM) # test distribution against normal distribution
# wrangle some data by deleting columns and adding in new columns of information
# biphon_nb <- select(biphon_nb, -(Subject.1:Order))
# biphon_nb <- select(biphon_nb, -(Unnamed..0:Unnamed..0.1.1))
# View(biphon_nb)
# wide_biphon_nb <- biphon_nb %>% spread(vowel_position, vowel_id)
# head(wide_biphon_nb, 24)
# A0396 <- filter(wide_biphon_nb, Subject == 'A0396')
# A0392 <- filter(wide_biphon_nb, Subject == 'A0392')
# A0416 <- filter(wide_biphon_nb, Subject == 'A0416')

###### SUBSET: based on native and non-native trials

biphon_native <- subset(biphon_nb, nativeness==1)
biphon_nonnative <- subset(biphon_nb, nativeness==0)

biphon_native_A <- filter(biphon_native, position=='1')
biphon_nonnative_A <- filter(biphon_nonnative, position=='1')

biphon_native_B <- filter(biphon_native, position=='2')
biphon_nonnative_B <- filter(biphon_nonnative, position=='2')

biphon_native_X <- filter(biphon_native, position=='3')
biphon_nonnative_X <- filter(biphon_nonnative, position=='3')

# subsets based on nonnative trials with given vowel token as X (decision) token
biphon_nonnative_X_ob <- subset(biphon_nonnative_X, vowel_iso == 'ob')
biphon_nonnative_X_uu <- subset(biphon_nonnative_X, vowel_iso == 'uu')
biphon_nonnative_X_u <- subset(biphon_nonnative_X, vowel_iso == 'u')
biphon_nonnative_X_y <- subset(biphon_nonnative_X, vowel_iso == 'y')
biphon_nonnative_X_yih <- subset(biphon_nonnative_X, vowel_iso == 'yih')
biphon_nonnative_X_i <- subset(biphon_nonnative_X, vowel_iso == 'i')
biphon_nonnative_X_ah <- subset(biphon_nonnative_X, vowel_iso == 'ah')
biphon_nonnative_X_ae <- subset(biphon_nonnative_X, vowel_iso == 'ae')

biphon_nonnative_X_ob__u <- subset(biphon_nonnative_X_ob, pos1_nospk == 'u' | pos2_nospk == 'u')
biphon_nonnative_X_u__ob <- subset(biphon_nonnative_X_u, pos1_nospk == 'ob' | pos2_nospk == 'ob')
biphon_nonnative_X_uu__u <- subset(biphon_nonnative_X_uu, pos1_nospk == 'u' | pos2_nospk == 'u')
biphon_nonnative_X_u__uu <- subset(biphon_nonnative_X_u, pos1_nospk == 'uu' | pos2_nospk == 'uu')

biphon_nonnative_X_y__i <- subset(biphon_nonnative_X_y, pos1_nospk == 'i' | pos2_nospk == 'i')
biphon_nonnative_X_i__y <- subset(biphon_nonnative_X_i, pos1_nospk == 'y' | pos2_nospk == 'y')
biphon_nonnative_X_yih__i <- subset(biphon_nonnative_X_yih, pos1_nospk == 'i' | pos2_nospk == 'i')
biphon_nonnative_X_i__yih <- subset(biphon_nonnative_X_i, pos1_nospk == 'yih' | pos2_nospk == 'yih')

biphon_nonnative_X_y__u <- subset(biphon_nonnative_X_y, pos1_nospk == 'u' | pos2_nospk == 'u')
biphon_nonnative_X_u__y <- subset(biphon_nonnative_X_u, pos1_nospk == 'y' | pos2_nospk == 'y')

#add in contrast column
biphon_nonnative_X_ob__u$contrast <- c('{u-ø}ø')
biphon_nonnative_X_u__ob$contrast <- c('{u-ø}u')
biphon_nonnative_X_uu__u$contrast <- c('{u-ɯ}ɯ')
biphon_nonnative_X_u__uu$contrast <- c('{u-ɯ}u')

vowel_grp2 <- c('{u-ø}ø','{u-ø}u','{u-ɯ}ɯ','{u-ɯ}u')

biphon_nonnative_X_y__i$contrast <- c('{i-y}y')
biphon_nonnative_X_i__y$contrast <- c('{i-y}i')
biphon_nonnative_X_yih__i$contrast <- c('{i-ʏ}ʏ')
biphon_nonnative_X_i__yih$contrast <- c('{i-ʏ}i')

vowel_grp1 <- c('{i-y}y','{i-y}i','{i-ʏ}ʏ','{i-ʏ}i')

biphon_nonnative_X_y__u$contrast <- c('{u-y}y')
biphon_nonnative_X_u__y$contrast <- c('{u-y}u')

vowel_grp4 <- c('{u-y}y','{u-y}u')

## convert accuracy scores to percentages
biphon_native_acc <- biphon_native %>%
  group_by(vowel_iso,correct) %>%
  summarise(count=n()) %>%
  mutate(perc=count/sum(count))

biphon_nonnative_acc <- biphon_nonnative %>%
  group_by(vowel_iso,correct) %>%
  summarise(count=n()) %>%
  mutate(perc=count/sum(count))

biphon_nonnative_acc__ob__u <- biphon_nonnative_X_ob__u %>%
  group_by(contrast,correct) %>%
  summarise(count=n()) %>%
  mutate(perc=count/sum(count))

biphon_nonnative_acc__u__ob <- biphon_nonnative_X_u__ob %>%
  group_by(contrast,correct) %>%
  summarise(count=n()) %>%
  mutate(perc=count/sum(count))

biphon_nonnative_acc__uu__u <- biphon_nonnative_X_uu__u %>%
  group_by(contrast,correct) %>%
  summarise(count=n()) %>%
  mutate(perc=count/sum(count))

biphon_nonnative_acc__u__uu <- biphon_nonnative_X_u__uu %>%
  group_by(contrast,correct) %>%
  summarise(count=n()) %>%
  mutate(perc=count/sum(count))

biphon_nonnative_acc__y__i <- biphon_nonnative_X_y__i %>%
  group_by(contrast,correct) %>%
  summarise(count=n()) %>%
  mutate(perc=count/sum(count))

biphon_nonnative_acc__i__y <- biphon_nonnative_X_i__y %>%
  group_by(contrast,correct) %>%
  summarise(count=n()) %>%
  mutate(perc=count/sum(count))

biphon_nonnative_acc__yih__i <- biphon_nonnative_X_yih__i %>%
  group_by(contrast,correct) %>%
  summarise(count=n()) %>%
  mutate(perc=count/sum(count))

biphon_nonnative_acc__i__yih <- biphon_nonnative_X_i__yih %>%
  group_by(contrast,correct) %>%
  summarise(count=n()) %>%
  mutate(perc=count/sum(count))

biphon_nonnative_acc__y__u <- biphon_nonnative_X_y__u %>%
  group_by(contrast,correct) %>%
  summarise(count=n()) %>%
  mutate(perc=count/sum(count))

biphon_nonnative_acc__u__y <- biphon_nonnative_X_u__y %>%
  group_by(contrast,correct) %>%
  summarise(count=n()) %>%
  mutate(perc=count/sum(count))

biphon_nonnative_acc_perc <- rbind(biphon_nonnative_acc__ob__u, biphon_nonnative_acc__u__ob, biphon_nonnative_acc__uu__u, biphon_nonnative_acc__u__uu, biphon_nonnative_acc__y__i, biphon_nonnative_acc__i__y,
                                   biphon_nonnative_acc__yih__i, biphon_nonnative_acc__i__yih, biphon_nonnative_acc__y__u, biphon_nonnative_acc__u__y)


#' Build the full model 
# mdl = lmer(dSPM ~ vowel_id*pos1*pos2*type*trial_order + (1|subject), data=biphon_native_X)
# mdl2 = glmer(correct ~ vowel_id*pos1*pos2*type*trial_order + (1|subject), data=biphon_native_X, family = binomial)
# mdl3 = lmer(dSPM ~ vowel_id*pos1*pos2*type*trial_order + (1|subject), data=biphon_nonnative_X)
# mdl4 = glmer(correct ~ vowel_id*pos1*pos2*type*trial_order + (1|subject), data=biphon_nonnative_X, family = binomial)
# 
# ## test some things out
# # mdl = lmer(dSPM ~ vowel_id + trial_order + (1|trial_order) + (1|subject), data=biphon_native_X)
# mdl2 = glmer(correct ~ vowel_iso + type + trial_order + (1 + trial_order|subject) + (1|vowel_iso), data=biphon_native_X, family = binomial)
# mdl3 = lmer(LogRT ~ vowel_iso + type + trial_order + (1 + trial_order|subject) + (1|vowel_iso), data=biphon_native_X)
# 
# # mdl4 = lmer(dSPM ~ vowel_id + trial_order + (1|trial_order) + (1|subject), data=biphon_nonnative_X)
# mdl5 = glmer(correct ~ vowel_iso + type + trial_order + (1 + trial_order|subject) + (1|vowel_iso), data=biphon_nonnative_X, family = binomial)
# mdl6 = lmer(LogRT ~ vowel_iso + type + trial_order + (1 + trial_order|subject) + (1|vowel_iso), data=biphon_nonnative_X)


## bodo multiple cat variables, view just the estimates and how much they vary from the intercept, not mixed effects, just linear model
# mdl3 = lmer(dSPM ~ vowel_id + (1|subject), data=biphon_nonnative_X)
# tempmdl <- lm(dSPM ~ vowel_id, data=biphon_nonnative_A)
# tidy(tempmdl) %>% select(term:estimate) %>% mutate(estimate = round(estimate, 2))
# bph_non_A_preds <- tibble(vowel_id = sort(unique(biphon_nonnative_A$vowel_id)))
# bph_non_A_preds$fit <- round(predict(tempmdl,bph_non_A_preds),2)
# bph_non_A_preds
# 
# tempmdl <- lm(dSPM ~ vowel_id, data=biphon_nonnative_B)
# tidy(tempmdl) %>% select(term:estimate) %>% mutate(estimate = round(estimate, 2))
# bph_non_B_preds <- tibble(vowel_id = sort(unique(biphon_nonnative_B$vowel_id)))
# bph_non_B_preds$fit <- round(predict(tempmdl,bph_non_B_preds),2)
# bph_non_B_preds
# 
# tempmdl <- lm(dSPM ~ vowel_id, data=biphon_nonnative_X)
# tidy(tempmdl) %>% select(term:estimate) %>% mutate(estimate = round(estimate, 2))
# bph_non_X_preds <- tibble(vowel_id = sort(unique(biphon_nonnative_X$vowel_id)))
# bph_non_X_preds$fit <- round(predict(tempmdl,bph_non_X_preds),2)
# bph_non_X_preds

### Trial counts

biphon_nb_trial_counts <- biphon_nb %>%
  group_by(subject,dialect) %>%
  summarise(count=n())
 


####### NON-NATIVE PLOTTING
# boxplots
#dspm
# biphon_nonnative_A %>% ggplot(aes(x = reorder(vowel_iso, dSPM, FUN=median), y = dSPM, fill = vowel_iso)) + stat_boxplot(geom ='errorbar') + geom_boxplot() + theme_minimal() + ggtitle('dSPM distribution: A tokens') +
#   labs(title="dSPM distribution per A token type", subtitle = 'Non-native Trials', y="dSPM", x="A Tokens", fill='A Tokens')
# biphon_nonnative_B %>% ggplot(aes(x = reorder(vowel_iso, dSPM, FUN=median), y = dSPM, fill = vowel_iso)) + stat_boxplot(geom ='errorbar') + geom_boxplot() + theme_minimal() + ggtitle('dSPM distribution: B tokens') +
#   labs(title="dSPM distribution per B token type", subtitle = 'Non-native Trials', y="dSPM", x="B Tokens", fill='B Tokens')
# biphon_nonnative_X %>% ggplot(aes(x = reorder(vowel_iso, dSPM, FUN=median), y = dSPM, fill = vowel_iso)) + stat_boxplot(geom ='errorbar') + geom_boxplot() + theme_minimal() + ggtitle('dSPM distribution: X tokens') +
#   labs(title="dSPM distribution per X token type", subtitle = 'Non-native Trials', y="dSPM", x="X Tokens", fill='X Tokens')

#rt boxplot
#biphon_nonnative_X %>% ggplot(aes(x = reorder(vowel_iso,LogRT, FUN=median), y = LogRT, fill = vowel_iso)) + stat_boxplot(geom ='errorbar') + geom_boxplot() + theme_minimal() +
#  labs(title="LogRT distribution per X token type", y="LogRT", x="X Tokens")

#rt violin
biphon_nonnative_X %>% ggplot(aes(x = reorder(vowel_iso, LogRT, FUN=median), y = LogRT, fill = vowel_iso)) + 
  geom_violin() + geom_boxplot(width=0.1) + theme_minimal() + 
  labs(title="LogRT distribution per X token type", subtitle = 'Non-native Trials', y="LogRT", x="X Tokens") + 
  scale_fill_discrete(name = "X Tokens")

# accuracy scores
#overall, not split by contrasts
plot <- ggplot(biphon_nonnative_acc, aes(x=biphon_nonnative_acc$vowel_iso, y=perc*100, fill=correct))
plot + geom_bar(stat='identity', width = 0.7) + labs(title="Raw Accuracy on Non-native Trials", subtitle="All Token Contrasts Included", y="Percent Correct", x="X Tokens", fill='Correct')

# split for error rate vs accuracy
biphon_nonnative_acc_perc_incorrect <- subset(biphon_nonnative_acc_perc, correct == 0)
biphon_nonnative_acc_perc_correct <- subset(biphon_nonnative_acc_perc, correct == 1)

# error rate for specific contrasts
plot <- ggplot(biphon_nonnative_acc_perc_incorrect, aes(x=reorder(biphon_nonnative_acc_perc_incorrect$contrast, -biphon_nonnative_acc_perc_incorrect$perc), y=perc*100, fill=correct))
plot + geom_bar(stat='identity', width = 0.7, fill='salmon') + geom_text(aes(label=round(biphon_nonnative_acc_perc_incorrect$perc*100, digits=2)), vjust=1.6, color="white",position = position_dodge(0.9), size=4) +
  geom_hline(yintercept=50, linetype='dashed') +
  theme(axis.text.x = element_text(size=12)) +
  labs(title="Error Rate for Specific Non-native Contrasts", y="Error Rate (%)", x="X Token Subsets", fill='Correct')

# accuracy for specific contrasts
plot <- ggplot(biphon_nonnative_acc_perc_correct, aes(x=reorder(biphon_nonnative_acc_perc_correct$contrast, biphon_nonnative_acc_perc_correct$perc), y=perc*100, fill=correct))
plot + geom_bar(stat='identity', width = 0.7, fill='steelblue') + geom_text(aes(label=round(biphon_nonnative_acc_perc_correct$perc*100, digits=2)), vjust=1.6, color="white",position = position_dodge(0.9), size=4) +
  geom_hline(yintercept=50, linetype='dashed') +
  theme(axis.text.x = element_text(size=12)) +
  labs(title="Accuracy (%) for Specific Non-native Contrasts", y="Accuracy (%)", x="X Token Subsets", fill='Correct')

#non-native vowel groupings, vowel group 3 is 'ah and 'ae' native baselines
vowel_grp2 <- c('{u-ø}ø','{u-ø}u','{u-ɯ}ɯ','{u-ɯ}u')
vowel_grp1 <- c('{i-y}y','{i-y}i','{i-ʏ}ʏ','{i-ʏ}i')
vowel_grp4 <- c('{u-y}y','{u-y}u','{i-y}y','{i-y}i')

# new dfs based on vowel groupings for error rate and accuracy
biphon_nonnative_acc_perc_incorrect_vowel_grp1 <- subset(biphon_nonnative_acc_perc_incorrect, contrast == '{i-y}y' | contrast == '{i-y}i' | contrast == '{i-ʏ}ʏ' | contrast == '{i-ʏ}i')
biphon_nonnative_acc_perc_incorrect_vowel_grp2 <- subset(biphon_nonnative_acc_perc_incorrect, contrast == '{u-ø}ø' | contrast == '{u-ø}u' | contrast == '{u-ɯ}ɯ' | contrast == '{u-ɯ}u')
biphon_nonnative_acc_perc_incorrect_vowel_grp4 <- subset(biphon_nonnative_acc_perc_incorrect, contrast == '{i-y}y' | contrast == '{i-y}i' | contrast == '{u-y}y' | contrast == '{u-y}u')
biphon_nonnative_acc_perc_incorrect_vowel_grp5 <- subset(biphon_nonnative_acc_perc_incorrect, contrast == '{i-y}y' | contrast == '{i-y}i' | contrast == '{u-y}y' | contrast == '{u-y}u' | contrast == '{i-ʏ}ʏ' | contrast == '{i-ʏ}i')


# error rate of vowel group 1
plot <- ggplot(biphon_nonnative_acc_perc_incorrect_vowel_grp1, aes(x=reorder(biphon_nonnative_acc_perc_incorrect_vowel_grp1$contrast, -biphon_nonnative_acc_perc_incorrect_vowel_grp1$perc), y=perc*100, fill=correct))
plot + geom_bar(stat='identity', width = 0.7, fill='#fd8d3c') + geom_text(aes(label=round(biphon_nonnative_acc_perc_incorrect_vowel_grp1$perc*100, digits=2)), vjust=1.6, color="white",position = position_dodge(0.9), size=10) +
  geom_hline(yintercept=50, linetype='dashed') +
  theme(axis.text.x = element_text(size=40)) +
  theme(text = element_text(size = 40)) +
  labs(title="Prediction for /ʏ/ and /y/: Error Rate with English /i/", y="Error Rate (%)", x="X Token Subsets", fill='Correct')

# error rate of vowel group 2
plot <- ggplot(biphon_nonnative_acc_perc_incorrect_vowel_grp2, aes(x=reorder(biphon_nonnative_acc_perc_incorrect_vowel_grp2$contrast, -biphon_nonnative_acc_perc_incorrect_vowel_grp2$perc), y=perc*100, fill=correct))
plot + geom_bar(stat='identity', width = 0.7, fill='#66c2a4') + geom_text(aes(label=round(biphon_nonnative_acc_perc_incorrect_vowel_grp2$perc*100, digits=2)), vjust=1.6, color="white",position = position_dodge(0.9), size=10) +
  geom_hline(yintercept=50, linetype='dashed') +
  theme(axis.text.x = element_text(size=40)) +
  theme(text = element_text(size = 40)) +
  labs(title="Prediction for /ø/ and /ɯ/: Error Rate with English /u/", y="Error Rate (%)", x="X Token Subsets", fill='Correct')

# error rate of vowel group 3
plot <- ggplot(biphon_nonnative_acc_perc_incorrect_vowel_grp4, aes(x=reorder(biphon_nonnative_acc_perc_incorrect_vowel_grp4$contrast, -biphon_nonnative_acc_perc_incorrect_vowel_grp4$perc), y=perc*100, fill=correct))
plot + geom_bar(stat='identity', width = 0.7, fill='salmon') + geom_text(aes(label=round(biphon_nonnative_acc_perc_incorrect_vowel_grp4$perc*100, digits=2)), vjust=1.6, color="white",position = position_dodge(0.9), size=4) +
  geom_hline(yintercept=50, linetype='dashed') +
  theme(axis.text.x = element_text(size=12)) +
  labs(title="Validation of /y-u/ Relationship: Error Rate for Specific Non-native Contrasts", y="Error Rate (%)", x="X Token Subsets", fill='Correct')

# error rate of yih, u, i, y
plot <- ggplot(biphon_nonnative_acc_perc_incorrect_vowel_grp5, aes(x=reorder(biphon_nonnative_acc_perc_incorrect_vowel_grp5$contrast, -biphon_nonnative_acc_perc_incorrect_vowel_grp5$perc), y=perc*100, fill=correct))
plot + geom_bar(stat='identity', width = 0.7, fill='salmon') + geom_text(aes(label=round(biphon_nonnative_acc_perc_incorrect_vowel_grp5$perc*100, digits=2)), vjust=1.6, color="white",position = position_dodge(0.9), size=4) +
  geom_hline(yintercept=50, linetype='dashed') +
  theme(axis.text.x = element_text(size=12)) +
  labs(title="/y-u-yih-i/ Relationship: Error Rate for Specific Non-native Contrasts", y="Error Rate (%)", x="X Token Subsets", fill='Correct')


# plot <- ggplot(biphon_nonnative_acc__ob__u, aes(x=biphon_nonnative_acc__ob__u$contrast, y=perc*100, fill=correct))
# plot + geom_bar(stat='identity', width = 0.7) + labs(title="Raw Accuracy on Non-native Trials", subtitle="All Token Contrasts Included", y="Percent Correct", x="X Tokens", fill='Correct')
# 
# plot <- ggplot(biphon_nonnative_acc__u__ob, aes(x=biphon_nonnative_acc__u__ob$contrast, y=perc*100, fill=correct))
# plot + geom_bar(stat='identity', width = 0.7) + labs(title="Raw Accuracy on Non-native Trials", subtitle="All Token Contrasts Included", y="Percent Correct", x="X Tokens", fill='Correct')
# 
# plot <- ggplot(biphon_nonnative_acc__uu__u, aes(x=biphon_nonnative_acc__uu__u$contrast, y=perc*100, fill=correct))
# plot + geom_bar(stat='identity', width = 0.7) + labs(title="Raw Accuracy on Non-native Trials", subtitle="All Token Contrasts Included", y="Percent Correct", x="X Tokens", fill='Correct')
# 
# plot <- ggplot(biphon_nonnative_acc__u__uu, aes(x=biphon_nonnative_acc__u__uu$contrast, y=perc*100, fill=correct))
# plot + geom_bar(stat='identity', width = 0.7) + labs(title="Raw Accuracy on Non-native Trials", subtitle="All Token Contrasts Included", y="Percent Correct", x="X Tokens", fill='Correct')
# 
# plot <- ggplot(biphon_nonnative_acc__y__i, aes(x=biphon_nonnative_acc__y__i$contrast, y=perc*100, fill=correct))
# plot + geom_bar(stat='identity', width = 0.7) + labs(title="Raw Accuracy on Non-native Trials", subtitle="All Token Contrasts Included", y="Percent Correct", x="X Tokens", fill='Correct')
# 
# plot <- ggplot(biphon_nonnative_acc__i__y, aes(x=biphon_nonnative_acc__i__y$contrast, y=perc*100, fill=correct))
# plot + geom_bar(stat='identity', width = 0.7) + labs(title="Raw Accuracy on Non-native Trials", subtitle="All Token Contrasts Included", y="Percent Correct", x="X Tokens", fill='Correct')
# 
# plot <- ggplot(biphon_nonnative_acc__yih__i, aes(x=biphon_nonnative_acc__yih__i$contrast, y=perc*100, fill=correct))
# plot + geom_bar(stat='identity', width = 0.7) + labs(title="Raw Accuracy on Non-native Trials", subtitle="All Token Contrasts Included", y="Percent Correct", x="X Tokens", fill='Correct')
# 
# plot <- ggplot(biphon_nonnative_acc__i__yih, aes(x=biphon_nonnative_acc__i__yih$contrast, y=perc*100, fill=correct))
# plot + geom_bar(stat='identity', width = 0.7) + labs(title="Raw Accuracy on Non-native Trials", subtitle="All Token Contrasts Included", y="Percent Correct", x="X Tokens", fill='Correct')
# 
# plot <- ggplot(biphon_nonnative_acc__y__u, aes(x=biphon_nonnative_acc__y__u$contrast, y=perc*100, fill=correct))
# plot + geom_bar(stat='identity', width = 0.7) + labs(title="Raw Accuracy on Non-native Trials", subtitle="All Token Contrasts Included", y="Percent Correct", x="X Tokens", fill='Correct')
# 
# plot <- ggplot(biphon_nonnative_acc__u__y, aes(x=biphon_nonnative_acc__u__y$contrast, y=perc*100, fill=correct))
# plot + geom_bar(stat='identity', width = 0.7) + labs(title="Raw Accuracy on Non-native Trials", subtitle="All Token Contrasts Included", y="Percent Correct", x="X Tokens", fill='Correct')


####### NATIVE PLOTTING
#dspm
biphon_native_A %>% ggplot(aes(x = reorder(vowel_iso, dSPM, FUN=median), y = dSPM, fill = vowel_iso)) + stat_boxplot(geom ='errorbar') + geom_boxplot() + theme_minimal() + ggtitle('dSPM distribution: A tokens')
biphon_native_B %>% ggplot(aes(x = reorder(vowel_iso, dSPM, FUN=median), y = dSPM, fill = vowel_iso)) + stat_boxplot(geom ='errorbar') + geom_boxplot() + theme_minimal() + ggtitle('dSPM distribution: B tokens')
biphon_native_X %>% ggplot(aes(x = reorder(vowel_iso, dSPM, FUN=median), y = dSPM, fill = vowel_iso)) + stat_boxplot(geom ='errorbar') + geom_boxplot() + theme_minimal() + ggtitle('dSPM distribution: X tokens')

#rt
# biphon_native_X %>% ggplot(aes(x = reorder(vowel_id, LogRT, FUN=median), y = LogRT, fill = vowel_id)) + stat_boxplot(geom ='errorbar') + geom_boxplot() + theme_minimal() + ggtitle('LogRT distribution per token type')

#rt violin
biphon_native_X %>% ggplot(aes(x = reorder(vowel_iso, LogRT, FUN=median), y = LogRT, fill = vowel_iso)) + 
  geom_violin() + geom_boxplot(width=0.1) + theme_minimal() + 
  labs(title="LogRT distribution per X token type", subtitle = 'Native Trials', y="LogRT", x="X Tokens") + 
  scale_fill_discrete(name = "X Tokens")

# accuracy scores
plot <- ggplot(biphon_native_acc, aes(x=biphon_native_acc$vowel_iso, y=perc*100, fill=correct))
plot + geom_bar(stat='identity', width = 0.7) + labs(title="Raw Accuracy on Non-native Trials", subtitle="All Token Contrasts Included", y="Percent Correct", x="X Tokens", fill='Correct')

### DENSITY CHECKS
#density
biphon_nonnative_X %>% ggplot(aes(x = dSPM, fill = vowel_id)) + geom_density(alpha = 0.5)

#' Model Summary
summary(mdl)

#' Anova to get p values of the model
anova(mdl)


#' backward elimination of non-significant effects: Showint elimination tables for random- and fixed-effect terms:
stp = step(mdl)
stp

#' Extract the model that step found:
final_model <- get_model(stp)
final_model

plot(final_model)

posthoc <- difflsmeans(final_model)
posthoc

plot(ls_means(final_model))

