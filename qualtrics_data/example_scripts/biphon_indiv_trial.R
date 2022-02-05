# biphon indiv trial script
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

#split by speakers
biphon_nonnative_X_ob_1_u <- subset(biphon_nonnative_X_ob__u, pos3 == 'ob1')
biphon_nonnative_X_ob_2_u <- subset(biphon_nonnative_X_ob__u, pos3 == 'ob2')

biphon_nonnative_X_u_1_ob <- subset(biphon_nonnative_X_u__ob, pos3 == 'u1')
biphon_nonnative_X_u_2_ob <- subset(biphon_nonnative_X_u__ob, pos3 == 'u2')

biphon_nonnative_X_uu_1_u <- subset(biphon_nonnative_X_uu__u, pos3 == 'uu1')
biphon_nonnative_X_uu_2_u <- subset(biphon_nonnative_X_uu__u, pos3 == 'uu2')

biphon_nonnative_X_u_1_uu <- subset(biphon_nonnative_X_u__uu, pos3 == 'u1')
biphon_nonnative_X_u_2_uu <- subset(biphon_nonnative_X_u__uu, pos3 == 'u2')

biphon_nonnative_X_y_1_i <- subset(biphon_nonnative_X_y__i, pos3 == 'y1')
biphon_nonnative_X_y_2_i <- subset(biphon_nonnative_X_y__i, pos3 == 'y2')

biphon_nonnative_X_i_1_y <- subset(biphon_nonnative_X_i__y, pos3 == 'i1')
biphon_nonnative_X_i_2_y <- subset(biphon_nonnative_X_i__y, pos3 == 'i2')

biphon_nonnative_X_yih_1_i <- subset(biphon_nonnative_X_yih__i, pos3 == 'yih1')
biphon_nonnative_X_yih_2_i <- subset(biphon_nonnative_X_yih__i, pos3 == 'yih2')

biphon_nonnative_X_i_1_yih <- subset(biphon_nonnative_X_i__yih, pos3 == 'i1')
biphon_nonnative_X_i_2_yih <- subset(biphon_nonnative_X_i__yih, pos3 == 'i2')

biphon_nonnative_X_y_1_u <- subset(biphon_nonnative_X_y__u, pos3 == 'y1')
biphon_nonnative_X_y_2_u <- subset(biphon_nonnative_X_y__u, pos3 == 'y2')

biphon_nonnative_X_u_1_y <- subset(biphon_nonnative_X_u__y, pos3 == 'u1')
biphon_nonnative_X_u_2_y <- subset(biphon_nonnative_X_u__y, pos3 == 'u2')


#add in contrast column
biphon_nonnative_X_ob_1_u$contrast <- c('{u-ø}ø1')
biphon_nonnative_X_ob_2_u$contrast <- c('{u-ø}ø2')

biphon_nonnative_X_u_1_ob$contrast <- c('{u-ø}u1')
biphon_nonnative_X_u_2_ob$contrast <- c('{u-ø}u2')

biphon_nonnative_X_uu_1_u$contrast <- c('{u-ɯ}ɯ1')
biphon_nonnative_X_uu_2_u$contrast <- c('{u-ɯ}ɯ2')

biphon_nonnative_X_u_1_uu$contrast <- c('{u-ɯ}u1')
biphon_nonnative_X_u_2_uu$contrast <- c('{u-ɯ}u2')

biphon_nonnative_X_y_1_i$contrast  <- c('{i-y}y1')
biphon_nonnative_X_y_2_i$contrast  <- c('{i-y}y2')

biphon_nonnative_X_i_1_y$contrast <- c('{i-y}i1')
biphon_nonnative_X_i_2_y$contrast  <- c('{i-y}i2')

biphon_nonnative_X_yih_1_i$contrast <- c('{i-ʏ}ʏ1')
biphon_nonnative_X_yih_2_i$contrast  <- c('{i-ʏ}ʏ2')

biphon_nonnative_X_i_1_yih$contrast <- c('{i-ʏ}i1')
biphon_nonnative_X_i_2_yih$contrast <- c('{i-ʏ}i2')

biphon_nonnative_X_y_1_u$contrast <- c('{u-y}y1')
biphon_nonnative_X_y_2_u$contrast <- c('{u-y}y2')

biphon_nonnative_X_u_1_y$contrast <- c('{u-y}u1')
biphon_nonnative_X_u_2_y$contrast <- c('{u-y}u2')


df.list <- list(biphon_nonnative_X_ob_1_u, biphon_nonnative_X_ob_2_u, biphon_nonnative_X_u_1_ob, biphon_nonnative_X_u_2_ob, biphon_nonnative_X_uu_1_u, biphon_nonnative_X_uu_2_u,
                biphon_nonnative_X_u_1_uu, biphon_nonnative_X_u_2_uu, biphon_nonnative_X_y_1_i, biphon_nonnative_X_y_2_i, biphon_nonnative_X_i_1_y, biphon_nonnative_X_i_2_y,
                biphon_nonnative_X_yih_1_i, biphon_nonnative_X_yih_2_i, biphon_nonnative_X_i_1_yih, biphon_nonnative_X_i_2_yih, biphon_nonnative_X_y_1_u, biphon_nonnative_X_y_2_u,
                biphon_nonnative_X_u_1_y, biphon_nonnative_X_u_2_y)
res <- lapply(df.list, function(x) {acc <- x %>% group_by(contrast,correct) %>% summarise(count=n()) %>% mutate(perc=count/sum(count))})
biphon_nonnative_acc_perc <- do.call(rbind.data.frame, res)

# split for error rate vs accuracy
biphon_nonnative_acc_perc_incorrect <- subset(biphon_nonnative_acc_perc, correct == 0)
biphon_nonnative_acc_perc_correct <- subset(biphon_nonnative_acc_perc, correct == 1)

bar_colors = c("blue",
               "blue",
               "#87CEFA",
               "#87CEFA",
               "#006400",
               "#006400",
               "green",
               "green",
               "#2F4F4F",
               "#2F4F4F",
               "#2F4F4F",
               "#C71585",
               "#C71585",
               "purple",
               "purple",
               "#fd8d3c",
               "#fd8d3c",
               "black",
               "black")

biphon_nonnative_acc_perc_incorrect$colors <- bar_colors

# error rate
plot <- ggplot(biphon_nonnative_acc_perc_incorrect, aes(x=reorder(biphon_nonnative_acc_perc_incorrect$contrast, -biphon_nonnative_acc_perc_incorrect$perc), y=perc*100, fill=correct))
plot + geom_bar(stat='identity', width = 0.7, fill=biphon_nonnative_acc_perc_incorrect$colors) + 
      geom_text(aes(label=round(biphon_nonnative_acc_perc_incorrect$perc*100, digits=2)), vjust=1.6, color="white",position = position_dodge(0.9), size=4) +
      geom_hline(yintercept=50, linetype='dashed') +
      theme(axis.text.x = element_text(size=12)) +
      theme(text = element_text(size = 12)) +
      labs(title="Error Rate, Specific Contrasts", y="Error Rate (%)", x="X Token Subsets", fill='Correct')

