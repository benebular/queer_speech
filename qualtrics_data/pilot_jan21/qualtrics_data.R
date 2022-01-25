library("dplyr")
library("tidyr")
library("lme4")
library("magrittr")
library("ggplot2")
library("plotly")

BIPHON <- read.csv("/Volumes/MEG/NYUAD-Lab-Server/Personal Files/Ben/sounds/BIPHON_test0.csv")
df_sub <- select(BIPHON, -contains("Click"))
df_sub2 <- select(BIPHON, -contains("Submit"))
df_sub3 <- df_sub2[-c(14,15,16),]
df_sub4 <- select(df_sub3, -c(ResponseID,Q0,NQ1,NQ2,NQ3,NQ4,NQ5,NQ6,NQ7,NQ8,Q23,NQ10,NQ10_5_TEXT,NQ11,NQ12,NQ13,S1_1))
df_sub6 <- select(df_sub6, -contains("Duration"))
df_sub6 <- select(df_sub6, -contains("Response"))
df_sub7 <- df_sub6[,-c(1)]
write.csv(df_sub7, "/Users/ben/")
BIPHON_subset <- select(BIPHON_subset, -contains("Q2"))
BIPHON_subset <- select(BIPHON_subset, -contains("NQ"))
BIPHON_subset <- select(BIPHON_subset, -contains("S2"))
BIPHON_subset <- select(BIPHON_subset, -contains("S3"))
BIPHON_subset <- select(BIPHON_subset, -contains("reactiontime"))
BIPHON_subset <- select(BIPHON_subset, -contains("PROLIFIC"))
within(biphon_cleaned_data, sub1 <- factor(sub1, labels = c(1, 2)))
p <- ggplot(data=biphon_vowels_cleaned, aes(x=unique_id,y=percent_correct,color=Nonnative)) + facet_grid(rows=vars(Nonnative)) + geom_text(aes(label=unique_id), angle=45) + scale_y_continuous(breaks=c(0,1,2,3,4,5,6,7,8,9,10,11,12)) + labs(title="Percent Correct per Trial of 8 Participants") + labs(x='Trial') + labs(y='Percent Correct') + labs(colour='Nativeness') + theme(axis.text.x=element_blank(),axis.ticks.x = element_blank())
plotly_build(p)

euc.dist <- function(x1, x2) sqrt(sum((x1 - x2) ^ 2))
biphon_norm <- biphon_vowels %>%
  +     select(F1_A, F2_A, F3_A,F1_B, F2_B, F3_B,F1_X...12, F2_X...13, F3_X...14)