## Queer Speech Vowel Plots
## Author: Ben Lang, blang@ucsd.edu

### importing libraries and data
library(tidyr)
library(lme4)
library(magrittr)
library(dplyr)
library(ggplot2)
library(wesanderson)
# library(ggpubr) # this plots normality

setwd('/Users/bcl/Documents/GitHub/queer_speech/feature_extraction')
vowel_data = read.csv('vs_output.csv')


## data manipulation
# vowel_data$Vowel <- vowel_data$vowel_iso
vowel_data$Vowel <- vowel_data$vowel_id_spk

## grab all vowel rows, group_by speaker, average F1, average F2, then average across all speakers
vowel_data %>% subset(vowel_data$Label == "IY") %>% select(sF1) %>% na.omit() %>% summarise(mean(sF1))
vowel_data %>% subset(vowel_data$Label == "IY") %>% select(sF2) %>% na.omit() %>% summarise(mean(sF2))
vowel_data %>% subset(vowel_data$Label == "UW") %>% select(sF1) %>% na.omit() %>% summarise(mean(sF1))
vowel_data %>% subset(vowel_data$Label == "UW") %>% select(sF2) %>% na.omit() %>% summarise(mean(sF2))
vowel_data %>% subset(vowel_data$Label == "AA") %>% select(sF1) %>% na.omit() %>% summarise(mean(sF1))
vowel_data %>% subset(vowel_data$Label == "AA") %>% select(sF2) %>% na.omit() %>% summarise(mean(sF2))
vowel_data %>% subset(vowel_data$Label == "AE") %>% select(sF1) %>% na.omit() %>% summarise(mean(sF1))
vowel_data %>% subset(vowel_data$Label == "AE") %>% select(sF2) %>% na.omit() %>% summarise(mean(sF2))

## plotting
pal <- wes_palette("Royal1", 16, type="continuous")
pal2 <- wes_palette("FantasticFox1", 16, type="continuous")
pal3 <- wes_palette("GrandBudapest2", 16, type="continuous")
pal4 <- wes_palette("Zissou1", 16, type="continuous")
pal5 <- wes_palette("Darjeeling1", 32, type="continuous")

# just vowels
ggplot(data=vowel_data, aes(x=F2, y=F1)) +
  geom_text(aes(label=Vowel, color=Vowel), size=20) + 
  scale_color_manual(values=pal5) +
  scale_x_reverse(position = "top") + 
  scale_y_reverse(position="right") +
  theme(legend.position = 'None') +
  theme(text = element_text(size = 40)) +
  theme(plot.background = element_rect(fill = "#905cb1")) +
  theme(panel.background = element_rect(fill = "white"), panel.grid.major = element_line(size = 0.5, linetype = 'solid', colour = "grey"), panel.grid.minor = element_line(size = 0.25, linetype = 'solid', colour = "grey")) +
  theme(axis.text.x=element_text(colour="white")) +
  theme(axis.text.y=element_text(colour="white")) +
  theme(axis.title.x = element_text(colour = "white"),
        axis.title.y = element_text(colour = "white")) +
  labs(x='F2', y='F1')
