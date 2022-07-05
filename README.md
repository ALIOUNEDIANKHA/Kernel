# Kernel
A kernel Project to submit in Kaggle
## Introduction

The goal of the data challenge is to learn how to implement machine learning algorithms, gain understanding about them and adapt them to structural data.
For this reason, we have chosen a sequence classification task: predicting whether a DNA sequence (or read) belongs to the SARS-CoV-2 (Covid-19).

## Data description

Both the training and evaluation data are sets of DNA sequencing reads: short DNA fragments (~100 to 300bp long), that come from sequencing experiments, or that were simulated from full genomes. Some of these fragments come from Covid-19 genomes, others from human or random bacteria. 
The goal is to discriminate the Covid-19 fragments, hence the task is a binary classification task: the labels are either 1 if the fragment is identified as Covid-19, and 0 otherwise.
