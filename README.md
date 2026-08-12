# EEG-based schizophrenia classification

This capstone project evaluates resting-state EEG using patient-level validation.

## Overview

This capstone project brings together a complete ML workflow. It uses a small clinical dataset of resting-state EEG. The goal is not a perfect model, but a careful, leakage-aware methodology.

## Dataset

The dataset includes 28 subjects (14 patients and 14 healthy controls). The recordings capture an eyes-closed resting state. They use 19 channels, 250 Hz, and a standard 10-20 montage. The raw data are publicly available, but they are not included in this repo.

## Method

First, I filtered the signals (1–40 Hz). Then I split the data into 4-second epochs. I extracted band power features for different frequency bands. I assigned the label at the person level. Finally, I used a patient-level split (GroupKFold) to avoid data leakage.

## Results

An epoch-level split inflates the accuracy. This means that the model memorizes the person, not the disorder. A person-level split solves this.

Gradient boosting slightly outperformed random forest.

The model is biased toward schizophrenia. Schizophrenia recall was high (84%), while control recall was lower (59%). As a result, it labels many healthy people as patients.

Using class_weight did not help. Even though I applied it, it slightly hurt the performance. This suggests that the imbalance comes from the nature of the classes, not their size.

## Honesty & Limitations

This study has a small sample with only 28 subjects. Because of this, the results are likely unstable. I also tested inter-channel synchronization, but it did not help. This feature is probably too coarse to capture the complex differences. Ultimately, the point is not a perfect model, but an honest methodology.

