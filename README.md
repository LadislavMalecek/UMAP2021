# Fairness-preserving Group Recommendations With User Weighting

This repository contains the source code to reproduce the results in our paper "Fairness-preserving Group Recommendations With User Weighting", which was submitted to UMAP 2021 LBR. The repository also contains results of additional group recommendation scenarios.

Source codes in this repository are based on the work of Mesut Kaya, Derek Bridge and Nava Tintarev "Ensuring Fairness in Group Recommendations by Rank-Sensitive Balancing of Relevance" available from https://github.com/mesutkaya/recsys2020

## Authors
Ladislav Malecek

Ladislav Peska

## Repository content
- data folder contains evaluation datasets (MovieLens 1M and KGRec-Music)
- java folder contains implementation of EP-FuzzDA algorithm as well as baseline methods and underlying ALS MF recommending algorithm
- python folder contains data pre-processing and results evaluation scripts
- Results folder contains results of all evaluated scenarios (i.e. MovieLens 1M and KGRec-Music datasets; divergent, similar and random composition of artificial groups; group sizes 2,3,4 and 8; evaluations with uniform user weights, weighted users and long-term fairness).

## Running evaluation
