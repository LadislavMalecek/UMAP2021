# Fairness-preserving Group Recommendations With User Weighting

This repository contains the source code to reproduce the results in our paper "Fairness-preserving Group Recommendations With User Weighting", which was submitted to UMAP 2021 LBR. The repository also contains results of additional group recommendation scenarios.

Source codes in this repository are based on the work of Mesut Kaya, Derek Bridge and Nava Tintarev "Ensuring Fairness in Group Recommendations by Rank-Sensitive Balancing of Relevance" available from https://github.com/mesutkaya/recsys2020

## Authors
Ladislav Malecek

Ladislav Peska

## Abstract
Group recommendations are an extension of "single-user" personalized recommender systems (RS), where the final recommendations should comply with preferences of several group members. 
An important challenge in group RS is the problem of fairness, i.e., no user's preferences should be largely ignored by the RS. Traditional strategies, such as "least misery" or "average rating", tackle the problem of fairness, but they resolve it separately for each item. This may cause a systematic bias against some group members.

In contrast, this paper considers both fairness and relevance as a rank-sensitive list property. We propose EP-FuzzDA algorithm that utilizes an optimization criterion encapsulating both fairness and relevance. In conducted experiments, EP-FuzzDA outperforms several state-of-the-art baselines. 
Another advantage of EP-FuzzDA is the capability to adjust on non-uniform importance of group members enabling e.g. to maintain the long-term fairness across several recommending sessions.

## Repository content
- data folder contains evaluation datasets (MovieLens 1M and KGRec-Music)
- java folder contains implementation of EP-FuzzDA algorithm as well as baseline methods and underlying ALS MF recommending algorithm
- python folder contains data pre-processing and results evaluation scripts
- Results folder contains results of all evaluated scenarios (i.e. MovieLens 1M and KGRec-Music datasets; divergent, similar and random composition of artificial groups; group sizes 2,3,4 and 8; evaluations with uniform user weights, weighted users and long-term fairness).

## Running evaluation
