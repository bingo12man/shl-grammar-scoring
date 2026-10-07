# Methodology Summary

## Problem
Predict a continuous grammar-proficiency score in the range 0–5 from spoken-English recordings.

## Pipeline
1. Resample/load speech at 16 kHz.
2. Generate transcripts with Whisper.
3. Extract WavLM and HuBERT speech embeddings.
4. Extract grammar-aware transcript embeddings with a CoLA-trained RoBERTa model.
5. Extract handcrafted acoustic descriptors.
6. Train individual regressors using five-fold stratified cross-validation.
7. Train a lightweight audio-text cross-attention model using frozen representations.
8. Blend complementary experts based on out-of-fold performance and error correlation.

## Final Selected Architecture
The strongest public submission used 70% of the V11 multimodal ensemble and 30% of the V12 cross-attention prediction.

### V11
- 50% WavLM Layer 9
- 20% HuBERT
- 20% grammar embedding
- 10% handcrafted acoustic ensemble

### V13
- 70% V11
- 30% V12 cross-attention

## Evaluation
Five-fold stratified CV with RMSE and Pearson correlation. Final V13 OOF performance was approximately RMSE 0.518 and Pearson 0.914. Best public leaderboard score: 0.3603.

## Public Repository Safety
Competition recordings, labels, test metadata, generated features, transcripts, checkpoints, and prediction CSVs are intentionally not included.
