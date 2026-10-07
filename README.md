# Spoken English Grammar Scoring

A multimodal machine-learning solution for predicting continuous spoken-English grammar proficiency scores from audio recordings.

## Approach

The final system combines complementary speech and language signals:

- **WavLM Layer 9** speech embeddings with RBF SVR
- **HuBERT** speech embeddings with RBF SVR
- **Grammar-aware RoBERTa/CoLA embeddings** from Whisper transcripts
- **Handcrafted acoustic features** with SVR + Extra Trees
- **Audio-text cross-attention model** trained on frozen WavLM and grammar-aware sentence representations

The strongest submitted model blended the multimodal expert ensemble with the cross-attention model.

## Validation

Five-fold stratified cross-validation was used with score bands to preserve target distribution across folds. Scaling and other trainable preprocessing were fit inside each training fold.

Selected OOF results:

| Model | OOF RMSE | Pearson |
|---|---:|---:|
| WavLM Layer 9 | ~0.566 | ~0.890 |
| HuBERT | ~0.602 | ~0.874 |
| Grammar embedding | ~0.815 | ~0.753 |
| Cross-modal model | ~0.552 | ~0.896 |
| Multimodal ensemble | ~0.540 | ~0.910 |
| Final V13 blend | ~0.518 | ~0.914 |

The best public leaderboard score was **0.3603**.

## Repository Contents

- `SHL_Grammar_Scoring_Final.ipynb` — final methodology notebook
- `src/audio_features.py` — handcrafted acoustic feature extraction template
- `src/modeling_notes.py` — compact reference for the final ensemble architecture
- `docs/methodology.md` — methodology summary
- `requirements.txt` — Python dependencies

## Data Policy

Competition audio, CSV files, generated embeddings, transcripts, model checkpoints, and submission files are intentionally excluded from this public repository.

## Reproducibility

Pretrained encoders used in the experiments include WavLM, HuBERT, Whisper, and a CoLA-trained RoBERTa model. The notebook documents the training and evaluation procedure while avoiding redistribution of competition-protected data.
