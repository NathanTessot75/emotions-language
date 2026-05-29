# How does language organize emotions?

This repository contains my final project for linguistics. The goal is to see if the way we use emotion words in everyday language reflects how we experience them psychologically.

To do this, I compare two things:
1. **Psychological data**: VAD scores (Valence, Arousal, Dominance) from Warriner et al. (2013).
2. **Linguistic data**: Word2Vec embeddings trained on Google News (Mikolov et al., 2013).

My main hypothesis is that Valence (the positive/negative aspect) is the main factor that drives how emotions are structured in language.

## Requirements
You'll need a few standard libraries to run the code:
- `pandas`, `numpy`
- `seaborn`, `matplotlib`, `plotly` (for the graphs)
- `gensim` (to load Word2Vec)
- `scipy`, `statsmodels` (for the statistical tests)

**Note**: The first time you run it, the code will download the Word2Vec model. It's about 1.7 GB, so it might take a few minutes depending on your internet connection. After that, it's saved locally so the next runs will be much faster.
