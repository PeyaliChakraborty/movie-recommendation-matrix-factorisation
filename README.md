# Movie Recommendation System using Matrix Factorisation

A recommendation system built from scratch using Matrix Factorisation and Gradient Descent.

## What it does

Takes a user-movie ratings matrix with missing entries and predicts what rating a user would give to a movie they haven't watched yet — the same idea behind Netflix and Spotify recommendations.

## Approach

- Decompose the ratings matrix **R** into two latent factor matrices **P** (users) and **Q** (movies) such that R ≈ P · Qᵀ
- Minimise reconstruction error using Stochastic Gradient Descent with L2 regularisation
- Predict missing ratings from the learned latent factors
- Compute user-user similarity in latent space using cosine similarity

## Contents

| File | Description |
|---|---|
| `movie_recommender.ipynb` | Full implementation with visualisations |

## Visualisations

- Ratings matrix heatmap (original vs reconstructed)
- Gradient descent loss curve
- Top-3 recommendations per user
- User similarity heatmap in latent space

## Libraries used

`numpy` `matplotlib` `seaborn`
