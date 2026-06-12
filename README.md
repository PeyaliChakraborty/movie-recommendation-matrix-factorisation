# Movie Recommendation System using Matrix Factorisation

A recommendation system built from scratch using Matrix Factorisation and Gradient Descent — no sklearn, no surprise library, just NumPy.

## Why I built this

I'd always wondered how Spotify or Amazon figures out what I'll like before I've even tried it. Turns out the core idea is surprisingly elegant: you decompose a ratings matrix into latent factors and let the model figure out what those factors mean. I wanted to understand this from first principles, so I implemented it myself.

## What it does

Takes a user-movie ratings matrix with missing entries and predicts what rating a user would give to a movie they haven't watched yet — the same idea behind Netflix and Spotify recommendations.

## Approach

- Decompose the ratings matrix **R** into two latent factor matrices **P** (users) and **Q** (movies) such that R ≈ P · Qᵀ
- Minimise reconstruction error using Stochastic Gradient Descent with L2 regularisation
- Predict missing ratings from the learned latent factors
- Compute user-user similarity in latent space using cosine similarity

## Results

- 6×6 ratings matrix: 6 users, 6 movies (Bollywood + Hollywood mix), 25 observed ratings, 11 missing
- k=3 latent factors, α=0.005, λ=0.02, 3000 epochs
- Final RMSE on observed ratings: **0.06** — the model reconstructs known ratings near-perfectly
- Loss converged within ~500 epochs; ran 3000 to confirm stability
- User-user cosine similarity in latent space picks up intuitive taste clusters

## What I found interesting

k=3 was small enough to avoid overfitting on a 6×6 matrix but enough to capture meaningful structure. The latent dimensions don't have labels — the model figures out what matters on its own, which is the part I find genuinely interesting about this approach.

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
