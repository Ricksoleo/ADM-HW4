ADM - HOMEWORK 4 
 GROUP 18 
 
NihaL Yaman Yılmaz - nihalyaman20@gmail.com  
Riccardo Soleo - Soleo.1911063@studenti.uniroma1.it  
Shekh Sadamhusen  - sadamhusen06120@gmail.com



⚠️ **Attention:** Before running the code, make sure to place the **`minhash_utils1.py`** file in the `archive` folder. Otherwise, the code will not work properly. 



# MovieLens Recommendation and Clustering System

This project focuses on analyzing the MovieLens dataset to develop a recommendation system and clustering algorithms for personalized movie suggestions. Additionally, a strategic game algorithm has been implemented to predict the outcomes of competitive number sequences.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preparation](#data-preparation)
3. [Recommendation System with LSH](#recommendation-system-with-lsh)
   - Minhash Signatures
   - Locality-Sensitive Hashing (LSH)
4. [Movie Clustering](#movie-clustering)
   - Feature Engineering
   - Dimensionality Reduction
   - Clustering Algorithms
5. [Algorithm Comparison](#algorithm-comparison)
6. [Bonus: K-means Iteration Visualization](#bonus-k-means-iteration-visualization)
7. [Algorithmic Game Question](#algorithmic-game-question)
   - Optimal Playing Strategy
8. [Conclusion](#conclusion)
9. [How to Run](#how-to-run)

---

## Introduction

The MovieLens dataset is utilized to build a movie recommendation system and explore clustering algorithms. The project aims to:
1. Improve recommendation efficiency through **Locality-Sensitive Hashing (LSH)**.
2. Group movies into clusters using **K-means** and **K-means++**, implemented from scratch.
3. Solve an algorithmic challenge for optimal decision-making in a game scenario.

---

## Data Preparation

The dataset is preprocessed to:
- Normalize features using `StandardScaler` for effective clustering.
- Create new features like `ratings_avg`, `relevant_genome_tag`, and `common_user_tag`.
- Reduce dimensionality using **Principal Component Analysis (PCA)** to retain 95% of variance.

---

## Recommendation System with LSH

### Minhash Signatures
- **Custom MinHash Algorithm**: 
  - Implemented from scratch to hash users' movie preferences.
  - Experimented with different hash functions and thresholds to find the optimal configuration.
  
### Locality-Sensitive Hashing (LSH)
- **Bucket Creation**:
  - Divided MinHash signatures into bands and grouped similar users into buckets.
- **Recommendation Logic**:
  - Recommends movies based on user similarity:
    - Common movies with average ratings.
    - Top-rated movies of the most similar users.

---

## Movie Clustering

### Feature Engineering
- Created features for clustering:
  - Genres (one-hot encoded)
  - Average ratings
  - Release year
  - Tags (relevant and common)
  - Additional features: `genre_count`, `ratings_count`, `popularity_ratio`.

### Dimensionality Reduction
- Applied PCA to reduce dimensions, capturing 95% of variance.
- Visualized explained variance with a Scree Plot.

### Clustering Algorithms
1. **K-means (MapReduce)**:
   - Implemented from scratch using MapReduce logic.
2. **K-means++**:
   - Improved initialization for faster convergence.
3. **Third Algorithm**:
   - **DBSCAN**: Suggested by LLM as a density-based clustering algorithm.

---

## Algorithm Comparison

- Evaluated clustering quality using:
  1. **Silhouette Score**: Measures intra-cluster cohesion and inter-cluster separation.
  2. **Davies-Bouldin Index**: Measures compactness and separation of clusters.
  3. **Inertia**: Measures within-cluster variance.

| Algorithm  | Silhouette Score | Davies-Bouldin Index | Inertia |
|------------|------------------|-----------------------|---------|
| K-means    | 0.68             | 0.81                 | 4500    |
| K-means++  | 0.70             | 0.78                 | 4300    |
| DBSCAN     | 0.72             | 0.75                 | N/A     |

---

## Bonus: K-means Iteration Visualization

Visualized the progression of cluster formations over 5 iterations, using the first two PCA components. The visualizations demonstrated:
- How centroids converge.
- How data points are assigned to clusters over iterations.

---

## Algorithmic Game Question

### Problem
Arya and Mario compete to maximize their scores by taking numbers from either end of an array. Arya wants to know if she can guarantee victory.

### Solution
1. **Recursive Algorithm**:
   - Used recursion to evaluate all possible moves.
2. **Dynamic Programming Optimization**:
   - Improved efficiency by storing intermediate results.
3. **LLM-Optimized Algorithm**:
   - Implemented a further optimized version suggested by ChatGPT.

| Input        | Output | Explanation                                    |
|--------------|--------|------------------------------------------------|
| `[1, 5, 2]`  | False  | Mario wins with perfect play.                  |
| `[1, 5, 233, 7]` | True   | Arya wins by strategically choosing numbers. |

---

## Conclusion

This project highlights the effectiveness of:
1. Custom MinHash and LSH for recommendation systems.
2. Dimensionality reduction techniques like PCA for clustering.
3. K-means and alternative clustering algorithms like DBSCAN for movie segmentation.

The game problem demonstrates the importance of optimizing algorithms for real-world applications.

---

