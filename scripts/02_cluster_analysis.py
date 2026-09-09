"""
Step 2: Derive intent taxonomy from AmazonHelp customer tweets using
TF-IDF + K-Means clustering, followed by manual inspection.
"""
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
import sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── 1. Load sample ──────────────────────────────────────────────────────
df = pd.read_csv("data/amazon_help_sample.csv")
print(f"Loaded {len(df)} tweets")

# ── 2. Clean text ────────────────────────────────────────────────────────
def clean_tweet(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s!?.,\'-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

df['clean_text'] = df['text'].apply(clean_tweet)
df = df[df['clean_text'].str.len() > 10].reset_index(drop=True)
print(f"After cleaning: {len(df)} tweets")

# ── 3. TF-IDF ────────────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    max_features=5000, min_df=3, max_df=0.5,
    ngram_range=(1, 2), stop_words='english', sublinear_tf=True,
)
X = vectorizer.fit_transform(df['clean_text'])
print(f"TF-IDF matrix: {X.shape}")

# ── 4. SVD ────────────────────────────────────────────────────────────────
svd = TruncatedSVD(n_components=50, random_state=42)
X_reduced = svd.fit_transform(X)
print(f"Explained variance (50 components): {svd.explained_variance_ratio_.sum():.2%}")

# ── 5. Evaluate K ────────────────────────────────────────────────────────
print("\n--- Evaluating cluster counts ---")
for k in [8, 10, 12, 14, 16]:
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=500, n_init=10)
    labels = km.fit_predict(X_reduced)
    sil = silhouette_score(X_reduced, labels, sample_size=2000, random_state=42)
    print(f"  K={k:2d}  silhouette={sil:.4f}")

# ── 6. Final clustering K=12 ─────────────────────────────────────────────
K = 12
print(f"\n{'='*80}")
print(f"Final clustering with K={K}")
print(f"{'='*80}")

km = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=500, n_init=20)
labels = km.fit_predict(X_reduced)
df['cluster'] = labels

# ── 7. Analyze each cluster ──────────────────────────────────────────────
feature_names = np.array(vectorizer.get_feature_names_out())

for c in range(K):
    idx = np.where(labels == c)[0]
    pct = len(idx) / len(df) * 100

    print(f"\n--- CLUSTER {c} --- ({len(idx)} tweets, {pct:.1f}%)")

    # Top TF-IDF terms (use numpy indexing on sparse matrix)
    cluster_mean = X[idx].mean(axis=0).A1
    top_ids = cluster_mean.argsort()[-15:][::-1]
    top_terms = [feature_names[i] for i in top_ids]
    print(f"  Top terms: {', '.join(top_terms[:12])}")

    bigrams = [t for t in top_terms if ' ' in t][:5]
    if bigrams:
        print(f"  Top bigrams: {', '.join(bigrams)}")

    # 5 representative tweets (closest to centroid)
    centroid = km.cluster_centers_[c]
    dists = np.linalg.norm(X_reduced[idx] - centroid, axis=1)
    closest = np.argsort(dists)[:5]

    print(f"  Representative tweets:")
    for i, ci in enumerate(closest):
        tweet = str(df.iloc[idx[ci]]['text'])[:180]
        tweet = tweet.encode('ascii', 'replace').decode('ascii')
        print(f"    {i+1}. {tweet}")

# ── 8. Distribution ──────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("Cluster size distribution:")
for c in range(K):
    n = (labels == c).sum()
    pct = n / len(df) * 100
    bar = '#' * int(pct * 2)
    print(f"  Cluster {c:2d}: {n:4d} ({pct:5.1f}%) {bar}")

# ── 9. Save ──────────────────────────────────────────────────────────────
df.to_csv("data/amazon_help_clustered.csv", index=False)
print(f"\nSaved to data/amazon_help_clustered.csv")

# ── 10. Per-cluster keyword export for taxonomy design ────────────────────
print(f"\n{'='*80}")
print("KEYWORD SUMMARY FOR TAXONOMY DESIGN")
print(f"{'='*80}")
for c in range(K):
    idx = np.where(labels == c)[0]
    cluster_mean = X[idx].mean(axis=0).A1
    top_ids = cluster_mean.argsort()[-8:][::-1]
    terms = [feature_names[i] for i in top_ids]
    pct = len(idx) / len(df) * 100
    
    # Show 3 more diverse tweets (random from cluster)
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(idx), min(8, len(idx)), replace=False)
    
    print(f"\nC{c} ({len(idx)}, {pct:.1f}%) keywords: {', '.join(terms)}")
    for si in sample_idx[:8]:
        tweet = str(df.iloc[idx[si]]['text'])[:160]
        tweet = tweet.encode('ascii', 'replace').decode('ascii')
        print(f"   - {tweet}")
