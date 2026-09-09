"""
Step 3: Sub-cluster the mega-cluster (C2, 45.5%) and do deeper manual
inspection to finalize the intent taxonomy.
"""
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
import sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

df = pd.read_csv("data/amazon_help_clustered.csv")
print(f"Total: {len(df)}")

# ── Filter to English only (exclude French/Spanish/German clusters) ──────
# Remove clusters 3 (French), 4 (Spanish), 5 (German), 6 (spam)
df_en = df[~df['cluster'].isin([3, 4, 5, 6])].copy()
print(f"English-only: {len(df_en)}")

# ── Sub-cluster C2 (the mega-cluster) ────────────────────────────────────
c2 = df_en[df_en['cluster'] == 2].copy().reset_index(drop=True)
print(f"\nSub-clustering C2: {len(c2)} tweets")

vectorizer = TfidfVectorizer(
    max_features=3000, min_df=2, max_df=0.4,
    ngram_range=(1, 2), stop_words='english', sublinear_tf=True,
)
X = vectorizer.fit_transform(c2['clean_text'])
svd = TruncatedSVD(n_components=30, random_state=42)
X_red = svd.fit_transform(X)

K_SUB = 10
km = MiniBatchKMeans(n_clusters=K_SUB, random_state=42, batch_size=300, n_init=20)
c2['sub_cluster'] = km.fit_predict(X_red)

feature_names = np.array(vectorizer.get_feature_names_out())

for sc in range(K_SUB):
    idx = np.where(c2['sub_cluster'].values == sc)[0]
    pct = len(idx) / len(c2) * 100
    
    cluster_mean = X[idx].mean(axis=0).A1
    top_ids = cluster_mean.argsort()[-10:][::-1]
    terms = [feature_names[i] for i in top_ids]
    
    centroid = km.cluster_centers_[sc]
    dists = np.linalg.norm(X_red[idx] - centroid, axis=1)
    closest = np.argsort(dists)[:6]
    
    print(f"\n--- C2-SUB{sc} --- ({len(idx)}, {pct:.1f}%)")
    print(f"  Keywords: {', '.join(terms)}")
    for ci in closest:
        tweet = str(c2.iloc[idx[ci]]['text'])[:160]
        tweet = tweet.encode('ascii', 'replace').decode('ascii')
        print(f"    - {tweet}")

# ── Manual keyword-based analysis across ALL English tweets ──────────────
print("\n" + "="*80)
print("KEYWORD-BASED INTENT SIGNAL ANALYSIS")
print("="*80)

patterns = {
    'refund/money_back': r'refund|money back|reimburse|charge|charged|overcharg',
    'delivery_late': r'late|delay|delayed|hasn.t arrived|not arrived|still waiting|where is|when will|not received|not delivered',
    'delivery_wrong': r'wrong (address|item|product|house)|delivered to|wrong place|someone else',
    'return/exchange': r'return|exchange|send back|replace|replacement|swap',
    'cancel': r'cancel|cancell',
    'prime_membership': r'prime (member|subscription|trial|free)|membership|subscribe|renew',
    'account_issue': r'account (lock|block|suspend|hack|disable|issue)|password|sign.in|log.?in|locked out',
    'product_defective': r'defective|broken|damaged|doesn.t work|not working|faulty|malfunction|dead on arrival',
    'tracking': r'track|tracking|shipment status|where.s my (order|package|parcel)',
    'kindle/device': r'kindle|fire (tablet|stick|tv|hd)|echo|alexa|device',
    'prime_video': r'prime video|streaming|watch|movie|show|series|content',
    'pricing': r'price|pricing|discount|coupon|promo|deal|offer|overpriced',
    'wrong_item': r'wrong item|wrong product|wrong size|wrong color|not what i ordered|different (item|product)',
    'missing_item': r'missing|not in (box|package)|empty box|partial|incomplete',
    'customer_service_complaint': r'worst|terrible|horrible|awful|rude|useless|pathetic|disgusting|incompetent|disgrace',
}

for name, pattern in patterns.items():
    matches = df_en[df_en['text'].str.contains(pattern, case=False, na=False, regex=True)]
    pct = len(matches) / len(df_en) * 100
    print(f"\n{name}: {len(matches)} tweets ({pct:.1f}%)")
    sample = matches.sample(min(3, len(matches)), random_state=42)
    for _, row in sample.iterrows():
        tweet = str(row['text'])[:160].encode('ascii', 'replace').decode('ascii')
        print(f"    - {tweet}")
