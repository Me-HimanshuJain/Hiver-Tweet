"""
Step 1 (v3): Download Customer Support on Twitter, filter to AmazonHelp, sample.
"""
import kagglehub
import pandas as pd
import os, sys, io

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── 1. Download ──────────────────────────────────────────────────────────
print("Downloading dataset...")
path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
print(f"Path: {path}")

csv_file = os.path.join(path, "twcs", "twcs.csv")
if not os.path.exists(csv_file):
    csv_file = os.path.join(path, "twcs.csv")

# ── 2. Load ──────────────────────────────────────────────────────────────
df = pd.read_csv(csv_file)
print(f"Total rows: {len(df):,}")

# ── 3. Filter inbound tweets mentioning @AmazonHelp ──────────────────────
inbound = df[df['inbound'] == True].copy()
print(f"Total inbound: {len(inbound):,}")

amazon_mentions = inbound[inbound['text'].str.contains('@AmazonHelp', case=False, na=False)]
print(f"Inbound mentioning @AmazonHelp: {len(amazon_mentions):,}")

# ── 4. Remove follow-up tweets (replies to AmazonHelp outbound) ─────────
amazon_outbound = df[(df['author_id'] == 'AmazonHelp') & (df['inbound'] == False)]
amazon_out_ids = set(amazon_outbound['tweet_id'].values)
print(f"AmazonHelp outbound tweets: {len(amazon_out_ids):,}")

# Mark tweets that are replies to AmazonHelp (follow-ups)
has_resp = amazon_mentions.dropna(subset=['in_response_to_tweet_id'])
has_resp_int = has_resp['in_response_to_tweet_id'].astype(int)
follow_up_ids = set(has_resp[has_resp_int.isin(amazon_out_ids)]['tweet_id'].values)
print(f"Follow-up tweets (replying to AmazonHelp): {len(follow_up_ids):,}")

initial_contact = amazon_mentions[~amazon_mentions['tweet_id'].isin(follow_up_ids)]
print(f"Initial contact tweets: {len(initial_contact):,}")

# ── 5. Sample ────────────────────────────────────────────────────────────
SAMPLE_SIZE = 3000
if len(initial_contact) > SAMPLE_SIZE:
    sample = initial_contact.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"Sampled {SAMPLE_SIZE} from {len(initial_contact):,}")
else:
    sample = initial_contact.copy()
    print(f"Using all {len(sample)} tweets")

# ── 6. Save ──────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
sample.to_csv("data/amazon_help_sample.csv", index=False)
print(f"Saved {len(sample)} tweets to data/amazon_help_sample.csv")

# ── 7. Stats ─────────────────────────────────────────────────────────────
print(f"\nAvg text length: {sample['text'].str.len().mean():.0f} chars")
print(f"Avg word count: {sample['text'].str.split().str.len().mean():.1f} words")

print(f"\n--- 15 random tweets ---")
for _, row in sample.sample(15, random_state=7).iterrows():
    text = str(row['text'])[:150]
    # Sanitize for console
    text = text.encode('ascii', 'replace').decode('ascii')
    print(f"  [{row['tweet_id']}] {text}")
