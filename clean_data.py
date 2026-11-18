import pandas as pd
import numpy as np

# 1. Load raw Apify dataset
df = pd.read_csv('Dataset.csv')

# 2. Collapse hashtags (0..9) and latest comments (0..11) into compact strings
hashtag_cols = [f'hashtags/{i}' for i in range(10) if f'hashtags/{i}' in df.columns]
comment_cols = [f'latestComments/{i}/text' for i in range(12) if f'latestComments/{i}/text' in df.columns]

df['hashtags'] = df[hashtag_cols].apply(lambda r: ', '.join(r.dropna().astype(str)), axis=1)
df['hashtag_count'] = df[hashtag_cols].notna().sum(axis=1)
df['top_comments'] = df[comment_cols].apply(lambda r: ' | '.join(r.dropna().astype(str)), axis=1)

# 3. Handle anomalies (-1 likes mean hidden likes on Instagram)
df['likes_hidden'] = df['likesCount'] == -1
df['likes'] = df['likesCount'].replace(-1, np.nan)
df['views'] = df['videoPlayCount']
df['comments'] = df['commentsCount']

# 4. Compute Normalized Sector Metrics
df['channel_median_views'] = df.groupby('ownerUsername')['views'].transform('median')
df['outlier_index'] = (df['views'] / df['channel_median_views']).round(2)

df['performance_tier'] = pd.cut(
    df['outlier_index'],
    bins=[0, 0.5, 1.2, 3.0, np.inf],
    labels=['Underperformer', 'Baseline', 'Above_Average', 'Viral_Outlier']
)

# Discussion Rate (Comments per 1,000 views)
df['discussion_rate_per_1k'] = ((df['comments'] / df['views']) * 1000).round(2)
df['like_rate_pct'] = ((df['likes'] / df['views']) * 100).round(2)

# 5. Temporal & Format Features (Convert UTC to IST)
dt_ist = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Asia/Kolkata')
df['post_date'] = dt_ist.dt.date
df['post_day'] = dt_ist.dt.day_name()
df['post_hour_ist'] = dt_ist.dt.hour

df['duration_bucket'] = pd.cut(
    df['videoDuration'],
    bins=[0, 20, 50, 90, np.inf],
    labels=['Micro (<20s)', 'Standard (20-50s)', 'Deep Explainer (50-90s)', 'Long (>90s)']
)

df['social_proof_bias_gap'] = np.nan  

# 6. Select clean schema
keep_cols = [
    'shortCode', 'url', 'ownerUsername', 'post_date', 'post_day', 'post_hour_ist',
    'views', 'likes', 'likes_hidden', 'comments', 'channel_median_views',
    'outlier_index', 'performance_tier', 'discussion_rate_per_1k', 'like_rate_pct',
    'videoDuration', 'duration_bucket', 'caption', 'hashtags', 'hashtag_count',
    'top_comments', 'social_proof_bias_gap', 'musicInfo/song_name', 'isPinned'
]

clean_df = df[keep_cols].copy()
clean_df.to_csv('cleaned_2d_sector_dataset.csv', index=False)
print("Clean dataset created successfully!")