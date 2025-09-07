import pandas as pd

def clean_data(df: pd.DataFrame):
    """Clean the raw YouTube dataset."""
    # Drop duplicates & missing values
    df = df.drop_duplicates()
    df = df.dropna(subset=['views', 'likes', 'dislikes', 'comment_count'])
    
    # Ensure numeric columns
    df['views'] = pd.to_numeric(df['views'], errors='coerce')
    df['likes'] = pd.to_numeric(df['likes'], errors='coerce')
    df['dislikes'] = pd.to_numeric(df['dislikes'], errors='coerce')
    df['comment_count'] = pd.to_numeric(df['comment_count'], errors='coerce')

    df = df.dropna(subset=['views', 'likes', 'dislikes', 'comment_count'])
    
    return df


def feature_engineering(df: pd.DataFrame):
    """Add new features for analysis & keep 'views' for target variable."""
    # Extract publish hour (if publish_time column exists)
    if 'publish_time' in df.columns:
        df['publish_hour'] = pd.to_datetime(df['publish_time'], errors='coerce').dt.hour.fillna(0).astype(int)
    else:
        df['publish_hour'] = 0
    
    # Title length
    if 'title' in df.columns:
        df['title_length'] = df['title'].astype(str).apply(len)
    else:
        df['title_length'] = 0
    
    # Select features + target
    df = df[['publish_hour', 'title_length', 'likes', 'dislikes', 'comment_count', 'views']]
    
    return df
