import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"🔹 MSE: {mse:.2f}")
    print(f"🔹 R²: {r2:.2f}")

def plot_visualizations(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Top Channels
    top_channels = df['channel_title'].value_counts().nlargest(10)
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_channels.values, y=top_channels.index)
    plt.title("Top 10 Channels with Most Trending Videos")
    plt.savefig(os.path.join(output_dir, "top_channels.png"))
    plt.close()

    # Top Tags
    df['tags'] = df['tags'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
    tags = df['tags'].explode().value_counts().nlargest(10)
    plt.figure(figsize=(10,6))
    sns.barplot(x=tags.values, y=tags.index)
    plt.title("Top 10 Tags in Trending Videos")
    plt.savefig(os.path.join(output_dir, "top_tags.png"))
    plt.close()

    # Views vs Likes
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=df['likes'], y=df['views'], alpha=0.5)
    plt.title("Views vs Likes")
    plt.savefig(os.path.join(output_dir, "views_vs_likes.png"))
    plt.close()

    # Publish Time Distribution
    plt.figure(figsize=(10,6))
    sns.histplot(df['publish_time'].dt.hour, bins=24, kde=False)
    plt.title("Publish Time Distribution (by Hour)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Videos")
    plt.savefig(os.path.join(output_dir, "publish_time_distribution.png"))
    plt.close()
