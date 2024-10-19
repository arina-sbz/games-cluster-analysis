import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors


# Show PCA plot based on first three components
def plot_PCA(components):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        components[:, 0],
        components[:, 1],
        components[:, 2],
        c=components[:, 2],
        cmap="plasma",
        s=60,
        alpha=0.7,
        edgecolors="k",
    )

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label("PC 3 Value", fontsize=12)

    # Set labels and title
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_zlabel("PC3", fontsize=12)
    ax.set_title(
        "3D PCA Plot of Steam Games Data (First Three Components)", fontsize=11
    )

    # Adjust viewing angle for better perspective
    ax.view_init(elev=25, azim=40)  # Change angles as needed

    plt.gcf().subplots_adjust(left=0.45)
    plt.savefig("pca", dpi=300, bbox_inches="tight")

    # Show the plot
    plt.show()


# Plot clustering result
def plot_dbscan(pca_components, labels):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot for the DBSCAN clustering
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    for k, color in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise (outliers)
            color = [0, 0, 0, 1]

        class_member_mask = labels == k

        # Plot the clustered points
        xyz = pca_components[class_member_mask]
        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=[color],
            s=60,
            alpha=0.7,
            edgecolors="k",
            label=f"Cluster {k}" if k != -1 else "Noise",
        )

    # Set labels and title
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_zlabel("PC3", fontsize=12)
    ax.set_title("3D DBSCAN Clustering Plot", fontsize=11)

    # Adjust viewing angle for better perspective
    ax.view_init(elev=25, azim=40)  # Change angles as needed

    plt.gcf().subplots_adjust(left=0.45)
    ax.legend(loc="best")
    plt.savefig("dbscan", dpi=300, bbox_inches="tight")
    # Show the plot
    plt.show()


# Show the elbow plot
def elbow_plot(pca_components, k):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(pca_components)
    distances, _ = neigh.kneighbors(pca_components)
    distances = np.sort(distances[:, k - 1])
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel("Points")
    plt.ylabel(f"{k}-th nearest neighbor distance")
    plt.title("K-distance Graph")
    plt.show()
    return distances


def plot_unappropriate_entries():
    labels = "Unappropriate entries", "Utilities", "Appropriate entries"
    sizes = np.array([(85102 - 77626), 77626 - 72361, 77626]) / 85102
    explode = (0.1, 0.1, 0)

    fig, ax = plt.subplots()
    ax.set_title(
        "Unappropriate entries, appropriate entries and utilities distribution"
    )
    ax.pie(sizes, explode=explode, labels=labels, autopct="%1.1f%%", shadow=True)


def plot_price_range(df):
    df["Price Range"] = pd.cut(
        df["Price"],
        bins=[0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 1000],
        labels=[
            "0-25",
            "26-75",
            "76-100",
            "101-125",
            "126-150",
            "151-175",
            "176-200",
            "201-225",
            "226-250",
            "251-275",
            "276-300",
            "301-325",
            "326+",
        ],
    )

    # Count the number of games in each price range
    price_range_counts = df["Price Range"].value_counts().sort_index()

    plt.figure(figsize=(12, 8))

    # Plot the distribution of games across price ranges
    ax = sns.barplot(x=price_range_counts.index, y=price_range_counts.values)

    # Add annotations to each bar
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
        )

    plt.title("Number of Games in Each Price Range")
    plt.xlabel("Price Range")
    plt.ylabel("Number of Games")
    plt.savefig("pricerange", dpi=300, bbox_inches="tight")
    plt.show()


def plot_price_scatter(df, title):
    df.plot(
        kind="scatter",
        x="Price",
        y="Average playtime forever",
        title=title,
    )
    plt.savefig("price_scatter", dpi=300, bbox_inches="tight")
    plt.show()


def plot_os_support(df):
    # Summarize the number of games that support each platform
    platform_support = df[["Windows", "Mac", "Linux"]].apply(
        lambda x: x.value_counts().get(True, 0)
    )

    # Create a bar plot with the number annotated on top of each bar
    ax = platform_support.plot(
        kind="bar",
        title="Comparison of Game Support for Windows, Mac, and Linux",
        ylabel="Number of Games Supported",
    )
    plt.xticks(rotation=0)

    # Annotate the bars with the values
    for i, v in enumerate(platform_support):
        ax.text(i, v + 0.1, str(v), ha="center", va="bottom")

    plt.savefig("os_support.png", dpi=300, bbox_inches="tight")

    plt.show()


def plot_os_no_support(df):
    # Count the number of `False` values in each platform column
    platform_no_support = df[["Windows", "Mac", "Linux"]].apply(
        lambda x: (x == False).sum()
    )

    # Create a bar plot with the number of games that do not support each platform
    ax = platform_no_support.plot(
        kind="bar",
        title="Games Without Support for Windows, Mac, and Linux",
        ylabel="Number of Games Not Supported",
    )
    plt.xticks(rotation=0)

    # Annotate the bars with the count of `False` values
    for i, v in enumerate(platform_no_support):
        ax.text(i, v + 0.1, str(v), ha="center", va="bottom")

    plt.savefig("noossuuport", dpi=300, bbox_inches="tight")
    plt.show()


# Plot correlation heatmap
def plot_corr(df):
    # Calculate the correlation matrix
    corr = df.corr()
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        mask=mask,
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title("Correlation Heatmap of Key Variables", fontsize=16)
    plt.savefig("heatmap", dpi=300, bbox_inches="tight")
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()


def plot_genres(df):
    # Split the genres into separate rows (explode)
    df_genres = df.assign(Genres=df["Genres"].str.split(",")).explode("Genres")

    # Strip any leading/trailing spaces
    df_genres["Genres"] = df_genres["Genres"].str.strip()

    # Get the count of each genre
    genre_counts = df_genres["Genres"].value_counts()

    # Plot the top 10 most common genres
    top_genres = genre_counts.head(10)
    sns.barplot(x=top_genres.values, y=top_genres.index)
    plt.title("Top 10 Most Common Genres")
    plt.xlabel("Number of Games")
    plt.ylabel("Genres")
    plt.savefig("genres.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_genre_price(df):
    df_genres = df.assign(Genres=df["Genres"].str.split(",")).explode("Genres")
    genres_to_plot = [
        "Indie",
        "Action",
        "Casual",
        "Adventure",
        "Strategy",
        "Simulation",
        "RPG",
        "Early Access",
        "Free to Play",
        "Sports",
    ]

    # Group by Genres and calculate the average price for each genre
    avg_price_per_genre = (
        df_genres.groupby("Genres")["Price"].mean().sort_values(ascending=False)
    )

    # Filter for the specified genres
    filtered_avg_price = avg_price_per_genre[
        avg_price_per_genre.index.isin(genres_to_plot)
    ]

    # Plot the top 10 genres with the highest average price
    sns.barplot(x=filtered_avg_price.values, y=filtered_avg_price.index)
    # sns.barplot(x=avg_price_per_genre.head(20).values, y=avg_price_per_genre.head(20).index)
    plt.title("Average Price per Genre")
    plt.xlabel("Average Price")
    plt.ylabel("Genres")
    plt.savefig("genre_price.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_pie_game_type(df):
    offline_games = df[df["online_offline"] == 0].shape[0]
    online_games = df[df["online_offline"] == 1].shape[0]
    total_games = df.shape[0]

    labels = (
        "Online games",
        "Offline games",
    )
    sizes = np.array([(online_games), (offline_games)]) / total_games
    explode = (0, 0)

    fig, ax = plt.subplots()
    ax.set_title("Distribution of Online and Offline Games")
    ax.pie(sizes, explode=explode, labels=labels, autopct="%1.1f%%", shadow=True)
    plt.savefig("onlineoffline", dpi=300, bbox_inches="tight")
    plt.show()


def plot_pie_reviews(df):
    # Calculate the sum of positive and negative reviews
    reviews_sum = df[["Positive_original", "Negative_original"]].sum()
    reviews_sum.plot(
        kind="pie",
        autopct="%1.1f%%",
        startangle=90,
        labels=["Positive Reviews", "Negative Reviews"],
        colors=["lightgreen", "lightcoral"],
        title="Positive vs Negative Reviews in Cluster 2",
    )

    plt.ylabel("")
    plt.savefig("reviewscluser2", dpi=300, bbox_inches="tight")
    plt.show()


def plot_year(df):
    # Count the number of games per release year
    release_year_counts = df["release_year"].value_counts().sort_index()

    release_year_counts.plot(kind="barh", title="Distribution of Release Year")

    plt.xlabel("Number of Games")
    plt.ylabel("Release Year")
    plt.savefig("yearsclus2", dpi=300, bbox_inches="tight")
    plt.show()


def plot_publishers(df):
    publishers_series = df["Publishers"].str.split(",").explode()

    # Get the top 10 publishers by count
    top_publishers = publishers_series.value_counts().head(10)

    top_publishers.plot(kind="bar", title="Top 10 Publishers")

    plt.xlabel("Publishers")
    plt.ylabel("Number of Games")
    plt.savefig("publishers", dpi=300, bbox_inches="tight")
    plt.show()
