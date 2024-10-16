import pandas as pd
import numpy as np
import re
from sklearn import preprocessing as pp
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D


# List of words to remove
inappropriate_words = ["sex", "sexual content", "nudity", "hentai", "nsfw"]

# List of columns to check
columns_to_check = ["Genres", "Categories", "Tags", "Notes", "About the game"]


def get_set_of_all_genres(df: pd.DataFrame):
    genres = []
    for genre in df["Genres"].astype(str).unique().tolist():
        genres.extend(genre.split(","))
    return set(genres)


# by this point, the df most be out of missing values
# Removing outliers
def remove_outliers(
    df: pd.DataFrame,
    id_df: pd.DataFrame,
    n_neighbors=20,
):
    df_numeric = df[
        [
            "Peak CCU",
            "Required age",
            "Price",
            "DLC count",
            "Windows",
            "Mac",
            "Linux",
            "Metacritic score",
            "User score",
            "Positive",
            "Negative",
            "Recommendations",
            "Average playtime forever",
            "Average playtime two weeks",
            "Median playtime forever",
            "Median playtime two weeks",
        ]
    ]

    # broad outliers detection
    clf = LocalOutlierFactor(n_neighbors=n_neighbors)
    df_filtered = clf.fit_predict(df_numeric)
    df = df[df_filtered != -1]
    id_df = id_df[df_filtered != -1]

    all_genres = get_set_of_all_genres(df)

    # for checking if a game has all genres
    def func(row):
        return set(str(row["Genres"]).split(",")) == all_genres

    # specific outliers detection
    # remove any game that happens to have all the possible genres
    df_filtered = df.apply(func, axis=1)
    df = df[~df_filtered]
    id_df = id_df[~df_filtered]
    return df, id_df


# Function to filter out rows containing any exact word in the specified columns
def contains_inappropriate_word(row):
    # Create a regex pattern that matches the exact words using word boundaries (\b)
    pattern = (
        r"\b(?:" + "|".join(re.escape(word) for word in inappropriate_words) + r")\b"
    )

    # Check if any word in the pattern exists in the row's specified columns
    return any(
        re.search(pattern, str(row[col]), re.IGNORECASE) for col in columns_to_check
    )


# Function to remove rows that we do not consider a game
def remove_non_games(df: pd.DataFrame) -> pd.DataFrame:
    # Remove any row where Genres or Tags field contains the string Utilities
    df = df[~df["Genres"].str.contains("Utilities")]
    df = df[~df["Tags"].str.contains("Utilities")]
    df = df[~df["Categories"].str.contains("Utilities")]

    # When running get_set_of_all_genres on only entries without singlepplayer/multiplayer in categories we will keep only the ones that are actually games
    actual_game_genres = ["Action", "Adventure", "RPG", "Racing", "Sports", "Strategy"]

    # Remove any row where categroy is neither single player/multiplayer and genre not game genre
    df = df[
        df["Categories"].str.contains("Single-player")
        | df["Categories"].str.contains("Multi-player")
        | df["Genres"].str.contains("|".join(actual_game_genres))
    ]

    return df


# Function to replace missing data or drop entries with missing data
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # Remove any games not worth lookin at (Don't have relevant info)
    drop_na_attributes = [
        "Name",
        "About the game",
        "Screenshots",
        "Genres",
        "Developers",
    ]
    df.dropna(subset=drop_na_attributes, inplace=True)

    # If there is no publisher specified or it is the same as the developer then set publisher to Selfpublished
    df["Publishers"] = df["Publishers"].fillna("Selfpublished")
    df.loc[df["Developers"] == df["Publishers"], "Publishers"] = "Selfpublished"

    df["Tags"] = df["Tags"].fillna("")

    # Should all be removed anyway before because not useful (besides Categories)
    replace_na_attributes = [
        "Reviews",
        "Support url",
        "Support email",
        "Website",
        "Metacritic url",
        "Score rank",
        "Notes",
        "Movies",
        "Categories",
    ]
    df[replace_na_attributes] = df[replace_na_attributes].fillna("")

    return df


# Function to scale all columns of dataframe assuming the dataframe is preprocessed
def scaling(df: pd.DataFrame, method: str) -> pd.DataFrame:
    # Apply StandardScaler to all columns
    if method == "standard":
        # Initialize StandardScaler
        scaler = pp.StandardScaler()

    # Apply MinMaxScaler to all columns
    elif method == "minmax":
        # Initialize MinMaxScaler
        scaler = pp.MinMaxScaler()

    # Apply RobustScaler to all columns
    elif method == "robust":
        # Initialize RobustScaler
        scaler = pp.RobustScaler()

    # Fit and transform the dataframe
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    return df


# PCA function
def implement_PCA(df, features) -> tuple:
    # df should be scaled before implementing PCA
    # PCA does not accept Nan values
    # features should be either int or float
    df_selected = df[features].select_dtypes(include=["int64", "float64"])

    print(f"Selected features: {df_selected.columns}")

    pca = PCA(n_components=4)
    pca_components = pca.fit_transform(df_selected)

    # Print explained variance ratio and PCA components
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(
        f"Cumulative explained variance ratio: {pca.explained_variance_ratio_.cumsum()}"
    )

    # Get the PCA loadings (principal component coefficients)
    loadings = pd.DataFrame(
        pca.components_,
        columns=df_selected.columns,
        index=[f"PC{i+1}" for i in range(4)],
    )

    return pca_components, loadings


# Function to add release season
def add_release_season_column(df, date_column="Release date"):
    # Ensure the date column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    # Map months to seasons and create the 'release_season' column
    season_mapping = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Fall",
        10: "Fall",
        11: "Fall",
    }

    df["release_season"] = df[date_column].dt.month.map(season_mapping)

    # Map seasons to numerical values
    season_to_num = {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3}
    df["release_season_num"] = df["release_season"].map(season_to_num)

    return df


# Function to convert estimated owners to midpoints
def convert_estimated_owners_to_midpoints(df, column="Estimated owners"):

    # Define a function to calculate midpoints of ranges
    def range_to_midpoint(range_str):
        # Split the range string and convert to integers
        low, high = range_str.replace(",", "").split(" - ")
        return (int(low) + int(high)) / 2

    # Apply the midpoint calculation to the 'Estimated owners' column
    df["estimated_owners_midpoint"] = df[column].apply(
        lambda x: range_to_midpoint(x) if pd.notnull(x) and x != "0 - 0" else 0
    )

    return df


# Function to add positive ratio and total reviews
def add_review_columns(df):
    # Fill NaN values in 'Positive' and 'Negative' columns with 0
    df["Positive"] = df["Positive"].fillna(0)
    df["Negative"] = df["Negative"].fillna(0)

    # Create 'total_reviews' column as the sum of 'Positive' and 'Negative'
    df["total_reviews"] = df["Positive"] + df["Negative"]

    # Set 'total_reviews' to 0 if both 'Positive' and 'Negative' are 0
    df["total_reviews"] = df["total_reviews"].apply(lambda x: 0 if x == 0 else x)

    # Create 'positive_ratio' column, setting it to 0 when 'total_reviews' is 0 to avoid division by zero
    df["positive_ratio"] = df.apply(
        lambda row: (
            row["Positive"] / row["total_reviews"] if row["total_reviews"] > 0 else 0
        ),
        axis=1,
    )

    return df


# Function to add a column based on single player and multi player
def add_player_type_numeric_column(df):
    # Create a new column 'player_type_numeric' based on presence of keywords in 'Categories'
    def check_player_type(categories):
        if isinstance(categories, str):
            single_player = "Single-player" in categories
            multi_player = "Multi-player" in categories or "Multi" in categories

            # Assign numerical values based on presence of single or multi-player
            if single_player and multi_player:
                return 3  # Both Single-player and Multi-player
            elif single_player:
                return 1  # Single-player only
            elif multi_player:
                return 2  # Multi-player only

        return 0  # Unknown or neither

    # Apply the function to create a new numerical column
    df["player_type_numeric"] = df["Categories"].apply(check_player_type)

    return df


# Function to add a column called online-offline
def add_online_offline_column(df):
    # 1 for Online games, 0 for Offline games based on the presence of 'online' in 'Categories'
    df["online_offline"] = df["Categories"].apply(
        lambda x: 1 if isinstance(x, str) and "online" in x.lower() else 0
    )
    return df


# Clustering
# Implement DBSCAN clustering
def implement_DBSCAN(pca_components, eps_value):
    db_clustering = DBSCAN(eps=eps_value, min_samples=160)
    db_labels = db_clustering.fit_predict(pca_components)
    return db_labels


# Choose best epsilon value for DBSCAN
def choose_best_eps(distances):
    kneedle = KneeLocator(
        range(len(distances)), distances, curve="convex", direction="increasing"
    )
    optimal_eps = round(distances[kneedle.elbow], 2)

    print(f"Optimal value for epsilon: {optimal_eps}")
    return optimal_eps


# Function to assign a score to each cluster based on the columns (positive columns mean the greater the better and negative columns mean the smaller the better)
def scoring_clusters(cluster_means):
    postive_score_cols = [
        "Peak CCU",
        "DLC count",
        "Metacritic score",
        "Positive",
        "User score",
        "Recommendations",
        "Average playtime forever",
        "Average playtime two weeks",
        "Median playtime forever",
        "Median playtime two weeks",
        "total_reviews",
        "positive_ratio",
    ]
    negative_score_cols = ["Negative"]
    cluster_scores = [0, 0, 0, 0]
    for col in postive_score_cols:
        index_max = np.argmax(cluster_means[col])
        cluster_scores[index_max] += 1

    index_min = np.argmin(cluster_means[negative_score_cols])
    cluster_scores[index_min] += 1
    return cluster_scores
