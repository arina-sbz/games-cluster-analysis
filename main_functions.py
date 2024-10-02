import pandas as pd
import re
from sklearn import preprocessing as pp
from sklearn.decomposition import PCA
import re

# List of words to remove
inappropriate_words = ["sex", "sexual content", "nudity", "hentai", "nsfw"]

# List of columns to check
columns_to_check = ["Genres", "Categories", "Tags", "Notes", "About the game"]


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


# Function to replace missing data or drop entries with missing data
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df


# Function to scale all columns of dataframe assuming the dataframe is preprocessed
def scaling(df: pd.DataFrame, method: str) -> pd.DataFrame:
    return df


# PCA function
def implement_PCA(df, features):
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

    # Show the plot
    plt.show()


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


def merge_genres_tags(df):
    # Fill NaN values in 'Tags' and 'Genres' with an empty string
    df["Genres"] = df["Genres"].fillna("")
    df["Tags"] = df["Tags"].fillna("")

    # Merge 'Genres' and 'Tags' columns into a new column 'Genres_Tags'
    df["Genres_Tags"] = df.apply(
        lambda row: ",".join(set(row["Genres"].split(",") + row["Tags"].split(","))),
        axis=1,
    )

    return df


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


def add_online_offline_column(df):
    # Create a new column 'online_offline' with values:
    # 1 for Online games, 0 for Offline games based on the presence of 'online' in 'Categories'
    df["online_offline"] = df["Categories"].apply(
        lambda x: 1 if isinstance(x, str) and "online" in x.lower() else 0
    )
    return df
