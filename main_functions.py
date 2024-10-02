import pandas as pd
import re
from sklearn import preprocessing as pp
from sklearn.decomposition import PCA

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
    # Remove any games not worth lookin at (Don't have relevant info)
    drop_na_attributes = ["Name", "About the game", "Screenshots", "Genres", "Developers", "Categories"]
    df.dropna(subset=drop_na_attributes, inplace=True)

    # If there is no publisher specified or it is the same as the developer then set publisher to Selfpublished
    df["Publishers"] = df["Publishers"].fillna("Selfpublished")
    df.loc[df["Developers"] == df["Publishers"], "Publishers"] = "Selfpublished"

    df["Tags"] = df["Tags"].fillna("")

    # Should all be removed anyway before because not useful
    replace_na_attributes = ["Reviews", "Support url", "Support email", "Website", "Metacritic url", "Score rank", "Notes", "Movies"]
    df[replace_na_attributes] = df[replace_na_attributes].fillna("")

    return df


# Function to scale all columns of dataframe assuming the dataframe is preprocessed
def scaling(df: pd.DataFrame, method: str) -> pd.DataFrame:

    # Apply StandardScaler to all columns
    if method == "standard":
        # Initialize StandardScaler
        scaler = pp.StandardScaler()

        # Fit and transform the dataframe
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Apply MinMaxScaler to all columns
    elif method == "minmax":
        # Initialize MinMaxScaler
        scaler = pp.MinMaxScaler()

        # Fit and transform the dataframe
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Apply RobustScaler to all columns
    elif method == "robust":
        # Initialize RobustScaler
        scaler = pp.RobustScaler()

        # Fit and transform the dataframe
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

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
