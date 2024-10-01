import re
import pandas as pd
from sklearn import preprocessing as pp

# List of words to remove
words_to_remove = ["sex", "sexual content", "nudity", "hentai", "nsfw"]

# List of columns to check
columns_to_check = ["Genres", "Categories", "Tags", "Notes", "About the game"]


# Function to filter out rows containing any exact word in the specified columns
def contains_any_word(row):
    # Create a regex pattern that matches the exact words using word boundaries (\b)
    pattern = r"\b(?:" + "|".join(re.escape(word) for word in words_to_remove) + r")\b"

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
