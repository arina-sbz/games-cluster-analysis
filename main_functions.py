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
    return df

# Function to scale all columns of dataframe assuming the dataframe is preprocessed
def scaling(df: pd.DataFrame, method: str) -> pd.DataFrame:
    return df
