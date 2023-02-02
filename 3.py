import torch
import pandas as pd

# Custom function to read the data from a file
def read_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Custom function to preprocess the data
def preprocess_data(df):
    # Perform any necessary preprocessing steps, such as feature scaling, encoding categorical variables, etc.
    return df

# Load the dataset using the custom functions
file_path = "./tree_dataset.csv"
df = read_data(file_path)
df = preprocess_data(df)

# Convert the pandas dataframe to a PyTorch tensor
X = torch.tensor(df.drop("target", axis=1).values, dtype=torch.float32)
y = torch.tensor(df["target"].values, dtype=torch.long)

# Use the preprocessed data for further analysis or modeling
