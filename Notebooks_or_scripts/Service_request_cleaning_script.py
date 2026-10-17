import pandas as pd
from config import RAW_DATA, CLEAN_DATA


# Step 1: Load the dataset from the data folder.
df = pd.read_csv(RAW_DATA)

# Step 2: clean columns (fill blanks, lower case and remove whitespace and delete duplicates)

#request_text Column
df["request_text"] = df["request_text"].fillna("unknown request")
df["request_text"] = df["request_text"].str.lower().str.strip()

#category Column
df["category"] = df["category"].fillna("critical error!")
df["category"] = df["category"].str.lower().str.strip()

#suburb Column
df["suburb"] = df["suburb"].fillna("unknown Suburb")
df["suburb"] = df["suburb"].str.lower().str.strip()

#priority Column
df["priority"] = df["priority"].fillna("unknown priority")
df["priority"] = df["priority"].str.lower().str.strip()

#date_reported Column
df["date_reported"] = df["date_reported"].fillna("unknown date of request")
df["date_reported"] = df["date_reported"].str.lower().str.strip()

#drop any duplicate rows
df = df.drop_duplicates()

#check that the fill has worked and that no column has an empty field
print(df.isnull().sum())


df.to_csv(CLEAN_DATA, index=False)
print("Saved cleaned file.")