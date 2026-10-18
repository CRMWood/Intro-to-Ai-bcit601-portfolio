import pandas as pd
from config import RAW_DIR, PROCESSED_DIR


# Step 1: Load the dataset from the data folder.
df = pd.read_csv(RAW_DIR/ "BCIT601_starter_service_requests_dataset.csv")

# Step 2: clean columns (fill blanks, lower case and remove whitespace and delete duplicates)

COLUMNS = ["request_text", "category", "suburb", "priority"]
for column in COLUMNS:
    if column == "category":
        df[column] = df[column].fillna("Critical Error").str.lower().str.strip()
    else:
        df[column] = df[column].fillna("unknown")
    df[column] = df[column].str.lower().str.strip()


#drop any duplicate rows
df = df.drop_duplicates()

#check that the fill has worked and that no column has an empty field
print(df.isnull().sum())


df.to_csv(PROCESSED_DIR/ "BCIT601_starter_service_requests_dataset_clean.csv", index=False)
print("Saved cleaned file.")