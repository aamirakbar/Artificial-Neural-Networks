import pandas as pd

file_path = "banknotes.csv"

# Read the CSV data into a DataFrame
df = pd.read_csv(file_path)

# Print the top 10 records
print(df.head(10))
