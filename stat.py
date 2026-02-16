import pandas as pd

df = pd.read_csv('clinical_data.csv')
print(df.head())
print(f"Number of patients with status 1: {sum(df['status'] == 1)}")

print(f"Number of patients with status 0: {sum(df['status'] == 0)}")