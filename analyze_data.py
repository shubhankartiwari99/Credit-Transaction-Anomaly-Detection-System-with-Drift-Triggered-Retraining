import pandas as pd

df = pd.read_csv('creditcard.csv')

print("Data description:")
print(df.describe())

print("\nClass value counts:")
print(df['Class'].value_counts())