import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

def preprocess():
    # Load data
    data_path = os.path.join('..', 'data', 'creditcard.csv')
    df = pd.read_csv(data_path)

    # Features and target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Scale Amount and Time
    scaler = StandardScaler()
    X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])

    # Split into train, val, test (70-15-15)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Apply SMOTE to training set
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    print(f"Original training set: {y_train.value_counts()}")
    print(f"SMOTE training set: {pd.Series(y_train_sm).value_counts()}")

    # Save processed data
    processed_path = os.path.join('..', 'data', 'processed_data.pkl')
    joblib.dump((X_train_sm, y_train_sm, X_val, y_val, X_test, y_test, scaler), processed_path)
    print(f"Processed data saved to {processed_path}")

if __name__ == '__main__':
    preprocess()