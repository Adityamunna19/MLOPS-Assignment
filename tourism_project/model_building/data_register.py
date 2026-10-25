
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {
    'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier',
    'DurationOfPitch', 'Occupation', 'Gender', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
    'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
    'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'
}

def load_and_validate(csv_path):
    df = pd.read_csv(csv_path)
    unnamed = [c for c in df.columns if c.lower().startswith('unnamed')]
    df = df.drop(columns=unnamed, errors='ignore')
    missing = REQUIRED_COLUMNS - set(df.columns)
    extra = set(df.columns) - REQUIRED_COLUMNS
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if extra:
        raise ValueError(f'Unexpected columns: {sorted(extra)}')
    if df.empty:
        raise ValueError('The dataset is empty.')
    if not set(df['ProdTaken'].dropna().unique()).issubset({0, 1}):
        raise ValueError('ProdTaken must contain only 0 and 1.')
    if df['CustomerID'].isna().any() or df['CustomerID'].duplicated().any():
        raise ValueError('CustomerID must be present and unique.')
    return df

if __name__ == '__main__':
    data_path = Path(__file__).resolve().parents[1] / 'data' / 'tourism.csv'
    data = load_and_validate(data_path)
    print(f'Validation passed: {data.shape[0]:,} rows × {data.shape[1]:,} columns')
    print('Target distribution:')
    print(data['ProdTaken'].value_counts().sort_index().to_string())
    print(f'Total missing values: {int(data.isna().sum().sum()):,}')
