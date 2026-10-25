
from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model_building'))
from data_register import load_and_validate

df = load_and_validate(ROOT / 'data' / 'tourism.csv')
df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})
df = df.drop(columns=['CustomerID'])  # Identifier; not a behavioural feature.

X = df.drop(columns=['ProdTaken'])
y = df['ProdTaken']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

artifacts = ROOT / 'artifacts'
artifacts.mkdir(exist_ok=True)
pd.concat([X_train, y_train], axis=1).to_csv(artifacts / 'train.csv', index=False)
pd.concat([X_test, y_test], axis=1).to_csv(artifacts / 'test.csv', index=False)
print(f'Saved {len(X_train):,} training rows and {len(X_test):,} test rows.')
