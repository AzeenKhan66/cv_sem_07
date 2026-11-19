# Comparative Implementation of ML Regression Algorithms
**Dataset:** California Housing (20,640 districts), target = `median_house_value`

**Models:** Linear Regression, Decision Tree, Random Forest, Neural Network (MLP), XGBoost, KNN

**Setup:** 80:20 split (`random_state=42`), median imputation and scaling fitted on training data only.

**How to run:** Runtime > Run all. The dataset downloads automatically, so no upload is needed.

```python
import warnings
warnings.filterwarnings('ignore')

import io
import tarfile
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)
print('Libraries loaded')
```

## 1. Load the dataset automatically
The `housing.csv` file is downloaded from the Hands-On ML project on GitHub and read directly into pandas.

```python
URL = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz'

try:
    with urllib.request.urlopen(URL, timeout=120) as response:
        archive_bytes = response.read()
except Exception as e:
    raise RuntimeError('Could not download the dataset from ' + URL + ' : ' + str(e))

with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode='r:gz') as tar:
    csv_member = [m for m in tar.getmembers() if m.name.endswith('housing.csv')][0]
    df = pd.read_csv(tar.extractfile(csv_member))

print('Dataset shape:', df.shape)
df.head()
```

## 2. Exploratory data analysis

```python
df.info()
print()
print('Missing values per column:')
print(df.isnull().sum())
print()
print('ocean_proximity categories:')
print(df['ocean_proximity'].value_counts())
print()
df.describe()
```

## 3. Preprocessing and 80:20 split
- One-hot encode `ocean_proximity`
- Split 80:20 with `random_state=42`
- Fill missing `total_bedrooms` with the **training** median
- Create scaled copies for Linear Regression, KNN and Neural Network

```python
target = 'median_house_value'

# Make category names safe for XGBoost (no '<' or spaces), then one-hot encode
df['ocean_proximity'] = df['ocean_proximity'].str.replace('<', 'LT ', regex=False).str.replace(' ', '_')
df = pd.get_dummies(df, columns=['ocean_proximity'], dtype=int)

X = df.drop(columns=[target])
y = df[target]

# 80:20 split, identical for all six models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fill missing total_bedrooms with the TRAINING median only
imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns, index=X_test.index)

# Scaled copies for Linear Regression, KNN and Neural Network (fitted on training data only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('Train:', X_train.shape, '| Test:', X_test.shape)
print('Missing values left in training set:', int(X_train.isnull().sum().sum()))
```

## 4. Define the six models
Each entry is `(model, needs_scaled_features)`. The Neural Network also uses a scaled target, which is inverse-transformed automatically when predicting.

```python
models = {
    'Linear Regression': (LinearRegression(), True),
    'Decision Tree': (DecisionTreeRegressor(max_depth=8, random_state=42), False),
    'Random Forest': (RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1), False),
    'Neural Network': (
        TransformedTargetRegressor(
            regressor=MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                                   solver='adam', max_iter=500, random_state=42),
            transformer=StandardScaler(),
        ),
        True,
    ),
    'XGBoost': (XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6,
                             random_state=42), False),
    'KNN': (KNeighborsRegressor(n_neighbors=5), True),
}

print('Models defined:', list(models.keys()))
```

## 5. Train, predict and evaluate
Metrics: MAE, MSE, RMSE and R².

```python
results = []
predictions = {}

for name, (model, scaled) in models.items():
    X_tr = X_train_scaled if scaled else X_train
    X_te = X_test_scaled if scaled else X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    predictions[name] = y_pred

    mse = mean_squared_error(y_test, y_pred)
    results.append({
        'Model': name,
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'R2': r2_score(y_test, y_pred),
    })
    print(name, 'finished')

results_df = pd.DataFrame(results).sort_values('RMSE').reset_index(drop=True)
results_df.index = results_df.index + 1
results_df.index.name = 'Rank'

results_df.style.format({'MAE': '{:,.0f}', 'MSE': '{:,.0f}', 'RMSE': '{:,.0f}', 'R2': '{:.4f}'})
```

## 6. Actual vs predicted plots

```python
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for ax, (name, y_pred) in zip(axes.ravel(), predictions.items()):
    ax.scatter(y_test, y_pred, alpha=0.3, s=10)
    lims = [0, max(float(y_test.max()), float(np.max(y_pred)))]
    ax.plot(lims, lims, 'r--', lw=2, label='Perfect prediction')
    ax.set_title(name + ' (R² = ' + format(r2_score(y_test, y_pred), '.3f') + ')')
    ax.set_xlabel('Actual house value (USD)')
    ax.set_ylabel('Predicted house value (USD)')
    ax.legend()

plt.suptitle('Actual vs Predicted: Test Set (4,128 rows)', fontsize=16)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150)
plt.show()
```

## 7. RMSE comparison chart

```python
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(results_df['Model'], results_df['RMSE'], color='steelblue')
ax.invert_yaxis()
ax.set_xlabel('RMSE (USD), lower is better')
ax.set_title('Model comparison by RMSE')
plt.tight_layout()
plt.show()
```

## 8. Best model and export

```python
best = results_df.iloc[0]
print('Best model:', best['Model'])
print('RMSE (USD):', format(best['RMSE'], ',.0f'))
print('R2:', format(best['R2'], '.4f'))

results_df.to_csv('comparison_table.csv')

try:
    from google.colab import files
    files.download('comparison_table.csv')
    files.download('actual_vs_predicted.png')
except ImportError:
    print('Files saved: comparison_table.csv, actual_vs_predicted.png')
```
