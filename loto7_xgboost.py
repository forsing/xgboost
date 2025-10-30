"""
Loto Skraceni Sistemi 
https://www.lotoss.info
ABBREVIATED LOTTO SYSTEMS
"""


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from qiskit_machine_learning.utils import algorithm_globals
import random


import xgboost as xgb

print()
print("XGBoost version:")
print(xgb.__version__)
print()
"""
XGBoost version:
3.0.5
"""



# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED



"""
svih 4502 izvlacenja Loto 7/39 u Srbiji
30.07.1985.- 28.10.2025.
"""

# 1. Učitaj loto podatke
df = pd.read_csv("/data/loto7_4502_k85.csv", header=None)


###################################


print()
print("Prvih 5 ucitanih kombinacija iz CSV fajla:")
print()
print(df.head())
print()
"""
Prvih 5 ucitanih kombinacija iz CSV fajla:

    0   1   2   3   4   5   6
0   5  14  15  17  28  30  34
1   2   3  13  18  19  23  37
2  13  17  18  20  21  26  39
3  17  20  23  26  35  36  38
4   3   4   8  11  29  32  37
"""

print()
print("Zadnjih 5 ucitanih kombinacija iz CSV fajla:")
print()
print(df.tail())
print()
"""
Zadnjih 5 ucitanih kombinacija iz CSV fajla:

      0   1   2   3   4   5   6
4497  4  13  14  19  27  35  37
4498  1   7  13  18  25  30  34
4499  1   5   6   7  11  24  37
4500  2   4   6  11  21  33  35
4501  1   3  11  12  19  35  38
"""


# Pretpostavljamo da prve 7 kolona sadrže brojeve lutrije
df = df.iloc[:, :7]

# Kreiranje ulaznih (X) i izlaznih (y) podataka
X = df.shift(1).dropna().values
y = df.iloc[1:].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=39)



####################################



# 2. Minimalni i maksimalni dozvoljeni brojevi po poziciji
min_val = [1, 2, 3, 4, 5, 6, 7]
max_val = [33, 34, 35, 36, 37, 38, 39]

# 3. Funkcija za mapiranje brojeva u indeksirani opseg [0..range_size-1]
def map_to_indexed_range(df, min_val, max_val):
    df_indexed = df.copy()
    for i in range(df.shape[1]):
        df_indexed[i] = df[i] - min_val[i]
        # Provera da li su svi brojevi u validnom opsegu
        if not df_indexed[i].between(0, max_val[i] - min_val[i]).all():
            raise ValueError(f"Vrednosti u koloni {i} nisu u opsegu 0 do {max_val[i] - min_val[i]}")
    return df_indexed

# 4. Primeni mapiranje
df_indexed = map_to_indexed_range(df, min_val, max_val)



print()
print("Prvih 5 mapiranih kombinacija:")
print()
print(df_indexed.head())
print()
"""
Prvih 5 mapiranih kombinacija:

    0   1   2   3   4   5   6
0   4  12  12  13  23  24  27
1   1   1  10  14  14  17  30
2  12  15  15  16  16  20  32
3  16  18  20  22  30  30  31
4   2   2   5   7  24  26  30
"""

print()
print("Zadnjih 5 mapiranih kombinacija:")
print()
print(df_indexed.tail())
print()
"""
Zadnjih 5 mapiranih kombinacija:

      0   1   2   3   4   5   6
4497  3  11  11  15  22  29  30
4498  0   5  10  14  20  24  27
4499  0   3   3   3   6  18  30
4500  1   2   3   7  16  27  28
4501  0   1   8   8  14  29  31
"""

# Pretpostavljamo da prve 7 kolona sadrže brojeve lutrije
df_indexed = df_indexed.iloc[:, :7]

# Kreiranje ulaznih (X) i izlaznih (y) podataka
X_x = df_indexed.shift(1).dropna().values
y_x = df_indexed.iloc[1:].values


# Train/test split
X_train_x, X_test_x, y_train_x, y_test_x = train_test_split(X_x, y_x, test_size=0.25, random_state=39)


########################################


# Train XGBoost model
xgb_model = xgb.XGBRFRegressor(objective='reg:squarederror', n_estimators=1000, verbosity=0, random_state=39, base_score=0.5, max_depth=5, use_label_encoder=False)
# xgb_model = xgb.XGBRFRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 1000)

xgb_model.fit(X_train, y_train)

# Predict lottery numbers
# predicted_numbers = xgb_model.predict(X_test)
predicted_numbers = xgb_model.predict(X_test[0].reshape(1, -1))


# Convert predictions to integers
predicted_numbers = np.round(predicted_numbers).astype(int)

print()
print("Predicted Next Lottery Numbers X y:", predicted_numbers)
print()
"""
XGBRFRegressor
Predicted Next Lottery Numbers X y: [[ 5 10 15 20 25 30 35]]



XGBRegressor

"""


#######################################


# Train XGBoost model
xgb_model_x = xgb.XGBRFRegressor(objective='reg:squarederror', n_estimators=1000, verbosity=0, random_state=39, base_score=0.5, max_depth=5, use_label_encoder=False)
# xgb_model_x = xgb.XGBRFRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 1000)


xgb_model_x.fit(X_train_x, y_train_x)

# Predict lottery numbers
# predicted_numbers = xgb_model.predict(X_test)
predicted_numbers_x = xgb_model_x.predict(X_test_x[0].reshape(1, -1))


# Convert predictions to integers
predicted_numbers_x = np.round(predicted_numbers_x).astype(int)

print()
print("Predicted Next Lottery Numbers X_x y_x:", predicted_numbers_x)
print()
"""
XGBRFRegressor
Predicted Next Lottery Numbers X_x y_x: [[ 4  8 x x x 24 28]]



XGBRegressor

"""


#######################################


# 5. Provera rezultata
print()
print(f"Učitano kombinacija: {df.shape[0]}, Broj pozicija: {df.shape[1]}")
print()
"""
Učitano kombinacija: 4502, Broj pozicija: 7
"""


#######################################



print()
input("Press Enter to close the window ...")
print()

# Press Enter to close the window ...

