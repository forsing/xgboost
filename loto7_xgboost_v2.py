"""
Loto Skraceni Sistemi 
https://www.lotoss.info
ABBREVIATED LOTTO SYSTEMS
"""


import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from qiskit_machine_learning.utils import algorithm_globals
import random


import xgboost as xgb

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--no-pause", action="store_true")
_ARGS, _ = _ap.parse_known_args()

print()
print("XGBoost version:")
print(xgb.__version__)
print()
"""
XGBoost version:
3.1.3
"""



# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED



"""
svih 4584 izvlacenja Loto 7/39 u Srbiji
30.07.1985.- 20.03.2026.
"""

# 1. Učitaj loto podatke
df = pd.read_csv("/Users/4c/Desktop/GHQ/data/loto7_4584_k23.csv", header=None)


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
4579   9  10  27  29  30  34  37
4580  11  19  20  21  24  36  38
4581   1   5  11  14  15  25  39
4582   7  22  23  30  31  34  38
4583   1   8  11  12  29  36  39       0   1   2   3   4   5   6
4579   8   8  24  25  25  28  30
4580  10  17  17  17  19  30  31
4581   0   3   8  10  10  19  32
4582   6  20  20  26  26  28  31
4583   0   6   8   8  24  30  32
"""

# Pretpostavljamo da prve 7 kolona sadrže brojeve lutrije
df_indexed = df_indexed.iloc[:, :7]

# Kreiranje ulaznih (X) i izlaznih (y) podataka
X_x = df_indexed.shift(1).dropna().values
y_x = df_indexed.iloc[1:].values


# Train/test split (v2: vremenski)
_nx = len(X_x)
_split_x = int(_nx * 0.75)
X_train_x, X_test_x = X_x[:_split_x], X_x[_split_x:]
y_train_x, y_test_x = y_x[:_split_x], y_x[_split_x:]


########################################


# Train XGBoost model
xgb_model = xgb.XGBRFRegressor(
    objective="reg:squarederror",
    n_estimators=1000,
    verbosity=0,
    random_state=39,
    base_score=0.5,
    max_depth=5,
)
# xgb_model = xgb.XGBRFRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 1000)

xgb_model.fit(X_train, y_train)

# Predict lottery numbers (v2: poslednje poznato izvlačenje)
_x_last = df.iloc[-1:].values.astype(np.float32)
predicted_numbers = xgb_model.predict(_x_last)


# Convert predictions to integers
predicted_numbers = np.round(predicted_numbers).astype(int)

print()
print("Predicted Next Lottery Numbers X y:", predicted_numbers)
print()
"""
XGBRFRegressor
Predicted Next Lottery Numbers X y: [[ 5 x 15 y z 30 35]]



XGBRegressor

"""


#######################################


# Train XGBoost model
xgb_model_x = xgb.XGBRFRegressor(
    objective="reg:squarederror",
    n_estimators=1000,
    verbosity=0,
    random_state=39,
    base_score=0.5,
    max_depth=5,
)
# xgb_model_x = xgb.XGBRFRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 1000)


xgb_model_x.fit(X_train_x, y_train_x)

# Predict lottery numbers (v2: poslednji red mapiranih brojeva)
_x_last_x = df_indexed.iloc[-1:].values.astype(np.float32)
predicted_numbers_x = xgb_model_x.predict(_x_last_x)


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
Učitano kombinacija: 4584, Broj pozicija: 7
"""


#######################################



print()
if not _ARGS.no_pause:
    input("Press Enter to close the window ...")
print()

# Press Enter to close the window ...



"""
python3 loto7_xgboost_v2.py 
python3 loto7_xgboost_v2.py --no-pause
"""
