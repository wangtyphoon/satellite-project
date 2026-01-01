import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

csv_path = r"gpm_passes_swath_true.csv"

df = pd.read_csv(csv_path)
df = df[(df['intensity_bst'] > 55) & (df['delta_24h'] > 0)].copy()
df = df.loc[:, 'zFactorFinal_max_r100':]
