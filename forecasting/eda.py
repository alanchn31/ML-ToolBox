import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.tsa.seasonal import STL

def summarize_null_valuess(df):
    print("% of null values in each column")
    print(df.isnull().sum() / len(df)*100)

def forward_fill_col(df, idx_cols, ffill_col):
    df[ffill_col] = df.groupby(idx_cols)[ffill_col].ffill()
    return df

def week_of_month(df):
    df['week_of_month'] = pd.to_numeric(df['date'].dt.day/7)
    df['week_of_month'] = df['week_of_month'].apply(lambda x: math.ceil(x))
    return df

def day_of_week(df):
    df['day_of_week'] = df['date'].dt.day_name()
    return df

def gen_seasonal_feats(df):
    df = week_of_month(df)
    df = day_of_week(df)
    return df

def calc_seasonal_strength(item_id, df_sub, period=7):
    ts = df_sub['demand']
    stl = STL(ts, period=period, robust=True)
    result = stl.fit()
    seasonal_strength = max(0, 1-np.var(result.seasonal) / (np.var(result.seasonal) + np.var(result.resid)))
    seas_df = pd.DataFrame({
        'item_id': [item_id],
        'seas_strength': [seasonal_strength]})
    return seas_df

def test_week_of_month_seasonality(item_id, df_sub):
    # Using F-test to check for deterministic seasonality
    groups = [group['demand'].values for _, group in df_sub.groupby('week_of_month')]
    ht_df = pd.DataFrame({
        'item_id': [item_id],
        'pvalue': [f_oneway(*groups).pvalue]})
    return ht_df

def plot_seasonality(df, seasonal_var, y_var, item_id):
    plt.figure()
    sns.boxplot(x=seasonal_var, 
                y=y_var, 
                hue=seasonal_var, 
                data=df.query(f"item_id == '{item_id}'"))
    plt.tight_layout()