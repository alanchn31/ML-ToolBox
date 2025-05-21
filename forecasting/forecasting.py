import pandas as pd
import numpy as np
from mpire import WorkerPool
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirect

def _local_lgbm_forecast(id, 
                         df, 
                         lags,
                         rolling_feats,
                         forecast_periods, 
                         forecast_mode, 
                         target_col='target_col',
                         exog_feats=None):
    # 1. Recursive Multi-step Local LGBM Forecaster
    if forecast_mode == 'recursive':
        forecaster = ForecasterRecursive(
                        regressor=LGBMRegressor(random_state=RANDOM_SEED, verbose=-1),
                        lags=lags,
                        window_features=rolling_feats
                    )
    # 2. Direct Multi-step Local LGBM Forecaster
    elif forecast_mode == 'direct':
        forecaster = ForecasterDirect(
                        regressor=LGBMRegressor(random_state=RANDOM_SEED, verbose=-1),
                        steps=forecast_periods,
                        lags=lags,
                        window_features=rolling_feats,
                    )
    if exog_feats:    
        forecaster.fit(y=df[target_col],
                       exog=df[exog_feats]
                       )
    else:
        forecaster.fit(y=df[target_col]
                       )
    pred = forecaster.predict(steps=forecast_periods)
    pred_df = pd.DataFrame({'forecast': pred})
    pred_df["item_id"] = id
    pred_df['date_key'] = pd.date_range(df['date'].max()+pd.Timedelta(days=1),
                                        df['date'].max()+pd.Timedelta(days=forecast_periods))
    return pred_df


def multi_threaded_local_lgbm_forecast(df, lags, rolling_feats, forecast_period, target_col, forecast_mode='recursive'):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = functools.partial(_local_lgbm_forecast,
                                lags=lags,
                                rolling_feats=rolling_feats,
                                forecast_periods=forecast_period,
                                forecast_mode=forecast_mode,
                                target_col=target_col)
        with WorkerPool(n_jobs=NUM_CPU_CORES) as pool:
            list_forecast_lgbm = pool.map_unordered(model, df.groupby('item_id'), progress_bar=True)
        forecast_result_lgbm = pd.concat(list_forecast_lgbm, axis=0)
        return forecast_result_lgbm
    
# Global LightGBM:
def global_lgbm_forecast(series_dict, rolling_feats, forecast_periods):
    series_dict = series_long_to_dict(
                    data=df,
                    series_id='item_id',
                    index='date',
                    values='demand',
                    freq='D'
    )

    # Recursive Multi-series Global LGBM Forecaster
    forecaster = ForecasterRecursiveMultiSeries(
                    regressor=LGBMRegressor(random_state=RANDOM_SEED, verbose=-1),
                    lags=LAGS,
                    window_features=rolling_feats,
                    dropna_from_series=False
                )

    forecaster.fit(series=series_dict, suppress_warnings=True)
    forecasts_df = forecaster.predict(steps=forecast_periods)
    return forecasts_df

forecast_result_global_lgbm_recursive = global_lgbm_forecast(series_dict, rolling_feats, FORECAST_PERIOD)
forecast_result_global_lgbm_recursive.columns = ['forecast_' + c for c in forecast_result_global_lgbm_recursive.columns]
forecast_result_global_lgbm_recursive['date'] = forecast_result_global_lgbm_recursive.index
forecast_result_global_lgbm_recursive = pd.wide_to_long(forecast_result_global_lgbm_recursive, 
                                                        stubnames=['forecast_'], i="date", j="item_id",suffix='\w+')
