from sklearn.preprocessing import KBinsDiscretizer
import numpy as np


def discretize_variable(df, variable, n_bins=4, strategy="kmeans"):
    kbins = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    df[variable] = kbins.fit_transform(df[variable].to_numpy().reshape(-1, 1))
    return df


def eliminate_outliers_capping(df, variables):
    for variable in variables:
        upper_limit, lower_limit = find_skewed_boundaries(df, variable, 1.5)
        df[variable] = np.where(
            df[variable] > upper_limit,
            upper_limit,
            np.where(df[variable] < lower_limit, lower_limit, df[variable]),
        )
    return df


def find_skewed_boundaries(df, variable, distance=1.5):
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)
    return upper_boundary, lower_boundary


def eliminate_high_corr_columns(df, max_corr=0.95):
    corr = df.corr()
    indexes = np.where(np.abs(corr) > max_corr)
    indexes = [
        (corr.index[x], corr.columns[y]) for x, y in zip(*indexes) if x != y and x < y
    ]
    drop_cols = [pair[0] for pair in indexes]
    print(f"Eliminated columns: {drop_cols}")
    return df.drop(drop_cols, axis=1)


def get_higher_corr(df, variable, method="pearson"):
    corr = df.corr(method=method)[variable]
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
    return corr
