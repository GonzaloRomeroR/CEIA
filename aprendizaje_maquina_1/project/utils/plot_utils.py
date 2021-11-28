import matplotlib.pyplot as plt
import math
import seaborn as sns
import numpy as np

sns.set()


def display_dataset_distributions(dataset, figsize=(22, 10), unique=False):
    if unique:
        params = {"bins": np.unique(dataset)}
    else:
        params = {}
    fig = dataset.hist(xlabelsize=12, ylabelsize=12, figsize=figsize, **params)
    [x.title.set_size(14) for x in fig.ravel()]
    plt.tight_layout()
    plt.show()


def display_dataset_categorical(dataset, n_rows=1, figsize=(20, 10)):
    fig, ax = plt.subplots(
        n_rows, int(len(dataset.columns) / n_rows) + 1, figsize=figsize
    )
    fig.tight_layout(pad=5.0)
    for i, categorical_feature in enumerate(dataset):
        dataset[categorical_feature].value_counts().plot(
            kind="bar", ax=ax[i // (n_rows + 1)][i % (n_rows + 1)], grid=True
        ).set_title(categorical_feature)
    plt.grid()


def plot_box_whiskers(df, variables, row_num=4, figsize=(20, 30)):
    rows = int(len(variables) / row_num)
    cols = int(math.ceil(len(variables) / rows))
    _, axes = plt.subplots(rows, cols, figsize=figsize)
    for i in range(len(variables)):
        row = i % rows
        col = i // rows
        sns.boxplot(y=df[variables[i]], ax=axes[row][col])
        axes[row][col].set_title(variables[i])


def plot_correlation_heatmap(df):
    plt.subplots(figsize=(20, 20))
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, cmap="YlGnBu", mask=mask, square=True, annot=True)
    sns.set(font_scale=0.9)

