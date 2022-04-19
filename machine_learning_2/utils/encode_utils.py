import pandas as pd
from sklearn import preprocessing
import category_encoders as ce


def one_hot_encoding(df, column_name):
    one_hot = pd.get_dummies(df[column_name])
    df = df.drop(column_name, axis=1)
    # Change columns names
    for column in one_hot.columns:
        one_hot.rename(columns={column: column_name + "_" + column}, inplace=True)
    return df.join(one_hot)


def boolean_encoding(df, column_name, mapping={"Yes": 1, "No": 0}, default=-1):
    df[column_name] = getattr(df, column_name).map(mapping).fillna(default).astype(int)
    print(df)
    return df


def binary_encoding(df, column_name):
    encoder = ce.BinaryEncoder(cols=[column_name], return_df=True)
    return encoder.fit_transform(df)


def count_encoding(df, column_name):
    count_map = df[column_name].value_counts().to_dict()
    df[column_name] = df[column_name].map(count_map)
    return df


def label_encoding(df, column_name):
    le = preprocessing.LabelEncoder()
    le.fit(df[column_name])
    df[column_name] = le.transform(df[column_name])
    return df


def average_encoding(df, column_name, target_column):
    average = df.groupby([column_name])[target_column].mean().to_dict()
    df[column_name] = df[column_name].map(average)
    return df
