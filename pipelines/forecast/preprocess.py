"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd
import awswrangler as wr

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def filter_month(df, year, month):
    df = df[df['year']==year]
    
    return df[df['month']==month]

prefix = 's3://'
files = ['ingredient.parquet.gz','menu_ingredient.parquet.gz','menu.parquet.gz',
         'order-*.parquet.gz','order_detail-*.parquet.gz','shop.parquet.gz']
selected_ingredients = ['Air Mineral','Gula Pasir','Telur','Kopi','Susu Sapi']

if __name__ == "__main__":
    logger.debug("Starting preprocessing.")

    base_dir = "/opt/ml/processing"
    bucket = ''

    logger.debug("Reading data from S3.")
    df_ing = wr.s3.read_parquet(path=f'{prefix}{bucket}/{files[0]}')[['id','name']]
    df_men_ing = wr.s3.read_parquet(path=f'{prefix}{bucket}/{files[1]}')
    df_ord = wr.s3.read_parquet(path=f'{prefix}{bucket}/{files[3]}')[['id','shop_id','is_completed','created_time']]
    df_ord_det = wr.s3.read_parquet(path=f'{prefix}{bucket}/{files[4]}')[['order_id','menu_id','quantity']]
    df_shop = wr.s3.read_parquet(path=f'{prefix}{bucket}/{files[5]}')

    logger.debug("Preparing Dataset.")
    df_ord = df_ord[df_ord['shop_id']==1]
    df_ord = df_ord[df_ord['is_completed']==1]
    df = df_ord.join(df_ord_det.set_index('order_id'), on='id')
    df = df.join(df_men_ing.set_index('menu_id'), on='menu_id')
    df = df.join(df_ing.set_index('id'), on='ingredient_id')
    df = df.drop(['id','shop_id','is_completed','menu_id','ingredient_id'], axis=1)
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    df = df[df['name'].isin(selected_ingredients)]
    df = df.drop(['amount'], axis=1)
    df = df.groupby([df['created_time'].dt.strftime('%Y-%m-%d'),'name']).sum().reset_index()
    
    logger.debug("Feature Engineering.")
    df['days_of_week'] = df['created_time'].apply(lambda x: pd.Period(x,'D').day_of_week)
    df['day'] = df['created_time'].apply(lambda x: pd.Period(x,'D').day)
    df['month'] = df['created_time'].apply(lambda x: pd.Period(x,'D').month)
    df['year'] = df['created_time'].apply(lambda x: pd.Period(x,'D').year)
    final_df = df.drop('created_time', axis=1)

    logger.debug("Defining transformers.")
    categorical_features = ['name']

    categorical_transformer = Pipeline(
        steps=[
            ("one", OneHotEncoder(drop="first",handle_unknown="error",sparse=False))
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
        ("cat", categorical_transformer, categorical_features),
    ],
        remainder='passthrough'
    )
    
    logger.info("Applying transforms.")
    y = final_df.pop("quantity")
    X_pre = preprocess.fit_transform(final_df)
    y_pre = y.to_numpy().reshape(len(y), 1)

    X = np.concatenate((y_pre, X_pre), axis=1)

    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X))
    last_month = len(filter_month(final_df, 2020, 12))
    last_two_month = len(filter_month(final_df, 2020, 11)) + last_month
    train, validation, test = np.split(X, [len(X)-last_two_month, len(X)-last_month])

    logger.info("Writing out datasets to %s.", base_dir)
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
