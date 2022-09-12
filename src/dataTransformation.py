
from src.utils.all_utils import read_yaml,create_directory
import argparse
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import os 

def read_df(config_path):
    config = read_yaml(config_path)

    remote_data_path = config["data_source"]
    df = pd.read_csv(remote_data_path,sep=",")

    # Save the dataset in the local directory 
    # Create path to directory: artifacts/raw_local_dir/data.csv
    artifacts_dir = config["artifacts"]['artifacts_dir']
    raw_local_dir = config["artifacts"]['raw_local_dir']
    raw_local_file = config["artifacts"]['raw_local_file']


    raw_local_file_path = os.path.join(artifacts_dir,raw_local_dir,raw_local_file)

    print(raw_local_file_path)

    df = pd.read_csv(raw_local_file_path)
    return df

def process(df, config_path):
    """
    Some strategies adopted:
    - Binaries were converted from strings ('Yes' and 'No') to 1 and 0.
    - The attributes related to quantities were normalized (std dev equal to 1)
    per row. Therefore, parts with different order of magnitudes are 
    approximated. For example: 1 unit of a expensive machine may be different 
    from 1 unit of a screw, but if we standard deviate all the quantities we 
    have, we can get a better proportion of equivalence between those items.
    - Missing values for lead_time and perf_month_avg were replaced using 
    series median and mean. 
    """
    # Imput missing lines and drop line with problem
    config = read_yaml(config_path)


    df['lead_time'] = SimpleImputer(strategy='median').fit_transform(
                                    df['lead_time'].values.reshape(-1, 1))
    df = df.dropna()
    for col in ['perf_6_month_avg', 'perf_12_month_avg']:
        df[col] = SimpleImputer(missing_values=-99).fit_transform(
                                    df[col].values.reshape(-1, 1))
    # Convert to binaries
    for col in ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk',
               'stop_auto_buy', 'rev_stop', 'went_on_backorder']:
        df[col] = (df[col] == 'Yes').astype(int)
    # Normalization    
    from sklearn.preprocessing import normalize
    qty_related = ['national_inv', 'in_transit_qty', 'forecast_3_month', 
                   'forecast_6_month', 'forecast_9_month', 'min_bank',
                   'local_bo_qty', 'pieces_past_due', 'sales_1_month', 
                   'sales_3_month', 'sales_6_month', 'sales_9_month',]
    df[qty_related] = normalize(df[qty_related], axis=1)
    df = df.astype(float)

    # Create transformed directory

    artifacts_dir = config["artifacts"]['artifacts_dir']
    transform_data_dir = config["artifacts"]['transform_data_dir']
    transform_data_file = config["artifacts"]['transform_data_file']

    transform_data_dir_path = os.path.join(artifacts_dir,transform_data_dir)
    transform_data_file_path = os.path.join(artifacts_dir,transform_data_dir,\
        transform_data_file)

    create_directory([transform_data_dir_path])

    df.to_csv(transform_data_file_path, index=False)

def transformation(new_df):
    # StandardScaler for scaling the values
    X= new_df.drop(new_df.columns[-1], axis = 1)
    #y = df['went_on_backorder'] 
    X_std = StandardScaler().fit_transform(X)

    # PCA for dimensionality reduction 
    dec = PCA(n_components=2)
    X_reduced = dec.fit_transform(X_std)
    return X_reduced


if __name__=='__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config","-c",default="config/config.yaml")

    parsed_args = args.parse_args()

    df= read_df(config_path=parsed_args.config)
    process(df, config_path=parsed_args.config)
    #X_reduced=transformation(df)
