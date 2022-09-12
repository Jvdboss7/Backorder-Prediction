from src.utils.all_utils import read_yaml, create_directory, save_local_df
import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_and_save(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts_dir = config["artifacts"]['artifacts_dir']
    transform_data_dir = config["artifacts"]['transform_data_dir']
    transform_data_file = config["artifacts"]['transform_data_file']

    transform_data_file_path = os.path.join(artifacts_dir,transform_data_dir,\
        transform_data_file)

    print(transform_data_file_path)
    
    df = pd.read_csv(transform_data_file_path)

    split_ratio = params["base"]["test_size"]
    random_state = params["base"]["random_state"]

    train, test = train_test_split(df, test_size=split_ratio, random_state=random_state)

    split_data_dir = config["artifacts"]["split_data_dir"]

    create_directory([os.path.join(artifacts_dir, split_data_dir)])

    train_data_filename = config["artifacts"]["train"]
    test_data_filename = config["artifacts"]["test"]


    train_data_path = os.path.join(artifacts_dir, split_data_dir, train_data_filename)
    test_data_path = os.path.join(artifacts_dir, split_data_dir, test_data_filename)

    for data, data_path in (train, train_data_path), (test, test_data_path):
        save_local_df(data, data_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()

    split_and_save(config_path=parsed_args.config, params_path=parsed_args.params)