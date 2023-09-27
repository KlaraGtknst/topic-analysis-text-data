from elasticSearch import db_elasticsearch
from constants import CLIENT_ADDR, MODEL_NAMES


def main(src_paths, image_src_path, client_addr=CLIENT_ADDR, n_pools=1, model_names: list = MODEL_NAMES):
    for model_name in MODEL_NAMES:
        print(f'Running {model_name}...')
        db_elasticsearch.main(src_paths, image_src_path, client_addr, n_pools, model_names=[model_name])