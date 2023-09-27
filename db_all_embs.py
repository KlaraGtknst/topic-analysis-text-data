import os
from matplotlib import pyplot as plt
import pandas as pd
from elasticSearch import db_elasticsearch
from constants import CLIENT_ADDR, MODEL_NAMES
from timeit import default_timer as timer

def get_times(src_paths, image_src_path, client_addr=CLIENT_ADDR, n_pools=1, model_names: list = MODEL_NAMES):
    times = pd.read_json('times_per_emb.json') if os.path.exists('times_per_emb.json') else pd.DataFrame(columns=['model', 'time'])
    for model_name in model_names:
        print(f'Running {model_name}...')
        start = timer()
        db_elasticsearch.main(src_paths, image_src_path, client_addr, n_pools, model_names=[model_name])
        end = timer()
        duration = end - start
        print('time ellapsed: ', duration)
        times = pd.concat([times, pd.DataFrame({'model': model_names, 'time': duration})], ignore_index=True)
    times.to_json('times_per_emb.json')

def display_times():
    times = pd.read_json('times_per_emb.json')
    fig, ax = plt.subplots()
    bars = ax.barh(times['model'], times['time'])
    ax.bar_label(bars)
    for bars in ax.containers:
        ax.bar_label(bars)
    plt.title('Time per embedding')
    plt.show()

def main(src_paths, image_src_path, client_addr=CLIENT_ADDR, n_pools=1, model_names: list = MODEL_NAMES):
    #get_times(src_paths, image_src_path, client_addr, n_pools, model_names)
    display_times()