import os
from matplotlib import pyplot as plt
import pandas as pd
from elasticSearch import db_elasticsearch
from constants import CLIENT_ADDR, MODEL_NAMES
from timeit import default_timer as timer

def get_times(src_paths, image_src_path, client_addr=CLIENT_ADDR, n_pools=1, model_names: list = MODEL_NAMES):
    '''
    deprecated
    '''
    times = pd.read_json('times_per_emb.json') if os.path.exists('times_per_emb.json') else pd.DataFrame(columns=['model', 'time'])
    for model_name in model_names:
        print(f'Running {model_name}...')
        start = timer()
        db_elasticsearch.main(src_paths, image_src_path, client_addr, n_pools, model_names=[model_name])
        end = timer()
        duration = end - start
        print('time ellapsed: ', duration)
        times = pd.concat([times, pd.DataFrame({'model': model_name, 'time': [duration]})])
    times.reset_index(inplace=True)
    times.to_json('times_per_emb.json')

def get_specific_times(src_paths, client_addr=CLIENT_ADDR, model_names: list = MODEL_NAMES):
    times = pd.read_json('times_per_emb.json') if os.path.exists('times_per_emb.json') else pd.DataFrame(columns=['model', 'time'])
    start = timer()
    db_elasticsearch.initialize_db(src_paths, client_addr=client_addr) # WORKS
    end = timer()
    duration = end - start
    times = pd.concat([times, pd.DataFrame({'model': 'init new db', 'time': [duration]})])

    print('start creating documents using bulk')
    start = timer()
    db_elasticsearch.create_documents(src_paths = src_paths, client_addr=client_addr) # WORKS
    end = timer()
    duration = end - start
    times = pd.concat([times, pd.DataFrame({'model': 'create docs', 'time': [duration]})])
    print('finished creating documents using bulk')

    print('start inserting documents embeddings using bulk')
    for model_name in model_names:
        print('started with model: ', model_name)
        start = timer()
        db_elasticsearch.insert_embedding(src_paths = src_paths, client_addr=client_addr, model_name = model_name)
        end = timer()
        duration = end - start
        times = pd.concat([times, pd.DataFrame({'model': model_name, 'time': [duration]})])
        print('finished model: ', model_name)
    print('finished inserting documents embeddings using bulk')
    times.reset_index(inplace=True)
    times.to_json('times_per_emb.json')

def display_times():
    times = pd.read_json('times_per_emb.json')
    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.barh(times['model'], times['time'])
    ax.bar_label(bars)
    for bars in ax.containers:
        ax.bar_label(bars)
    plt.title('Time per embedding')
    plt.savefig('results/time_per_emb.pdf', format="pdf")
    plt.show()

def main(src_paths, image_src_path, client_addr=CLIENT_ADDR, n_pools=1, model_names: list = MODEL_NAMES):
    #get_times(src_paths, image_src_path, client_addr, n_pools, model_names) # deprecated
    get_specific_times(src_paths, client_addr=client_addr, model_names=model_names)
    display_times()