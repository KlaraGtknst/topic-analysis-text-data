import os
from matplotlib import pyplot as plt
import pandas as pd
from doc_images.PCA import PCA_image_clustering 
from constants import CLIENT_ADDR, MODEL_NAMES, NUM_PCA_COMPONENTS
from timeit import default_timer as timer
from elasticSearch import insert_embeddings, create_documents, create_database


def get_specific_times(src_path: str, image_src_path: str, client_addr: str=CLIENT_ADDR, model_names: list = MODEL_NAMES, dir_to_save:str = 'results/'):
    times = pd.read_json(dir_to_save + 'times_per_emb.json') if os.path.exists(dir_to_save + 'times_per_emb.json') else pd.DataFrame(columns=['model', 'time'])
    start = timer()
    create_database.initialize_db(src_path, client_addr=client_addr) # WORKS
    end = timer()
    duration = end - start
    times = pd.concat([times, pd.DataFrame({'model': 'init new db', 'time': [duration]})])

    print('start creating documents using bulk')
    start = timer()
    create_documents.create_documents(src_path = src_path, client_addr=client_addr) # WORKS
    end = timer()
    duration = end - start
    times = pd.concat([times, pd.DataFrame({'model': 'create docs', 'time': [duration]})])
    print('finished creating documents using bulk')

    print('start inserting documents embeddings using bulk')
    for model_name in model_names:
        print('started with model: ', model_name)
        start = timer()
        insert_embeddings.insert_embedding(src_path = src_path, client_addr=client_addr, model_name = model_name)
        end = timer()
        duration = end - start
        times = pd.concat([times, pd.DataFrame({'model': model_name, 'time': [duration]})])
        print('finished model: ', model_name)

    start = timer()
    insert_embeddings.insert_precomputed_clusters(src_path=src_path, image_src_path=image_src_path, client_addr=client_addr)
    end = timer()
    duration = end - start
    times = pd.concat([times, pd.DataFrame({'model': 'pca_optics', 'time': [duration]})])
    print('finished inserting pca-OPTICS cluster df')
    
    print('finished inserting documents embeddings using bulk')
    times.reset_index(inplace=True)
    times.to_json(dir_to_save + 'times_per_emb.json')

def display_times(dir_to_save:str = 'results/'):
    times = pd.read_json(dir_to_save + 'times_per_emb.json')
    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.barh(times['model'], times['time'])
    ax.bar_label(bars)
    for bars in ax.containers:
        ax.bar_label(bars)
    plt.title('Time per embedding')
    plt.savefig(dir_to_save + 'time_per_emb.pdf', format="pdf")
    plt.show()

def main(src_path:str, image_src_path:str, client_addr=CLIENT_ADDR, n_pools:int=1, model_names: list = MODEL_NAMES):
    dir_to_save = 'results/' if os.path.exists('/Users/klara/Developer/Uni/') else '/mnt/stud/home/kgutekunst/visualizations/' # server vs local
    get_specific_times(src_path, client_addr=client_addr, model_names=model_names, image_src_path=image_src_path, dir_to_save=dir_to_save)
    display_times(dir_to_save=dir_to_save)