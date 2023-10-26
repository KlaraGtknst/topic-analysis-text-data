from multiprocessing import Pool
import os
import sys

def insert_embeddings(x,y,z,d):
    print(len(x),y,z,d)

class wrapper:
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, src_paths):
        insert_embeddings(src_paths, "fixed 1", "fixed 2", self.model_name)

def scanRecurse(baseDir: str):
    baseDir = baseDir.split('*')[0] if  '*' in baseDir else baseDir
    baseDir = baseDir if baseDir.endswith('/') else baseDir + '/'

    for entry in os.scandir(baseDir):
        if entry.is_file():
            yield os.path.join(baseDir, entry.name)
        else:   # recurse needs from, otherwise generator object is returned
            yield from scanRecurse(entry.path + '/')


def chunks(lst:list, n:int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def temp(x):
    print('Process ', len(x))


MODEL_NAMES = ['doc2vec', 'universal', 'hugging', 'infer', 'ae', 'tfidf']

def main(src_path: str, model_names: list = MODEL_NAMES, num_cpus:int=1):
    src_path = '/mnt/datasets/Bahamas/B/5'
    print('start inserting documents embeddings on ', src_path)

    # all paths
    document_paths = list(scanRecurse(src_path))
    sub_lists = list(chunks(document_paths, int(len(document_paths)/num_cpus)))

    # process n_cpus sublists
    with Pool(processes=num_cpus) as pool:
        for model_name in model_names:  # function und diese parallisieren: run_process(doc_paths)
            proc_wrap = wrapper(model_name)
            print('started with model: ', model_name)
            pool.map(proc_wrap, sub_lists)
            print('finished model: ', model_name)

    print('finished inserting documents embeddings')


if __name__=='__main__':
    # python elasticSearch/test_pool.py /Users/klara/Documents/Uni/bachelorarbeit/data/0/ 2
    main(sys.argv[1], num_cpus=int(sys.argv[2]))