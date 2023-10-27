import os
import random 


def scanRecurse(baseDir: str):
    baseDir = baseDir.split('*')[0] if  '*' in baseDir else baseDir

    for entry in os.scandir(baseDir):
        if entry.is_file():
            yield os.path.join(baseDir, entry.name)
        else:   # recurse needs from, otherwise generator object is returned
            yield from scanRecurse(entry.path + '/')


def chunks(lst:list, n:int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def save_docs_to_file(baseDir: str, num_docs:int, res_path:str='results/'):
    all_paths = list(scanRecurse(baseDir=baseDir))
    if len(all_paths) < num_docs:
        return
    with open(res_path + 'selected_doc_paths.txt', 'w') as f:
        for line in random.sample(all_paths, num_docs):
            f.write(line)
            f.write('\n')


def main(baseDir):
    res_path = 'results/'
    server_model_path = '/mnt/stud/home/kgutekunst/results/'
    if os.path.exists(server_model_path):
        res_path = server_model_path
    save_docs_to_file(baseDir=baseDir, num_docs=2000, res_path=res_path)