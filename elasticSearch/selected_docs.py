import os
import random

def select_rep_path(baseDir: str, num_items_per_dir:int=1):
    baseDir = baseDir.split('*')[0] if  '*' in baseDir else baseDir
    baseDir = baseDir if baseDir.endswith('/') else baseDir + '/'
    
    selected_docs = []
    for root, dirs, files in os.walk(baseDir):
        for dir in dirs:
            paths = os.listdir(os.path.join(root, dir))
            if len(paths) > 0:
                files = random.choices(paths, k=min(num_items_per_dir, len(paths)))
                for file in files:
                    file_path = os.path.join(root, dir, file)

                    if (os.path.isfile(file_path) and (not file.startswith('.'))):
                        selected_docs.append(file_path)

    return selected_docs
        
def main(baseDir:str):
    print(select_rep_path(baseDir, 10))

if __name__ == '__main__':
    baseDir = '/Users/klara/Documents/uni/'
    print(select_rep_path(baseDir, 2))