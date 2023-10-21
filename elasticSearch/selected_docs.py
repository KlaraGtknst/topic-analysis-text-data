import os
import random

def select_rep_path(baseDir: str):
    print(baseDir)
    baseDir = baseDir.split('*')[0] if  '*' in baseDir else baseDir
    baseDir = baseDir if baseDir.endswith('/') else baseDir + '/'
    
    selected_docs = []
    for root, dirs, files in os.walk(baseDir):
        for dir in dirs:
            file = random.choice(os.listdir(os.path.join(root, dir)))
            file_path = os.path.join(root, dir, file)

            if (os.path.isfile(file_path) and (not file.startswith('.'))):
                selected_docs.append(file_path)

    return selected_docs
        


if __name__ == '__main__':
    baseDir = '/Users/klara/Documents/uni/'
    print(select_rep_path(baseDir))