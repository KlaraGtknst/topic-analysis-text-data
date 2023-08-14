from read_pdf import *
from cli import *
import seaborn as sns


'''------Code to encode documents as sentence embeddings using pretrained models (InferSent)-------
The code below is based on:
    https://github.com/facebookresearch/InferSent

run this code by typing and altering the path:
    python3 infer_pretrained.py -d '/Users/klara/Downloads/*.pdf' -o '/Users/klara/Downloads/'
    python3 infer_pretrained.py -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf' -o '/Users/klara/Downloads/'
'''


if __name__ == '__main__':
    args = arguments()
    file_paths = get_input_filepath(args)
    outpath = get_filepath(args, option='output')