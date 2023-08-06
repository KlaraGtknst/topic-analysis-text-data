import glob
from pdf2image import convert_from_path

if __name__ == '__main__':
    for path in glob.glob('/Users/klara/Downloads/*.pdf'):
        pages = convert_from_path(path, dpi=500, output_folder='/'.join(path.split('/')[:-1]), output_file=path.split('.')[0], fmt='png')

