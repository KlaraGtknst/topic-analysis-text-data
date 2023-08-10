from cli import *
from pdf2image import convert_from_path

'''------Convert PDF files to PNG files-------
run this code by typing and altering the path:
    python3 code_file.py -i '/Users/klara/Downloads/SAC2-12.pdf'
    python3 code_file.py -i '/Users/klara/Downloads/SAC2-12.pdf' '/Users/klara/Downloads/SAC1-6.pdf'
    python3 code_file.py -d '/Users/klara/Downloads/*.pdf'
'''

if __name__ == '__main__':
    args = arguments()
    file_paths = get_input_filepath(args)

    for path in file_paths:
        pages = convert_from_path(path, dpi=75, output_folder='/'.join(path.split('/')[:-1]), output_file=path.split('.')[0], fmt='png')

        # TODO: 100 pdf a 64x64 pixel/dpi, in 10x10 matrix -> erkennen -> CNN Ähnlichkeiten erkennen
        # erste seite reivht

