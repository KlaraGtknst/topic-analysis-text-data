from cli import *
from pdf2image import convert_from_path

'''------Convert PDF files to PNG files-------
run this code by typing and altering the path:
    python3 code_file.py -i '/Users/klara/Downloads/SAC2-12.pdf'
    python3 code_file.py -i '/Users/klara/Downloads/SAC2-12.pdf' '/Users/klara/Downloads/SAC1-6.pdf'
    python3 code_file.py -d '/Users/klara/Downloads/*.pdf'
'''

def pdf_to_png(file_path: str, outpath: str = None) -> None:
    '''
    :param file_path: path to file
    :param outpath: path to output folder; if not set, the output folder is the same as the input folder.
    :return: None

    This function converts a PDF file to a PNG file.
    The name of the PNG file is the same as the PDF file with .png instead of .pdf.
    '''
    for path in file_paths:
        file_name = (path.split('.')[0]).split('/')[-1]
        outpath = outpath if outpath else '/'.join(path.split('/')[:-1])
        pages = convert_from_path(pdf_path=path, dpi=75, output_folder=outpath, output_file=file_name, fmt='png')

if __name__ == '__main__':
    args = arguments()
    file_paths = get_input_filepath(args)
    outpath = get_output_filepath(args)

    pdf_to_png(file_paths, outpath=outpath)

    # TODO: 100 pdf a 64x64 pixel/dpi, in 10x10 matrix -> erkennen -> CNN Ähnlichkeiten erkennen
    # erste seite reivht

