from cli import *
from pdf2image import convert_from_path

'''------Convert PDF files to PNG files-------
run this code by typing and altering the path:
    python3 code_file.py -i '/Users/klara/Downloads/SAC2-12.pdf'
    python3 code_file.py -i '/Users/klara/Downloads/SAC2-12.pdf' '/Users/klara/Downloads/SAC1-6.pdf'
    python3 code_file.py -d '/Users/klara/Downloads/*.pdf'
'''

def pdf_to_png(file_path: str, outpath: str = None, cluster: bool = False) -> None:
    '''
    :param file_path: path to file
    :param outpath: path to output folder; if not set, the output folder is the same as the input folder.
    :return: None

    This function converts a PDF file to a PNG file.
    The name of the PNG file is the same as the PDF file with .png instead of .pdf.
    '''
    for path in file_path:
        #print('5' + path)
        #print('6' + path.split('.')[0])
        file_name = (path.split('.')[0]).split('/')[-1]
        outpath = outpath if outpath else '/'.join(path.split('/')[:-1])
        if cluster:
            pages = convert_from_path(pdf_path=path, dpi=75, output_folder=outpath, output_file=file_name, fmt='png', poppler_path='/mnt/stud/work/kgutekunst/bsc-py/lib/python3.9/site-packages')
        else:
            pages = convert_from_path(pdf_path=path, dpi=75, output_folder=outpath, output_file=file_name, fmt='png')

if __name__ == '__main__':
    args = arguments()
    file_paths = get_input_filepath(args)
    outpath = get_output_filepath(args)

    #print('3' + file_paths[0])
    #print('4' + outpath)
    pdf_to_png(file_paths[0], outpath=outpath, cluster=args.cluster)