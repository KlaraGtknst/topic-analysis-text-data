from cli import *
from pdf2image import convert_from_path

'''------Convert PDF files to PNG files-------
run this code by typing and altering the path:
    python3 convert_pdf2image.py -i '/Users/klara/Documents/uni/bachelorarbeit/data/0/SAC29-14.pdf' -o '/Users/klara/Downloads/'
    python3 convert_pdf2image.py -d '/Users/klara/Documents/uni/bachelorarbeit/data/0/*.pdf' -o '/Users/klara/Documents/uni/bachelorarbeit/images/'
'''

def pdf_to_png(file_path: list, outpath: str = None, cluster: bool = False) -> None:
    '''
    :param file_path: list of paths (to files); type has to be a list of strings
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
            pages = convert_from_path(pdf_path=path, dpi=75, last_page=1, output_folder=outpath, output_file=file_name, fmt='png', poppler_path='/mnt/stud/work/kgutekunst/bsc-py/lib/python3.9/site-packages/poppler')
        else:
            pages = convert_from_path(pdf_path=path, dpi=75, last_page=1, output_folder=outpath, output_file=file_name, fmt='png')

if __name__ == '__main__':
    args = arguments()
    file_paths = get_input_filepath(args)
    outpath = get_filepath(args, option='output')

    #print('3' + file_paths[0])
    #print('4' + outpath)
    pdf_to_png(file_paths, outpath=outpath, cluster=args.cluster)