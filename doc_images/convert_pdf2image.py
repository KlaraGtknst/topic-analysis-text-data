from user_interface.cli import *
import fitz 

'''------Convert PDF files to PNG files-------
run this code by typing and altering the path:
    python3 convert_pdf2image.py -i '/Users/klara/Documents/uni/bachelorarbeit/data/0/SAC29-14.pdf' -o '/Users/klara/Downloads/'
    python3 convert_pdf2image.py -d '/Users/klara/Documents/uni/bachelorarbeit/data/0/*.pdf' -o '/Users/klara/Documents/uni/bachelorarbeit/images/'
    python3 convert_pdf2image.py -d'/Users/klara/Downloads/*.pdf' -o '/Users/klara/Downloads/'
'''

def pdf_to_png(file_path: list, outpath: str = None) -> None:
    '''
    :param file_path: list of paths (to files); type has to be a list of strings
    :param outpath: path to output folder; if not set, the output folder is the same as the input folder.
    :return: None

    This function converts a PDF file to a PNG file.
    The name of the PNG file is the same as the PDF file with .png instead of .pdf.

    cf. for more information: 
    https://stackoverflow.com/questions/69643954/converting-pdf-to-png-with-python-without-pdf2image
    https://pymupdf.readthedocs.io/en/latest/pixmap.html#Pixmap.set_dpi
    '''
    for path in file_path:
        file_name = (path.split('.')[0]).split('/')[-1]
        outpath = outpath if outpath else '/'.join(path.split('/')[:-1])
        doc = fitz.open(path)  # open document
        pix = doc[0].get_pixmap()  # render first page to an image
        pix.set_dpi(75, 75) # image resolution
        pix.save(f"{outpath}/{file_name}.png")

def main(file_paths, outpath):

    pdf_to_png(file_paths, outpath=outpath)