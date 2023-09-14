from user_interface.cli import *
from doc_images import convert_pdf2image


if __name__ == '__main__':
    args = arguments()

    file_paths = get_input_filepath(args)
    out_file = get_filepath(args, option='output')
    image_src_path = get_filepath(args, option='image')
    file_to_run = args.file_to_run
    print(file_to_run[0])

    if file_to_run[0] == 'convert_pdf2image.py':
        # python3 main.py 'convert_pdf2image.py' -i '/Users/klara/Documents/uni/bachelorarbeit/data/0/SAC29-14.pdf' -o '/Users/klara/Downloads/'
        # python3 main.py 'convert_pdf2image.py' -d '/Users/klara/Documents/uni/bachelorarbeit/data/0/*.pdf' -o '/Users/klara/Documents/uni/bachelorarbeit/images/'
        # python3 main.py 'convert_pdf2image.py' -d'/Users/klara/Downloads/*.pdf' -o '/Users/klara/Downloads/'
        # pdf below does not work, pdf schief
        # python3 main.py 'convert_pdf2image.py' -i '/Users/klara/Documents/uni/bachelorarbeit/data/0/SAC34-38.pdf' -o '/Users/klara/Downloads/'
        convert_pdf2image.main(file_paths, out_file)