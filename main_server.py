from user_interface.cli import *
from doc_images import convert_pdf2image
#from elasticSearch import db_elasticsearch

# srun --partition=main --mem=128g -n 1 --cpus-per-task=32 --pty /usr/sbin/sshd -D -f ~/sshd/sshd_config


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

    elif file_to_run[0] == 'db_elasticsearch.py':
        # python3 main.py 'db_elasticsearch.py' -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/'
        #db_elasticsearch.main(file_paths, image_src_path)
        print('todo')