from user_interface.cli import *
from text_visualizations import visualize_texts


if __name__ == '__main__':
    args = arguments()

    file_paths = get_input_filepath(args)
    out_file = get_filepath(args, option='output')
    file_to_run = args.file_to_run
    print(file_to_run[0])

    if file_to_run[0] == 'visualize_texts.py':
        # python3 main.py 'visualize_texts.py' -i '/Users/klara/Downloads/SAC2-12.pdf' '/Users/klara/Downloads/SAC1-6.pdf'
        visualize_texts.main(file_paths, out_file)


    else:
        print('Error: No file to run.')
    