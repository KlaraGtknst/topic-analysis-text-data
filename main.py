from user_interface.cli import *
from text_visualizations import visualize_texts


if __name__ == '__main__':
    args = arguments()

    file_paths = get_input_filepath(args)
    out_file = get_filepath(args, option='output')

    visualize_texts.main(file_paths, out_file)