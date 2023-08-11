from read_pdf import *
from cli import *
import cv2 

'''------Code for matrix of PDF layouts-------
run this code by typing and altering the path:
    python3 pdf_matrix.py --number 5 -d '/Users/klara/Documents/uni/bachelorarbeit/images/*.png' -o '/Users/klara/Downloads/'
'''


def alter_axes(ax: plt.axes) -> None:
    '''
    :param ax: axis
    :return: None
    
    This function alters the axis (makes ticks and spines/ frames invisible) of the plot.'''
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def create_image_matrix(input_files: str, dim: int= 10, output_path: str = None) -> None:
    '''
    :param dim: dimension of the image matrix, i.e. number of images in one row/ column
    :param input_files: list of input files; must have at least dim*dim elements
    :param output_path: path to save the image
    :return: None

    This function creates a matrix of the first dim x dim images from input_files.
    '''
    if len(input_files) < dim*dim:
        raise ValueError('Input files must have at least dim*dim elements.')
    fig, axs = plt.subplots(dim, dim, figsize=(dim, dim))
    fig.subplots_adjust(hspace = .00, wspace= .00)
    for i, img in enumerate(input_files[:dim*dim]):
        image = cv2.imread(img, cv2.IMREAD_COLOR)
        ax = fig.add_subplot(dim, dim, i+1)
        plt.imshow(image)
        axs[i//dim, i%dim].axis('off')
        alter_axes(ax)
    plt.axis('off')
    if output_path:
        plt.savefig(output_path + 'image_matrix.pdf', format="pdf", bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    args = arguments()

    file_paths = get_input_filepath(args)
    out_file = get_output_filepath(args)

    create_image_matrix(dim=args.number, input_files=file_paths, output_path=out_file)