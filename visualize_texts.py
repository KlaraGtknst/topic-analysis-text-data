from read_pdf import *
from cli import *
from wordcloud import WordCloud
import cv2 

'''------Code for visualization-------
run this code by typing and altering the path:
    python3 visualize_texts.py -i '/Users/klara/Downloads/SAC2-12.pdf' -o '/Users/klara/Downloads/'
    python3 visualize_texts.py -i '/Users/klara/Downloads/SAC2-12.pdf' '/Users/klara/Downloads/SAC1-6.pdf'
    python3 visualize_texts.py -d '/Users/klara/Downloads/*.pdf'
'''

def term_frequency(tokens: list, file_name: str, outpath: str = None) -> None:
    '''
    :param tokens: list of tokens
    :return: None

    This function plots the term frequency of the tokens.
    '''
    plt.figure(figsize=(15, 10))
    plt.hist(tokens, bins=100, orientation='vertical', color='green')
    plt.xticks(rotation=90, fontsize=5)
    title = 'Term frequency in ' + file_name
    plt.title(title)
    if outpath:
        plt.savefig(outpath + '/' + title, format="pdf", bbox_inches="tight")
    plt.show()


def word_cloud(tokens: list, file_name: str, outpath: str = None) -> None:
    '''
    :param tokens: list of tokens
    :param file_name: name of the file
    :return: None

    This function plots a word cloud of the tokens.
    cf. https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
    '''
    # possible to add stopwords to initiation of worldcloud
    wordcloud = WordCloud(width=800, height=500, random_state=21, contour_width=3, max_font_size=110, background_color='white', max_words=5000).generate(','.join(tokens))
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud, interpolation="bilinear") # displays image, interpolation: smoother image
    plt.axis('off')
    title = 'Word cloud in ' + file_name
    plt.title(title)
    if outpath:
        plt.savefig(outpath + '/' + title, format="pdf", bbox_inches="tight")
    plt.show()

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

def create_image_matrix(dim: int, input_files: str, output_path: str = None) -> None:
    '''
    :param dim: dimension of the image matrix, i.e. number of images in one row/ column
    :param input_files: list of input files; must have at least dim*dim elements
    :param output_path: path to save the image
    :return: None

    This function creates a matrix of the first dim x dim images from input_files.
    '''
    # python3 visualize_texts.py -d '/Users/klara/Documents/uni/bachelorarbeit/images/*.png'
    if len(input_files) < dim*dim:
        raise ValueError('Input files must have at least dim*dim elements.')
    fig, axs = plt.subplots(10,10, figsize=(10, 10))
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

    create_image_matrix(dim=10, input_files=file_paths, output_path=out_file)

    '''for path in file_paths:
        text = pdf_to_str(path)

        tokens = tokenize(text)
        filtered_tokens = remove_stop_words(tokens)
        stemmed_filtered_tokens = stemming(filtered_tokens)

        # visualize the texts
        term_frequency(stemmed_filtered_tokens, file_name=path.split('/')[-1], outpath=out_file)
        word_cloud(stemmed_filtered_tokens, file_name=path.split('/')[-1], outpath=out_file)'''
