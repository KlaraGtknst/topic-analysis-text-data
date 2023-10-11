import io
from PIL import Image
from matplotlib import image
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from text_embeddings.preprocessing.read_pdf import *
from user_interface.cli import *
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer

'''------Code for visualization-------
run this code by typing and altering the path:
    python3 text_visualizations/visualize_texts.py -i '/Users/klara/Downloads/SAC2-12.pdf' -o '/Users/klara/Downloads/'
    python3 visualize_texts.py -i '/Users/klara/Downloads/SAC2-12.pdf' '/Users/klara/Downloads/SAC1-6.pdf'
    python3 visualize_texts.py -d '/Users/klara/Downloads/*.pdf'
'''

def term_frequency(tokens: list, file_name: str, return_img: bool = False, outpath: str = None) -> Image:
    '''
    :param tokens: list of tokens
    :param file_name: name of the file
    :param outpath: path to save the term frequency
    :return: None; if return_png is true, the term frequency will be returned as PIL image

    This function plots the term frequency of the tokens.
    '''
    fig = Figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)
    ax.hist(tokens, bins=100, orientation='vertical', color='green')
    ax.tick_params(axis='x', labelrotation=90, labelsize=5)
    title = 'Term frequency in ' + file_name
    ax.set_title(title)
    if return_img:
        canvas.draw()
        return Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb()) # PIL image
    if outpath:
        plt.savefig(outpath + '/' + title, format="pdf", bbox_inches="tight")
    return fig


def word_cloud(tokens: list, file_name: str, outpath: str = None, return_img:bool=False) -> Image:
    '''
    :param tokens: list of tokens
    :param file_name: name of the file
    :param outpath: path to save the wordcloud
    :return: None; if return_png is true, the wordcloud will be returned as PIL image

    This function plots a word cloud of the tokens.
    cf. https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
    '''
    try:
        # lemmatize tokens
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        # worldcloud
        wordcloud = WordCloud(width=800, height=500, random_state=21, contour_width=3, max_font_size=110, background_color='white', max_words=5000).generate(','.join(tokens))
        if return_img:
            return wordcloud.to_image()
        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud, interpolation="bilinear") # displays image, interpolation: smoother image
        plt.axis('off')
        title = 'Word cloud in ' + file_name
        plt.title(title)
        if outpath:
            plt.savefig((outpath + '/' + title.replace(' ', '_') + '.pdf'), format="pdf")
        plt.show()
    except ValueError:
        print('Error: No words to plot.')

def preprocess_doc(path:str) -> list:
    '''
    :param path: path to one document
    :return: list of preprocessed (filtered and stemmed) tokens
    '''
    text = pdf_to_str(path)
    tokens = tokenize(text)
    filtered_tokens = remove_stop_words(tokens)
    stemmed_filtered_tokens = stemming(filtered_tokens)
    return stemmed_filtered_tokens

def image_to_byte_array(image: image, format: str = 'png'):
    result = io.BytesIO()
    image.save(result, format=format)
    result = result.getvalue()
    return result

def get_one_visualization_from_text(option:str, texts:list) -> image:
    '''
    :param option: 'wordcloud' or 'term_frequency'
    :param texts: list of texts
    :return: Image object
    '''
    tokens = []
    for text in texts:
        tokens.extend(stemming(remove_stop_words(tokenize(text))))
    if option == 'wordcloud':
        img = word_cloud(tokens, file_name='file', outpath=None, return_img=True)
        return img
    elif option == 'term_frequency':
        img = term_frequency(tokens, file_name='file', outpath=None, return_img=True)
        return img
    else:
        print('Error: No valid option chosen.')


def get_one_visualization(option:str, paths:list, outpath:str=None) -> None:
    '''
    :param option: 'wordcloud' or 'term_frequency'
    :param path: paths to documents
    :param outpath: path to save the visualization; if None, the visualization will be shown
    :return: None

    This function plots a word cloud of the tokens of the document.
    '''
    tokens = []
    first_doc = paths[0].split('/')[-1]
    for path in paths:
        tokens.extend(preprocess_doc(path))
    if option == 'wordcloud':
        img = word_cloud(tokens, file_name= first_doc if (len(paths) == 1) else f'multiple docs similar to {first_doc}', outpath=outpath, return_img=True)
        return image_to_byte_array(img)
    elif option == 'term_frequency':
        term_frequency(tokens, file_name=first_doc if (len(paths) == 1) else f'multiple docs similar to {first_doc}', outpath=outpath)
        return None
    else:
        print('Error: No valid option chosen.')

def main(file_paths, out_file):

    for path in file_paths:
        text = pdf_to_str(path)

        tokens = tokenize(text)
        filtered_tokens = remove_stop_words(tokens)
        stemmed_filtered_tokens = stemming(filtered_tokens)

        # visualize the texts
        term_frequency(stemmed_filtered_tokens, file_name=path.split('/')[-1], outpath=out_file)
        word_cloud(stemmed_filtered_tokens, file_name=path.split('/')[-1], outpath=out_file)