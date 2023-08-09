from read_pdf import *
from cli import *
from wordcloud import WordCloud

'''------Code for visualization-------
run this code by typing and altering the path:
    python3 visualize_texts.py -i '/Users/klara/Downloads/SAC2-12.pdf'
    python3 visualize_texts.py -i '/Users/klara/Downloads/SAC2-12.pdf' '/Users/klara/Downloads/SAC1-6.pdf'
    python3 visualize_texts.py -d '/Users/klara/Downloads/*.pdf'
'''

def term_frequency(tokens: list, file_name: str) -> None:
    '''
    :param tokens: list of tokens
    :return: None

    This function plots the term frequency of the tokens.
    '''
    plt.figure(figsize=(15, 10))
    plt.title('Term frequency in ' + file_name)
    plt.hist(tokens, bins=100, orientation='vertical', color='green')
    plt.xticks(rotation=90, fontsize=5)
    plt.show()


def word_cloud(tokens: list, file_name: str) -> None:
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
    plt.title('Word cloud in ' + file_name)
    plt.show()


if __name__ == '__main__':
    args = arguments()
    print(args)

    file_paths = get_filepath(args)

    for path in file_paths:
        text = pdf_to_str(path)

        tokens = tokenize(text)
        filtered_tokens = remove_stop_words(tokens)
        stemmed_filtered_tokens = stemming(filtered_tokens)

        # visualize the texts
        term_frequency(stemmed_filtered_tokens, file_name=path.split('/')[-1])
        word_cloud(stemmed_filtered_tokens, file_name=path.split('/')[-1])
