from read_pdf import *
from wordcloud import WordCloud

def term_frequency(tokens, file_name):
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


def word_cloud(tokens, file_name):
    '''
    :param tokens: list of tokens
    :param file_name: name of the file
    :return: None

    This function plots a word cloud of the tokens.
    cf. https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
    '''
    print(tokens)
    wordcloud = WordCloud(width=800, height=500, random_state=21, contour_width=3, max_font_size=110, background_color='white', max_words=5000).generate(','.join(tokens))
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title('Word cloud in ' + file_name)
    plt.show()

if __name__ == '__main__':
    print('Visualize texts')
    for path in glob.glob('/Users/klara/Downloads/*.pdf'):
        text = pdf_to_str(path)

        tokens = tokenize(text)
        filtered_tokens = remove_stop_words(tokens)
        stemmed_filtered_tokens = stemming(filtered_tokens)
        #term_frequency(stemmed_filtered_tokens, file_name=path.split('/')[-1])
        word_cloud(stemmed_filtered_tokens, file_name=path.split('/')[-1])
