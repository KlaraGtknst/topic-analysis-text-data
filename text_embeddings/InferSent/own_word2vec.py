import glob
from gensim.models import Word2Vec

from elasticSearch.queries.query_documents_tfidf import get_docs_from_file_paths

def post_process_file(w2v_path):
    # only keep entries consisting of characters, whitespace and numbers
    with open(w2v_path, 'r+') as fp:
        # read an store all lines into list
        lines = fp.readlines()
        # move file pointer to the beginning of a file
        fp.seek(0)
        # truncate the file
        fp.truncate()

        for line in lines:
            if len(line.split(' ', 1)) > 1:
                fp.write(line)

def main(file_paths):
    model_path = '/Users/klara/Developer/Uni/bahamas_word2vec/'
    docs = get_docs_from_file_paths(file_paths)
    model = Word2Vec(docs, vector_size=300)
    model.wv.save_word2vec_format(model_path + 'bahamas_w2v.txt', binary=False)
    post_process_file(w2v_path=model_path + 'bahamas_w2v.txt')

if __name__ == '__main__':
    file_paths = glob.glob('/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf')
    main(file_paths)