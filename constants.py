EMBEDDINGS = ["doc2vec", "sim_docs_tfidf", "google_univ_sent_encoding", "huggingface_sent_transformer", "inferSent_AE"]
MODELS2EMB = {"doc2vec": "doc2vec", "tfidf": "sim_docs_tfidf", "universal": "google_univ_sent_encoding", "hugging": "huggingface_sent_transformer", "infer": "inferSent_AE", "ae": "inferSent_AE", "argmax_pca_cluster": "argmax_pca_cluster", "pca_optics_cluster": "pca_optics_cluster"}
DB_FIELDS = EMBEDDINGS + ["pca_optics_cluster", "path", "text", "image", "argmax_pca_cluster"]
CLIENT_ADDR = "http://localhost:9200"
MODEL_NAMES = ['doc2vec', 'universal', 'infer', 'ae', 'tfidf', 'hugging']
NUM_PCA_COMPONENTS = 13