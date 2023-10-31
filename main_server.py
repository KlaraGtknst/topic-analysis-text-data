from constants import CLIENT_ADDR
from user_interface.cli import *
from doc_images import convert_pdf2image
from elasticSearch import db_elasticsearch, create_documents, create_database, insert_embeddings, insert_embeddings_copy, insert_clusters, selected_docs, test_pool, recursive_search
from doc_images.PCA import PCA_image_clustering
from text_embeddings.InferSent import own_word2vec
from text_embeddings.autoencoder import opt_architecture_ae
from evaluation import com_embs
from constants import MODEL_NAMES
import db_all_embs as db_all_embs
import num_pca_comp 

# srun --partition=main --mem=16g -n 1 --cpus-per-task=4 --pty /usr/sbin/sshd -D -f ~/sshd/sshd_config


if __name__ == '__main__':
    args = arguments()

    #file_paths = get_input_filepath(args)
    file_path = args.directory
    file_paths = file_path # TODO: temporary
    out_file = get_filepath(args, option='output')
    image_src_path = get_filepath(args, option='image')
    file_to_run = args.file_to_run
    client_addr = args.client_addr if args.client_addr else CLIENT_ADDR
    n_pools = args.n_pools
    model_names = get_model_names(args)

    if file_to_run[0] == 'convert_pdf2image.py':
        # python3 main_server.py 'convert_pdf2image.py' -i '/Users/klara/Documents/uni/bachelorarbeit/data/0/SAC29-14.pdf' -o '/Users/klara/Downloads/'
        # python3 main_server.py 'convert_pdf2image.py' -d '/Users/klara/Documents/uni/bachelorarbeit/data/*.pdf' -o '/Users/klara/Documents/uni/bachelorarbeit/images/'
        # python3 main_server.py 'convert_pdf2image.py' -d'/Users/klara/Downloads/*.pdf' -o '/Users/klara/Downloads/'
        # pdf below does not work, pdf schief
        # python3 main_server.py 'convert_pdf2image.py' -i '/Users/klara/Documents/uni/bachelorarbeit/data/0/SAC34-38.pdf' -o '/Users/klara/Downloads/'
        convert_pdf2image.main(file_paths, out_file)

    elif file_to_run[0] == 'db_elasticsearch.py':
        # python3 main_server.py 'db_elasticsearch.py' -d '/Users/klara/Documents/uni/bachelorarbeit/data/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/' -p 1
        # python3 main_server.py 'db_elasticsearch.py' -d '/Users/klara/Documents/uni/bachelorarbeit/data/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/' -p 1 -m 'universal'
        db_elasticsearch.main(src_path=file_path, image_src_path=image_src_path, client_addr=client_addr, n_pools=n_pools, model_names=model_names)

    elif file_to_run[0] == 'PCA_image_clustering.py':
        # python3 main_server.py 'PCA_image_clustering.py' -d '/Users/klara/Documents/uni/bachelorarbeit/data/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/'
        PCA_image_clustering.main(file_paths, image_src_path, outpath=out_file)

    elif file_to_run[0] == 'db_all_embs.py':
        # python3 main_server.py 'db_all_embs.py' -d '/Users/klara/Documents/uni/bachelorarbeit/data/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/' -p 1
        db_all_embs.main(file_path, image_src_path, client_addr, n_pools, model_names=model_names)

    elif file_to_run[0] == 'own_word2vec.py':
        # python3 main_server.py 'own_word2vec.py' -d '/Users/klara/Documents/uni/bachelorarbeit/data/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/' -p 1
        own_word2vec.main(file_paths)

    elif file_to_run[0] == 'create_database.py':
        # python3 main_server.py 'create_database.py' -d '/Users/klara/Documents/uni/bachelorarbeit/data/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/' -p 1
        create_database.main(src_path=file_path, client_addr=client_addr)

    elif file_to_run[0] == 'create_documents.py':
        # python3 main_server.py 'create_documents.py' -d '/Users/klara/Documents/uni/bachelorarbeit/data/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/' -p 1
        create_documents.main(src_path=file_path, client_addr=client_addr, num_cpus=n_pools)

    elif file_to_run[0] == 'insert_embeddings.py':
        # python3 main_server.py 'insert_embeddings.py' -d '/Users/klara/Documents/uni/bachelorarbeit/data/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/' -p 1
        insert_embeddings.main(src_path=file_path, client_addr=client_addr, model_names=model_names, num_cpus=n_pools)

    elif file_to_run[0] == 'insert_clusters.py':
        # python3 main_server.py 'insert_clusters.py' -d '/Users/klara/Documents/uni/bachelorarbeit/data/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/' -p 1
        insert_clusters.main(src_path=file_path, client_addr=client_addr, image_src_path=image_src_path)

    elif file_to_run[0] == 'opt_architecture_ae.py':
        # python3 main_server.py 'opt_architecture_ae.py' -d '/Users/klara/Documents/uni/bachelorarbeit/data/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/' -p 1
        opt_architecture_ae.main(file_path, num_cpus=n_pools)

    elif file_to_run[0] == 'selected_docs.py':
        # python3 main_server.py 'selected_docs.py' -d '/Users/klara/Documents/Uni/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/' -p 1
        selected_docs.main(file_path)

    elif file_to_run[0] == 'test_pool.py':
        # python3 main_server.py 'test_pool.py' -d '/Users/klara/Documents/uni/bachelorarbeit/data/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/' -p 1
        test_pool.main(src_path=file_path, model_names=model_names, num_cpus=n_pools)

    elif file_to_run[0] == 'recursive_search.py':
        # python3 main_server.py 'recursive_search.py' -d '/Users/klara/Documents/uni/bachelorarbeit/data/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/' -p 1
        recursive_search.main(baseDir=file_path)

    elif file_to_run[0] == 'num_pca_comp.py':
        # python3 main_server.py 'num_pca_comp.py' -d '/Users/klara/Documents/uni/bachelorarbeit/data/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/' -p 1
        num_pca_comp.main()

    elif file_to_run[0] == 'com_embs.py':
        # python3 main_server.py 'com_embs.py' -d '/Users/klara/Documents/uni/bachelorarbeit/data/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/' -p 1
        com_embs.main(baseDir=file_path)

    elif file_to_run[0] == 'insert_embeddings_copy.py':
        # python3 main_server.py 'insert_embeddings_copy.py' -d '/Users/klara/Documents/uni/bachelorarbeit/data/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/' -p 1
        insert_embeddings_copy.main(src_path=file_path, client_addr=client_addr, model_names=model_names, image_src_path=image_src_path)




        