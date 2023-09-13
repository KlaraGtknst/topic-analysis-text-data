from user_interface.cli import *
from text_visualizations import visualize_texts
from text_embeddings.preprocessing import read_pdf
from text_embeddings.InferSent import infer_pretrained
from text_embeddings import universal_sent_encoder_tensorFlow, save_models
from text_embeddings import hugging_face_sentence_transformer
from elasticSearch import db_elasticsearch
from elasticSearch.queries import query_documents_tfidf, query_database
from doc_images import pdf_matrix, convert_pdf2image
from doc_images.PCA import PCA_image_clustering
from test_functions import word_embeddings
from elasticSearch.queries import query_database_inferSent
#from user_interface import user_interface
#from doc_images.AE import AE_image_clustering


if __name__ == '__main__':
    args = arguments()

    file_paths = get_input_filepath(args)
    out_file = get_filepath(args, option='output')
    image_src_path = get_filepath(args, option='image')
    file_to_run = args.file_to_run
    print(file_to_run[0])

    if file_to_run[0] == 'visualize_texts.py':
        # python3 main.py 'visualize_texts.py' -i '/Users/klara/Downloads/SAC2-12.pdf' '/Users/klara/Downloads/SAC1-6.pdf'
        # python3 main.py 'visualize_texts.py '-d '/Users/klara/Downloads/*.pdf'
        visualize_texts.main(file_paths, out_file)

    elif file_to_run[0] == 'read_pdf.py':
        # python3 main.py 'read_pdf.py' -i '/Users/klara/Downloads/SAC2-12.pdf'
        # python3 main.py 'read_pdf.py' -d '/Users/klara/Downloads/*.pdf'
        read_pdf.main(file_paths)

    elif file_to_run[0] == 'universal_sent_encoder_tensorFlow.py':
        # python3  main.py 'universal_sent_encoder_tensorFlow.py' -d '/Users/klara/Downloads/*.pdf' -o '/Users/klara/Downloads/'
        # python3  main.py 'universal_sent_encoder_tensorFlow.py' -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf' -o '/Users/klara/Downloads/'
        universal_sent_encoder_tensorFlow.main(file_paths, out_file)

    elif file_to_run[0] == 'hugging_face_sentence_transformer.py':
        # python3  main.py 'hugging_face_sentence_transformer.py' -d '/Users/klara/Downloads/*.pdf' -o '/Users/klara/Developer/Uni/hugging_face_sentence_transformer'
        # python3  main.py 'hugging_face_sentence_transformer.py' -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf' -o '/Users/klara/Developer/Uni/hugging_face_sentence_transformer'
        hugging_face_sentence_transformer.main(file_paths, out_file)

    elif file_to_run[0] == 'infer_pretrained.py':
        # python3 main.py 'infer_pretrained.py' -d '/Users/klara/Downloads/*.pdf' -o '/Users/klara/Downloads/'
        # python3 main.py 'infer_pretrained.py' -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf' -o '/Users/klara/Downloads/'
        infer_pretrained.main(file_paths, out_file)

    elif file_to_run[0] == 'db_elasticsearch.py':
        # python3 main.py 'db_elasticsearch.py' -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/'
        db_elasticsearch.main(file_paths, image_src_path)

    elif file_to_run[0] == 'query_documents_tfidf.py':
        # python3  main.py 'query_documents_tfidf.py' -d '/Users/klara/Downloads/*.pdf'
        # python3  main.py 'query_documents_tfidf.py' -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf'
        query_documents_tfidf.main(file_paths)

    elif file_to_run[0] == 'query_database.py':
        # python3  main.py 'query_database.py' -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/'
        query_database.main(file_paths, image_src_path)

    elif file_to_run[0] == 'pdf_matrix.py':
        # python3  main.py 'pdf_matrix.py' --number 5 -d '/Users/klara/Documents/uni/bachelorarbeit/images/*.png' -o '/Users/klara/Downloads/'
        pdf_matrix.main(file_paths, image_src_path, dim=args.number)

    elif file_to_run[0] == 'convert_pdf2image.py':
        # python3 main.py 'convert_pdf2image.py' -i '/Users/klara/Documents/uni/bachelorarbeit/data/0/SAC29-14.pdf' -o '/Users/klara/Downloads/'
        # python3 main.py 'convert_pdf2image.py' -d '/Users/klara/Documents/uni/bachelorarbeit/data/0/*.pdf' -o '/Users/klara/Documents/uni/bachelorarbeit/images/'
        # python3 main.py 'convert_pdf2image.py' -d'/Users/klara/Downloads/*.pdf' -o '/Users/klara/Downloads/'
        # pdf below does not work, pdf schief
        # python3 main.py 'convert_pdf2image.py' -i '/Users/klara/Documents/uni/bachelorarbeit/data/0/SAC34-38.pdf' -o '/Users/klara/Downloads/'
        convert_pdf2image.main(file_paths, out_file)

    elif file_to_run[0] == 'PCA_image_clustering.py':
        # python3 main.py 'PCA_image_clustering.py' -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/*.png'
        # python3 main.py 'PCA_image_clustering.py' -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf' -D '/Users/klara/Downloads/*.png'
        PCA_image_clustering.main(src_paths=file_paths, image_src_path=image_src_path, outpath=out_file)

    # test
    elif file_to_run[0] == 'word_embeddings.py':
        # python3 main.py 'word_embeddings.py' -i '/Users/klara/Downloads/SAC2-12.pdf' '/Users/klara/Downloads/SAC1-6.pdf'
        # python3 main.py 'word_embeddings.py' -d '/Users/klara/Downloads/*.pdf'
        word_embeddings.main(file_paths=file_paths)

    elif file_to_run[0] == 'query_database_inferSent.py':
        # python3 main.py 'query_database_inferSent.py' -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf'
        query_database_inferSent.main(file_paths, out_file)

    elif file_to_run[0] == 'user_interface.py':
        # python3 main.py 'user_interface.py' -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf'
        #user_interface.main()
        print('temporarily not available')

    elif file_to_run[0] == 'save_models.py':
        # python3 main.py 'save_models.py' -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf'
        save_models.main()

    else:
        print('Error: No file to run.')
    