#from OpenCV import *
from sklearn import decomposition
from math import sqrt
# own modules
from read_pdf import *
from cli import *
from pdf_matrix import *
from query_documents_tfidf import *
from universal_sent_encoder_tensorFlow import *
from hugging_face_sentence_transformer import *

'''------search in existing database-------
run this code by typing and altering the path:
    python3 image_clustering.py -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf' -D '/Users/klara/Documents/Uni/bachelorarbeit/images/images/*.png'
    python3 image_clustering.py -d '/Users/klara/Documents/Uni/bachelorarbeit/data/0/*.pdf' -D '/Users/klara/Downloads/*.png'
'''



if __name__ == '__main__':
    args = arguments()
    src_paths = get_input_filepath(args)
    image_src_path = get_filepath(args, option='image')

    IMG_SIZE = 600

    # TODO: center the images, cf. https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html#sphx-glr-auto-examples-decomposition-plot-faces-decomposition-py
    preprocessed_images = np.array([np.reshape(a=
                                      cv2.normalize(
        cv2.resize(
            cv2.imread(img, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE)), 
            None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            , newshape=IMG_SIZE**2) 
            for img in glob.glob(image_src_path)])
    
    # Global centering (focus on one feature, centering all samples)
    preprocessed_images -= np.mean(preprocessed_images, axis=0)
    # Local centering (focus on one sample, centering all features)
    preprocessed_images -= preprocessed_images.mean(axis=1).reshape(len(preprocessed_images), -1)

    
    print('shape of all data: ', preprocessed_images.shape)
    for img in preprocessed_images[:2]:
        plt.imshow(img.reshape(IMG_SIZE,IMG_SIZE), cmap='gray')
        plt.show()
    
    # FIXME: PCA needs more samples than features, cf. https://stackoverflow.com/questions/51040075/why-sklearn-pca-needs-more-samples-than-new-featuresn-components
    # FIXME: otherwise the number of new faetures will be way smaller than the original number of features
    # number of components to keep: https://www.mikulskibartosz.name/pca-how-to-choose-the-number-of-components/
    pca = decomposition.PCA(n_components=0.95, whiten=True)
    #print(preprocessed_images[0].shape)
    pca_img = pca.fit_transform(preprocessed_images)
    print('PCA components: ', pca.components_.shape)
    print('PCA return shape: ', pca_img.shape)

    # plot eigenvectors as images
    for img in pca.components_[:2]:
        plt.imshow(img.reshape(IMG_SIZE,IMG_SIZE), cmap='gray')
        plt.show()

    # TODO: identify clusters by similar weights (linear combination) of eigenvectors

    # TODO: visulaize the compressed data as an image
    for img in pca_img[:2]:
        print('Image shape: ', img.shape)
        dim = int(sqrt(len(img)))
        print('Image dim: ', dim)
        plt.imshow(np.reshape(img[:dim**2], dim, dim), cmap='gray')
        plt.show()

    