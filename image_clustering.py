#from OpenCV import *
import math
from sklearn import decomposition
from math import sqrt

from sklearn.cluster import KMeans
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

def preprocess_images(src_path: str, img_size: int)-> np.ndarray:
    '''
    :param src_path: path to the images to be preprocessed
    :param img_size: single dimension of the output images (resized to quadratic images)
    :return: numpy array of preprocessed images
    
    This function preprocesses the images to be clustered.
    It generates the paths for all images, reads, resizes, normalizes and centers them.
    More information in:
    https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html#sphx-glr-auto-examples-decomposition-plot-faces-decomposition-py
    '''
    image_paths = glob.glob(image_src_path)
    preprocessed_images = np.array([np.reshape(a=cv2.normalize(cv2.resize(cv2.imread(img, cv2.IMREAD_GRAYSCALE), (img_size, img_size)), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F), newshape=IMG_SIZE**2) for img in image_paths])
    # Global centering (focus on one feature, centering all samples)
    preprocessed_images_centered = preprocessed_images - np.mean(preprocessed_images, axis=0)
    # Local centering (focus on one sample, centering all features)
    preprocessed_images_centered -= preprocessed_images_centered.mean(axis=1).reshape(len(preprocessed_images_centered), -1)
    return preprocessed_images_centered

def plot_grey_images(image: list, title: str = None, save: bool = False) -> None:
    '''
    :param image: images to be plotted; list of greyvalues; if the image is a 1d array it will be reshaped to be displayable
    :param title: title of the plot
    :return: None
    '''
    if len(image.shape) == 1:
        image_size = int(sqrt(image.shape[0]))
        image = image[:image_size**2].reshape(image_size, image_size)
    plt.imshow(image, cmap='gray')
    if title:
        plt.title(title)
    if save:
        plt.savefig(title + '.pdf', format="pdf", bbox_inches="tight")
    plt.show()


def create_pca_df(image_src_path: list, pca_weights: list) -> pd.DataFrame:
    '''
    :param image_src_path: paths to the images; used as index
    :param pca_weights: weights/ factors of the pca components
    :return: dataframe of the pca weights and the corresponding image paths
    
    This function creates a dataframe of the pca weights and the corresponding image paths as index.
    For more information about inserting a list in a cell see:
    https://stackoverflow.com/questions/48000225/must-have-equal-len-keys-and-value-when-setting-with-an-iterable
    '''
    pca_df = pd.DataFrame({'path': image_src_path, 'pca_weights': [0 for i in range(len(image_src_path))]})
    pca_df.set_index('path', inplace=True)
    for i in range(len(image_src_path)):
       pca_df.loc[[image_src_path[i]], 'pca_weights'] = pd.Series([pca_img[i]], index=pca_df.index[[i]])
    return pca_df

def get_sample_doc_img_per_cluster(pca_df: pd.DataFrame, num_cluster: int) -> list:
    '''
    :param pca_df: dataframe of the pca weights and the corresponding image paths as index
    :param cluster: number of clusters
    :return: list of image paths of the sample per cluster
    
    This function returns the image paths of the samples of the clusters.
    '''
    return [pca_df[pca_df['cluster'] == i].sample(1).index.values[0] for i in range(num_cluster)]

def visualize_class_distr(pca_df: pd.DataFrame, num_cluster: int) -> None:
    '''
    :param pca_df: dataframe of the pca weights and the corresponding image paths as index
    :param cluster: number of clusters
    :return: None
    
    This function visualizes the class distribution of the clusters.
    For more information about inserting counts into a bar chart see:
    https://www.geeksforgeeks.org/adding-value-labels-on-a-matplotlib-bar-chart/
    '''
    pca_df['cluster'].value_counts().plot(kind='bar', title='Class distribution', xlabel='Class', ylabel='Count')
    cluster_counts = pca_df['cluster'].value_counts().sort_values(inplace=False, ascending=False).values
    for i in range(len(pca_df['cluster'].value_counts())):
            plt.text(i,cluster_counts[i], cluster_counts[i])
    plt.show()

if __name__ == '__main__':
    args = arguments()
    src_paths = get_input_filepath(args)
    image_src_path = get_filepath(args, option='image')
    outpath = get_filepath(args, option='output')

    IMG_SIZE = 600
    NUM_CLASSES = 4

    preprocessed_images = preprocess_images(image_src_path, IMG_SIZE)

    # plot preprocessed images
    for img in preprocessed_images[:2]:
        plot_grey_images(img)

    # PCA needs more samples than features, cf. https://stackoverflow.com/questions/51040075/why-sklearn-pca-needs-more-samples-than-new-featuresn-components
    # otherwise the number of new faetures will be way smaller than the original number of features
    # number of components to keep: https://www.mikulskibartosz.name/pca-how-to-choose-the-number-of-components/
    pca = decomposition.PCA(n_components=2, whiten=True)
    pca_img = pca.fit_transform(preprocessed_images)
    
    i = 1
    for img in pca.components_[:2]:
        plot_grey_images(img, title='eigenvector number ' + str(i))
        i += 1

    # PCA dataframe
    pca_df = create_pca_df(glob.glob(image_src_path), pca_img)

    # cluster the images
    kmeans = KMeans(n_clusters=NUM_CLASSES, random_state=0, n_init="auto").fit(pca_df['pca_weights'].to_list())
    pca_df['cluster'] = kmeans.labels_

    # get sample images per cluster
    sample_doc_img_per_cluster = get_sample_doc_img_per_cluster(pca_df, NUM_CLASSES)
    create_image_matrix(dim=int(math.sqrt(NUM_CLASSES)), input_files=sample_doc_img_per_cluster, output_path=outpath)
    for i in range(len(sample_doc_img_per_cluster)):
        img = sample_doc_img_per_cluster[i]
        plot_grey_images(cv2.imread(img, cv2.IMREAD_GRAYSCALE), title='example image of cluster ' + str(i))
    
    # visualize class distribution
    visualize_class_distr(pca_df, NUM_CLASSES)

    # visualize factors of pca components/ results of pca
    for i in range(len(pca_img[:2])):
        plot_grey_images(pca_img[i], title='weights of PCA components of image number' + str(i))
