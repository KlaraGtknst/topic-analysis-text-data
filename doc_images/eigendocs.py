'''Provides code to use Eigendocs, i.e. an adaption of Eigenfaces.
Most of the source code was provided by Dr. Christian Gruhl.'''

import numpy as np
import matplotlib as plt
from sklearn.decomposition import PCA

def rgb2gray(img: np.ndarray):
    '''returns array of greyscale values'''
    try:
        return 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    except IndexError: # already greyscale
        return img

def get_maximum_height_width(documents:list):
    '''returns maximum height and width of a list from images.'''
    max_w = 0
    max_h = 0
    for doc in documents:
        max_w = np.maximum(max_w, doc.shape[0])
        max_h = np.maximum(max_h, doc.shape[1])
    return max_w, max_h

def proprocess_docs(raw_documents:list, max_w:int, max_h:int):
    '''
    return same sized, greyscale documents as an array
    '''
    documents = []
    for doc in raw_documents:
        # same size for all documents
        C = np.ones((max_w,max_h))
        # convert to grayscale
        C[:doc.shape[0],:doc.shape[1]] = rgb2gray(doc) if len(doc.shape) == 3 else doc
        # 2d to 1d array
        documents.append(C.ravel())

    # list to array
    return np.asarray(documents)

def plot_expl_var(pca:PCA, save=False):
    '''plots (and saves) the explained variance for a trained pca model.'''
    fig, ax = plt.subplots(figsize=(10,6)) # type: ignore
    y = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(y, label="cumulative explained variance") # type: ignore
    plt.title("Cumulative explained variance") # type: ignore
    plt.xlabel("Number of components") # type: ignore
    plt.ylabel("Explained variance") # type: ignore
    plt.axhline(y = 0.9, color = 'r', linestyle = '-', label="90% explained variance") # type: ignore
    interp = np.interp(0.9, y, list(range(len(y))))
    plt.axvline(x=interp,color='grey', label="{} components".format(int(interp))) # type: ignore
    plt.legend() # type: ignore
    if save:
        plt.savefig("results/cumulative_explained_variance.pdf", format="pdf") # type: ignore
    plt.show() # type: ignore


def plot_rec_err(X_train:np.ndarray, X_test:np.ndarray, n_max:int=10, save:bool=False):
    '''plots the reconstruction error for different number of components (PCA)'''
    reconstr_err = []
    for i in range(1, n_max):
        pca = PCA(n_components=i, whiten=True, svd_solver="randomized")   
        pca.fit(X_train)
        X_test_pca = pca.transform(X_test)
        X_test_pca_inverse = pca.inverse_transform(X_test_pca)
        reconstr_err.append(np.mean((X_test - X_test_pca_inverse)**2))

    # plot reconstruction error
    plt.figure(figsize=(10,6)) # type: ignore
    plt.plot(reconstr_err, label="reconstruction error") # type: ignore
    plt.title("Reconstruction error") # type: ignore
    plt.xlabel("Number of components") # type: ignore
    plt.ylabel("Reconstruction error") # type: ignore
    plt.legend() # type: ignore
    if save:
        plt.savefig("results/reconstruction error.pdf", format="pdf") # type: ignore
    plt.show() # type: ignore
    