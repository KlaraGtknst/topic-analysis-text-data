'''Provides code to use Eigendocs, i.e. an adaption of Eigenfaces.
Most of the source code was provided by Dr. Christian Gruhl.'''

import numpy as np
import matplotlib as plt
from sklearn.decomposition import PCA

def rgb2gray(img:np.array):
    '''returns array of greyscale values'''
    return 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

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
        C[:doc.shape[0],:doc.shape[1]] = rgb2gray(doc)
        # 2d to 1d array
        documents.append(C.ravel())
    # list to array
    return np.asarray(documents)

def plot_expl_var(pca:PCA, save=False):
    '''plots (and saves) the explained variance for a trained pca model.'''
    fig, ax = plt.subplots(figsize=(10,6))
    y = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(y, label="cumulative explained variance")
    plt.title("Cumulative explained variance")
    plt.xlabel("Number of components")
    plt.ylabel("Explained variance")
    plt.axhline(y = 0.9, color = 'r', linestyle = '-', label="90% explained variance")
    interp = np.interp(0.9, y, list(range(len(y))))
    plt.axvline(x=interp,color='grey', label="{} components".format(int(interp)))
    plt.legend()
    if save:
        plt.savefig("results/cumulative_explained_variance.pdf", format="pdf")
    plt.show()


def plot_rec_err(X_train:np.array, X_test:np.array, n_max:int=10, save:bool=False):
    '''plots the reconstruction error for different number of components (PCA)'''
    reconstr_err = []
    for i in range(1, n_max):
        pca = PCA(n_components=i, whiten=True, svd_solver="randomized")   
        pca.fit(X_train)
        X_test_pca = pca.transform(X_test)
        X_test_pca_inverse = pca.inverse_transform(X_test_pca)
        reconstr_err.append(np.mean((X_test - X_test_pca_inverse)**2))

    # plot reconstruction error
    plt.figure(figsize=(10,6))
    plt.plot(reconstr_err, label="reconstruction error")
    plt.title("Reconstruction error")
    plt.xlabel("Number of components")
    plt.ylabel("Reconstruction error")
    plt.legend()
    if save:
        plt.savefig("results/reconstruction error.pdf", format="pdf")
    plt.show()
    