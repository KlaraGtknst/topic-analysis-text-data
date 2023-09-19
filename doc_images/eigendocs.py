#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import decomposition
import os

# %%

def rgb2gray(img):
    return 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]


documents_raw = [plt.imread(fp.path) for fp in os.scandir("/Users/klara/Documents/uni/bachelorarbeit/images") if fp.path.endswith(".png")]

# %%
def get_maximum_height_width(documents):
    max_w = 0
    max_h = 0
    for doc in documents:
        max_w = np.maximum(max_w, doc.shape[0])
        max_h = np.maximum(max_h, doc.shape[1])
    return max_w, max_h

max_w, max_h = get_maximum_height_width(documents_raw)
print(max_w, max_h)

#%%

def down_sample(img,stride=2):
    '''
    Downsample by replacing multiple pixels with one (their mean value).
    '''
    w = img.shape[0]
    h = img.shape[1]
    comp = np.zeros((w//stride, h//stride))    
    for i,i_w in enumerate(range(0,w-stride,stride)):
        for j,i_h in enumerate(range(0,h-stride,stride)):
            comp[i,j] = np.mean(img[i_w:i_w+stride,i_h:i_h+stride])
    return comp


    

#%%
documents = []
for doc in documents_raw:
    # same size for all documents
    C = np.ones((max_w,max_h))
    # convert to grayscale
    C[:doc.shape[0],:doc.shape[1]] = rgb2gray(doc)
    # 2d to 1d array
    documents.append(C.ravel())
# list to array
documents = np.asarray(documents)

# %%
# display documents
cols = 10
rows = int(np.ceil(len(documents)/cols))
plt.figure(figsize=(20,rows*2))
for i,doc in enumerate(documents):
    plt.subplot(rows, cols, 1+i)
    plt.imshow(doc.reshape(max_w,max_h), cmap=plt.cm.bone)
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()

# %%
X_train, X_test = train_test_split(documents, random_state=0)

sqr_n_c=2
# svd solver is part of eigendocs algorithm: XX^T to X^TX and use algorithm for eigenvectors
pca = decomposition.PCA(n_components=sqr_n_c**2, whiten=True, svd_solver="randomized")
# %%

pca.fit(X_train)
# %%

# %%
# 10 test documents
idx=np.random.choice(len(X_test), replace=False, size=10)
idx

trans = pca.transform(X_test[idx])
recon = pca.inverse_transform(trans)

width=max_w
height=max_h

# %%
# test images
# display original, reconstructed, difference (error of reconstruction), transformation, and transformation (?) plot
plt.figure(figsize=(20,10))
plt.tight_layout()
for n,i in enumerate(idx):
    plt.subplot(5,10,1+n)
    plt.imshow(X_test[i].reshape(width,height), cmap=plt.cm.bone)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(5,10,11+n)
    plt.imshow(recon[n].reshape(width,height), cmap=plt.cm.bone)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(5,10,21+n)
    plt.imshow((X_test[i]-recon[n]).reshape(width,height), cmap=plt.cm.bone)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(5,10,31+n)
    plt.imshow(trans[n].reshape(sqr_n_c,sqr_n_c), cmap=plt.cm.bone)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(5,10,41+n)
    plt.plot(trans[n],'m')  # m is magenta
    plt.xticks([])
    plt.yticks([])
plt.savefig("/Users/klara/Developer/Uni/topic-analysis-text-data/results/eigendocs.png")
plt.show()

# %%
# training images
trans_train = pca.transform(X_train)    # lower dimensional representation of training data
#%%
from sklearn.cluster import OPTICS

clt = OPTICS(min_samples=2, max_eps=10)
y = clt.fit_predict(trans_train)    # on lower dimensional representation of training data


fig, ax = plt.subplots()
scatter = ax.scatter(*trans_train[:,[0,1]].T, c=y)   # display docs first (PCA) two components and their cluster
legend = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend)
plt.show()

# %%

clazzes = np.unique(y)[1:]  # index zero is noise
n_cols = np.histogram(y)[0][1:].max()   # number of columns is the maximum number of documents in a cluster
n_rows = len(clazzes)   # number of rows is the number of clusters
X = X_train
plt.figure(figsize=(2*n_cols,2*n_rows))
for i,c in enumerate(clazzes):  # cluster number c
    for j,t in enumerate(X[y==c]):  # document number t
        plt.subplot(n_rows,n_cols,i*n_cols+j+1)
        plt.imshow(t.reshape(max_w,max_h))  # 1d to 2d array
        plt.xticks([])
        plt.yticks([])
plt.show()

# %%
def get_optics_plot(documents, max_eps=np.inf, cluster_method='xi', eps=np.inf, save=False, min_samples=10):
    if cluster_method == 'xi':
        clust = OPTICS(min_samples=min_samples, metric='euclidean', cluster_method='xi')
    else:
        clust = OPTICS(min_samples=min_samples, metric='euclidean', max_eps=max_eps, cluster_method='dbscan', eps=eps)
    clust.fit(documents)
    reachability = clust.reachability_[clust.ordering_]
    print(reachability)
    labels = clust.labels_[clust.ordering_]
    space = np.arange(len(documents))

    # Reachability plot
    colors = ["g.", "r.", "b.", "y.", "c."]
    for klass, color in zip(range(0, 5), colors):
        Xk = space[labels == klass]
        Rk = reachability[labels == klass]
        plt.plot(Xk, Rk, color, alpha=0.3, label=(f"class {klass}" if len(Xk) > 0 else None))
    plt.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3, label="noise")
    plt.ylabel("Reachability (epsilon distance)")
    plt.title("Reachability Plot")
    plt.legend()
    if save:
        plt.savefig('results/reachability_plot.pdf', format='pdf')
    plt.show()

    # OPTICS
    colors = ["g.", "r.", "b.", "y.", "c."]
    for klass, color in zip(range(0, 5), colors):
        Xk = documents[clust.labels_ == klass]
        plt.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, label=(f"class {klass}" if len(Xk) > 0 else None))
    plt.plot(documents[clust.labels_ == -1, 0], documents[clust.labels_ == -1, 1], "k+", alpha=0.6, label="noise")
    plt.title("Automatic Clustering\nOPTICS")
    plt.legend()
    plt.show()


    # OPTICS 3d
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x1 = documents[:, 0]
    x2 = documents[:, 1]
    x3 = documents[:, 2]

    ax.scatter(x1, x2, x3, marker='o', c=clust.labels_, cmap="brg")
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    plt.title('OPTICS identified clusters')
    if save:
        plt.savefig('results/OPTICS_cluster.pdf', format='pdf')
    plt.show()
    return labels
    

labels = get_optics_plot(trans_train, save=False, cluster_method='dbscan', min_samples=2, max_eps=10, eps=0.5)
# %%
import random
def display_example_images(documents, labels):
    colors = ["g.", "r.", "b.", "y.", "c."]
    for klass, color in zip(range(-1, 5), colors):
        cluster_instances = documents[labels == klass]
        if len(cluster_instances) == 0:
            continue
        doc = random.choice(cluster_instances)
        plt.imshow(doc.reshape(max_w,max_h), cmap=plt.cm.bone)
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Example image of cluster {klass}')
        #plt.savefig(f'results/example_image_cluster_{klass}.pdf', format='pdf')
        plt.show()
        
display_example_images(X_train, labels)
# %%
