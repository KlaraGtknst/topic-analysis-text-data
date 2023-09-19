#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn import datasets, decomposition
# %%
#faces = datasets.fetch_olivetti_faces()
#faces.data.shape

# %%

def rgb2gray(img):
    return 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

import os
documents_raw = [plt.imread(fp.path) for fp in os.scandir("/Users/klara/Documents/uni/bachelorarbeit/images") if fp.path.endswith(".png")]

# %%
max_w = 0
max_h = 0
for doc in documents_raw:
    max_w = np.maximum(max_w, doc.shape[0])
    max_h = np.maximum(max_h, doc.shape[1])
print(max_w, max_h)

#%%

def down_sample(img,stride=2):
    w = img.shape[0]
    h = img.shape[1]
    comp = np.zeros((w//stride, h//stride))    
    for i,i_w in enumerate(range(0,w-stride,stride)):
        for j,i_h in enumerate(range(0,h-stride,stride)):
            comp[i,j] = np.mean(img[i_w:i_w+stride,i_h:i_h+stride])
    return comp


    

#%%

#documents = [doc[:500,:500] for doc in documents_raw]
#documents = np.asarray([rgb2gray(doc).ravel() for doc in documents if doc.shape == (500,500,3)])

documents = []
for doc in documents_raw:
    C = np.ones((max_w,max_h))
    C[:doc.shape[0],:doc.shape[1]] = rgb2gray(doc)
    documents.append(C.ravel())
documents = np.asarray(documents)

# %%

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
pca = decomposition.PCA(n_components=sqr_n_c**2, whiten=True, svd_solver="randomized")
# %%

pca.fit(X_train)
# %%

# %%
idx=np.random.choice(len(X_test), replace=False, size=10)
idx

trans = pca.transform(X_test[idx])
recon = pca.inverse_transform(trans)

width=max_w
height=max_h

# %%

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
    plt.plot(trans[n],'m')
    plt.xticks([])
    plt.yticks([])
plt.savefig("/tmp/eigendocs.png")
plt.show()

# %%
trans_train = pca.transform(X_train)
#%%
from sklearn.cluster import OPTICS

clt = OPTICS(min_samples=2, max_eps=10)
y = clt.fit_predict(trans_train)

plt.figure()
plt.scatter(*trans_train[:,[0,1]].T, c=y)
plt.show()

# %%

clazzes = np.unique(y)[1:]
n_cols = np.histogram(y)[0][1:].max()
n_rows = len(clazzes)
X = X_train
plt.figure(figsize=(2*n_cols,2*n_rows))
for i,c in enumerate(clazzes):
    for j,t in enumerate(X[y==c]):
        plt.subplot(n_rows,n_cols,i*n_cols+j+1)
        plt.imshow(t.reshape(max_w,max_h))
        plt.xticks([])
        plt.yticks([])
plt.show()

# %%
