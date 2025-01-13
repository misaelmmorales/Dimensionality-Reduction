import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from skimage.metrics import structural_similarity

df_perm = np.load('data/data_500_128x128.npy')          # (500, 128, 128)
df_perm_f = np.reshape(df_perm, (df_perm.shape[0], -1)) # (500, 16384)

# Full PCA
pca = PCA(n_components=500, svd_solver='full')
pca.fit(df_perm_f.T)
z = pca.transform(df_perm_f.T)

cutoffs = [0.2, 0.5, 0.8, 0.95]
energy = np.cumsum(pca.explained_variance_ratio_)
nk = [np.argwhere(energy > cutoff)[0][0] for cutoff in cutoffs]

reconstructions = []
for i, k in enumerate(nk):
    pca = PCA(n_components=k)
    z = pca.fit_transform(df_perm_f.T)
    xhat = pca.inverse_transform(z)
    r = np.moveaxis(np.reshape(xhat, (128, 128, -1)), -1, 0)
    reconstructions.append(r)