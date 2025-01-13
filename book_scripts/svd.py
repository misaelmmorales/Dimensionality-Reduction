import numpy as np
from scipy.linalg import svd
from sklearn.metrics import r2_score
from skimage.metrics import structural_similarity

df_perm = np.load('data/data_500_128x128.npy')          # (500, 128, 128)
df_perm_f = np.reshape(df_perm, (df_perm.shape[0], -1)) # (500, 16384)

u, s, vt = svd(df_perm_f.T, full_matrices=True); ss = np.diag(s)

cutoffs = [0.2, 0.5, 0.8, 0.95] # SV energy cutoff values
energy = np.cumsum(np.diag(ss))/np.sum(np.diag(ss))
nk = [np.argwhere(energy > cutoff)[0][0] for cutoff in cutoffs]
dd = [u[:, :n] @ ss[:n, :n] @ vt[:n, :] for n in nk]
reconstructions = [np.moveaxis(np.reshape(d, (128, 128, -1)), -1, 0) for d in dd]