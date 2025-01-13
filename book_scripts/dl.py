import numpy as np
from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import SparseCoder

df_perm = np.load('data/data_500_128x128.npy')
df_perm_f = np.reshape(df_perm, (df_perm.shape[0], -1))

n_atoms = 500
dl = DictionaryLearning(n_components=n_atoms)
dictionary = dl.fit(df_perm_f)
atoms = dictionary.components_

sparse_code = SparseCoder(atoms).fit_transform(df_perm_f)
sparse_recs = np.reshape(sparse_code @ atoms, df_perm_f.shape)