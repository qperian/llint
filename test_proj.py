import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.linalg import orth
def get_proj_matrix(embeddings, n_comp=None):
    embeddings = orth(embeddings.T)
    P = embeddings@embeddings.T
    return np.eye(P.shape[0]) - P
    if n_comp == None:  
        n_comp = len(embeddings)
    tSVD = TruncatedSVD(n_components=n_comp)
    embeddings_ = tSVD.fit_transform(embeddings)
    basis = tSVD.components_.T

    # orthogonal projection
    proj = np.linalg.inv(np.matmul(basis.T, basis))
    proj = np.matmul(basis, proj)
    proj = np.matmul(proj, basis.T)
    proj = np.eye(proj.shape[0]) - proj
    return proj

print(np.linalg.norm(np.array([1,5,6])@get_proj_matrix(np.array([[1,0,4],[0,2,0]]))))