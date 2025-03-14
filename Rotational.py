from bend_utils import get_proj_matrix, get_cos_neighbors_tensor
import numpy as np
import torch
from scipy.linalg import orth
def get_proj_matrix(embeddings, n_comp=None):
    embeddings = orth(embeddings.T)
    P = embeddings@embeddings.T
    return np.eye(P.shape[0]) - P
def rotational_equalize(embeddings, ref_imgs, bias_space, low_conf_indices, labels):
    print(embeddings.shape)
    print(bias_space.shape)
    P = get_proj_matrix(bias_space)
    P_in = np.identity(P.shape[0]) - P.T

    outside_subspace = np.matmul(embeddings, P.T)
    in_subspace = np.matmul(embeddings, P_in)

    # print(ref_imgs.shape)
    # low_conf_embeddings = ref_imgs

    smallest_cat = 20#min([list(labels).count(i) for i in set(labels)])
    print(smallest_cat)
    equalized_indices = []
    for i in set(labels):
        equalized_indices.extend(np.random.choice(np.where(np.array(labels) == i)[0], smallest_cat))
    low_conf_embeddings = ref_imgs[equalized_indices]

    print(low_conf_embeddings.shape)
    low_conf_embeddings_in = np.mean(np.matmul(low_conf_embeddings, P_in), axis=0)
    print(low_conf_embeddings_in.shape) 
    low_conf_norm = low_conf_embeddings_in/np.linalg.norm(low_conf_embeddings_in).reshape(-1,1)
    print(low_conf_norm.shape, in_subspace.shape)
    return outside_subspace+low_conf_embeddings_in#np.linalg.norm(in_subspace, axis=-1).reshape(-1, 1)@low_conf_norm
# test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(rotational_equalize(test, test, np.array([[1, 0, 0], [0, 1, 0]]), [0]))
