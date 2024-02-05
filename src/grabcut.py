import numpy as np
import logging
import igraph as ig
from collections import namedtuple
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm as tq

con = namedtuple('_', ('FIX', 'UNK', 'FG', 'BG'))(1, 0, 1, 0)

def fit_gmms(img, alphas, n_components):
    fg = GaussianMixture(n_components=n_components)
    fg.fit(img[alphas == con.FG].reshape((-1, img.shape[-1])))

    bg = GaussianMixture(n_components=n_components)
    bg.fit(img[alphas == con.BG].reshape((-1, img.shape[-1])))

    return fg, bg

def graph_cut(img, types, alphas, fg_gmm, bg_gmm, beta, gamma, lamda, connect_diag):
    logging.info('GRAPH CUT')
    
    fg_D = - fg_gmm.score_samples(img.reshape((-1, img.shape[-1]))).reshape(img.shape[:-1])
    bg_D = - bg_gmm.score_samples(img.reshape((-1, img.shape[-1]))).reshape(img.shape[:-1])

    def compute_V(i, j, oi, oj):
        diff = img[i, j] - img[oi, oj]
        return gamma * np.exp(- beta * diff.dot(diff))

    fix_cap = lamda

    # BUILD GRAPH
    logging.info('BUILD GRAPH')
    num_pix = img.shape[0] * img.shape[1]

    def vid(i, j): # vertex ID
        return (img.shape[1] * i) + j

    def ind(idx): # image index
        return ((idx // img.shape[1]), (idx % img.shape[1]))
    
    graph = ig.Graph(directed=False)
    graph.add_vertices(num_pix + 2)
    S = num_pix
    T = num_pix+1

    edges = []
    weights = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            # add edges to S and T
            if types[i, j] == con.FIX:
                if alphas[i, j] == con.FG:
                    edges.append((vid(i, j), S))
                    weights.append(fix_cap)
                else:
                    edges.append((vid(i, j), T))
                    weights.append(fix_cap)
            else:
                edges.append((vid(i, j), S))
                weights.append(bg_D[i, j])

                edges.append((vid(i, j), T))
                weights.append(fg_D[i, j])
            
            # add edges to neighbours
            if i > 0:
                oi = i-1
                oj = j
                edges.append((vid(i, j), vid(oi, oj)))
                weights.append(compute_V(i, j, oi, oj))
            
            if j > 0:
                oi = i
                oj = j-1 
                edges.append((vid(i, j), vid(oi, oj)))
                weights.append(compute_V(i, j, oi, oj))

            if connect_diag:
                if i > 0 and j > 0:
                    oi = i-1
                    oj = j-1 
                    edges.append((vid(i, j), vid(oi, oj)))
                    weights.append(compute_V(i, j, oi, oj))

                if i > 0 and j < img.shape[1] - 1:
                    oi = i-1
                    oj = j+1 
                    edges.append((vid(i, j), vid(oi, oj)))
                    weights.append(compute_V(i, j, oi, oj))
    
    graph.add_edges(edges, attributes={'weight': weights})
    logging.info('MINCUT')
    cut = graph.st_mincut(S, T, capacity='weight')
    bg_vertices = cut.partition[0]
    fg_vertices = cut.partition[1]
    if S in bg_vertices:
        bg_vertices, fg_vertices = fg_vertices, bg_vertices
    
    new_alphas = np.zeros(img.shape[:2], dtype=np.uint8)
    for v in fg_vertices:
        if v not in (S, T):
            new_alphas[ind(v)] = 1
    return cut.value, new_alphas


def grab_cut(img_, types_, alphas_, n_components, gamma, lamda,
             num_iters, tol, connect_diag):
    
    logging.debug('GRAB CUT')
    img = img_.copy().astype(np.float32)
    types = types_.copy() 
    alphas = alphas_.copy() 
    
    # calculate beta
    logging.info('CALC BETA')
    beta = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i > 0:
                diff = img[i, j] - img[i-1, j]
                beta += diff.dot(diff)
            if j > 0:
                diff = img[i, j] - img[i, j-1]
                beta += diff.dot(diff)
            if connect_diag:
                if i > 0 and j > 0:
                    diff = img[i, j] - img[i-1, j-1]
                    beta += diff.dot(diff)
                if i > 0 and j < img.shape[1] - 1:
                    diff = img[i, j] - img[i-1, j+1]
                    beta += diff.dot(diff)
    if connect_diag:
        beta /= (4 * img.shape[0] * img.shape[1] - 3 * img.shape[0] - 3 * img.shape[1] + 2)
    else:
        beta /= (2 * img.shape[0] * img.shape[1] - img.shape[0] - img.shape[1])
    beta *= 2
    beta = 1 / beta
    
    prev_flow = -1
    for _ in tq(range(num_iters)):
        fg_gmm, bg_gmm = fit_gmms(img, alphas, n_components)
        flow, alphas = graph_cut(img, types, alphas, fg_gmm, bg_gmm, beta, gamma, lamda, connect_diag)
    
        if prev_flow != -1 and abs(prev_flow - flow) < tol:
            break
        
        prev_flow = flow
    
    logging.info('DONE')
    return alphas
