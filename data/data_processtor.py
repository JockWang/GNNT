import dgl
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import torch
import logging

def processtor(dataset='ml-100k', length=20):
    """
    make a graph base dataset.
    :param dataset: input the dataset name.
    :return: a graph bases networkx
    """
    G = nx.Graph()
    if dataset == 'ml-100k':
        logging.info('loading dataset: ml-100k ...')
        data = pd.read_csv('data/ml-100k/u.data', header=None, index_col=None, sep='\t')
        data.columns = ['user_id','item_id','rating','timestamp']
        data = data.sort_values('timestamp')
        logging.info(dataset+' shape: %s' % str(data.shape))
        users = dict(data.groupby('user_id')['item_id'].apply(lambda x: x.values))
        logging.info('Creat a graph ...')
        for u in users:
            for i in range(len(users[u])):
                for j in range(i+1, len(users[u])):
                    G.add_edge(users[u][i], users[u][j])
        H = nx.DiGraph(G.to_undirected())
        dglG = dgl.DGLGraph()
        dglG.from_networkx(nx_graph=H)
        items = pd.read_csv('data/ml-100k/u.item', header=None, index_col=None, sep='|')
        items.columns = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown',
                         'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                         'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
                         'War', 'Western']
        logging.info('items features shape:%s'%str(items.shape))
        items['year'] = items['release date'].apply(lambda x: str(x).split('-')[-1])
        encoder = OneHotEncoder().fit_transform(items['year'].values.reshape(-1,1)).toarray()
        del items['movie title'],items['release date'],items['video release date'],items['IMDb URL'],items['year']
        items = np.hstack((items.values, encoder))
        # index = items[:, 0].tolist()
        # items = torch.from_numpy(items[:, 1:])
        # dglG.ndata['feature'] = items
        # pd.to_pickle(dglG, 'data/ml-100k/ml-100.graph')
        # dglG = pd.read_pickle('data/ml-100k/ml-100.graph')
        for u in users:
            users[u] = users[u][-length:]
        logging.info('graph nodes number:%d' % dglG.number_of_nodes())
        logging.info('graph edges number:%d' % dglG.number_of_edges())

        return users, items[:,1:], dglG

