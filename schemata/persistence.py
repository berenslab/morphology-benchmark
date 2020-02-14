from .cell import Neuron

import datajoint as dj
import numpy as np
import pandas as pd
from scipy.stats import norm, gaussian_kde
from scipy.linalg import LinAlgError


schema = dj.schema('agberens_morphologies', locals())


@schema
class PersistenceFilterFunction(dj.Lookup):
    definition = """
    function_name: varchar(100)
    ---
    """


def get_filter_function(key):
    assert len(PersistenceFilterFunction() & key) == 1, "Please select only one filter function"

    filter_function = (PersistenceFilterFunction() & key).fetch1('function_name')

    if filter_function == 'radial distance':
        def radial_distance(G, u, v):
            n = G.node[u]['pos']
            r = G.node[v]['pos']

            return np.sqrt(np.dot(n - r, n - r))

        f = radial_distance
    elif filter_function == 'height':
        def height(G, u, v):
            n = G.node[u]['pos']
            r = G.node[v]['pos']
            return np.abs((n - r))[2]

        f = height
    elif filter_function == 'z-projection':
        def projection(G, u, v):
            n = G.node[u]['pos']
            r = G.node[v]['pos']
            return np.dot((n - r), np.array([0,0,1]))

        f = projection
    elif filter_function == 'path length':
        import networkx as nx

        def path_length(G, u, v):
            return nx.shortest_path_length(G, v, u, weight='path_length')

        f = path_length
    elif filter_function == 'branch order':
        import networkx as nx

        def branch_order(G, u, v):
            
            if u == v:
                bo = 0
            else:
                path = nx.shortest_path(G, v, u)
                path.remove(u) # the last node doesn't count
                bo = np.sum(np.array(list(nx.degree(G, path).values())) > 2)
            return bo

        f = branch_order
    else:
        f = None

    return f

@schema
class Persistence(dj.Computed):
    definition = """
    -> Neuron
    -> PersistenceFilterFunction
    node_id: int
    ---
    birth: double
    death: double
    node_type: int
    """

    @property
    def key_source(self):
        return Neuron()*PersistenceFilterFunction()& "reduction_id=0"

    def _make_tuples(self, key):
        N = Neuron().get_tree(key)

        filter_function = get_filter_function(key)

        df = N.get_mst().get_persistence(f=filter_function)
        for k in key.keys():
            df[k] = key[k]

        self.insert((r.to_dict() for _, r in df.iterrows()))

    def get_persistence_image(self, dim=1, t=1, bins=100, xmax=None, xmin=None,  ymax=None, ymin=None):

        data = pd.DataFrame(self.fetch())

        if not data.empty:
            if dim == 1:
                y = np.zeros((bins,))
                if xmax is None:
                    xmax = np.max(data['birth'])
                if xmin is None:
                    xmin = np.min(data['birth'])
                x = np.linspace(xmin, xmax, bins)

                for k, p in data.iterrows():
                    m = np.abs(p['birth'] - p['death'])
                    y += m * norm.pdf(x, loc=p['birth'], scale=t)
                return y, x

            elif dim == 2:
                
                try:
                    kernel = gaussian_kde(data[['birth', 'death']].values.T)
                except LinAlgError as e:
                    print('Warning! Captured error %s'%e)
                    print('Add one additional data point.')
                    
                    birth = data['birth'].subtract(1)
                    death = data['death'].add(1)
                    data = pd.concat((birth, death), axis=1)
                    data = data.append(pd.DataFrame(dict(birth=np.max(data['birth']+1), death=0), index=[len(data)]))
                    
                    kernel = gaussian_kde(data[['birth', 'death']].values.T)
                    
                if xmax is None:
                    xmax = np.max(data['birth'].values)
                if xmin is None:
                    xmin = np.min(data['birth'].values)
                if ymax is None:
                    ymax = np.max(data['death'].values)
                if ymin is None:
                    ymin = np.min(data['death'].values)

                X, Y = np.mgrid[xmin:xmax:np.complex(bins), ymin:ymax:np.complex(bins)]
                positions = np.vstack([X.ravel(), Y.ravel()])

                Z = np.reshape(kernel(positions).T, X.shape)
                return Z, [np.unique(X), np.unique(Y)]
