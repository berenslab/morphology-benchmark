from .cell import Dataset, NeuronPart, Neuron, Cell, ReductionMethod
from .density import SampledDensityMap, DMParam
from .morphometry import MorphometricDistribution, Morphometry, MorphometryDistributionType
from .persistence import Persistence

import numpy as np
import copy
from scipy.sparse import csc_matrix

import networkx as nx
import datajoint as dj
import pandas as pd
from utils.utils import computeStat
from scipy.linalg.basic import LinAlgError
import warnings

schema = dj.schema('agberens_morphologies', locals())


def _get_feature_vector(graph, weight, statType, steps=10, no_samples=100, avg=False):

    # helper function
    def calc_feature(adj, d, statType, steps):
        ind = np.array(np.argsort(d, axis=0))
        maxDist = np.max(d)
        numNeighb = d.size - 1
        stats = np.zeros((1, steps))
        for t in range(steps):
            trunc = ind[0:int(1 + np.floor(numNeighb * (t + 1) / steps))]
            stats[0, t] = computeStat(statType, adj[trunc][:, trunc], d[trunc], maxDist)

        return stats

    adj = nx.adjacency_matrix(graph.get_graph(), weight=weight)
    # make adjacency matrix symmetric
    WA = adj + adj.T
    W = csc_matrix(WA)

    # get shortest paths from soma to all nodes
    b = np.array(list(nx.single_source_dijkstra_path_length(graph.get_graph(), source=graph.get_root(),
                                                            weight=weight).values()))

    # if the statistic is supposed to be averaged, rename the statType and pick no_sample samples.
    if avg:
        sample_vecs = np.zeros((no_samples, steps))

        for k in range(no_samples):
            startNode = int(np.random.uniform(low=0, high=b.size))
            b_ = np.abs(b - b[startNode])
            sample_vecs[k, :] = calc_feature(W, b_, statType, steps)
        stats = np.mean(sample_vecs, axis=0)
    else:
        stats = calc_feature(W, b, statType, steps)

    return stats


@schema
class StatisticType(dj.Lookup):
    definition = """
    statistic_type : varchar(50)   # statistic type
     """
    contents = [['density_map'], ['graph'], ['morphometry'], ['morphometryDist'], ['combined'], ['persistence']]


@schema
class DimensionalityReductionMethod(dj.Lookup):
    definition = """
    # defines the methods used for dimensionality reduction
    dim_reduction_id: int
    ---
    description: varchar(100)
    """
    contents = [[1, 'No reduction method used'], [2, 'principal component analysis']]

    class PCA(dj.Part):
        definition = """
        # stores parameter for principal component analysis
        -> DimensionalityReductionMethod
        pca_id: int # id for different pca parameter sets
        ---
        n_components: int  # number of components kept
        whiten: boolean # whether input is whitened
        ___
        UNIQUE INDEX(n_components, whiten)
        """

    class Ignore(dj.Part):
        definition = """
        -> DimensionalityReductionMethod
        ---
        """


@schema
class EdgeAttribute(dj.Lookup):
    definition = """
    edge_attr_id : int
    ---
    attr_name: varchar(50)
    """

    contents = [[1, 'euclidean_dist'], [2, 'path_length']]

@schema
class GraphStatisticName(dj.Lookup):
    definition = """
    statistic_name: varchar(50)
    ---
    """
    contents = [['4starMotif', 'maxDist', 'meanEdgeLength_norm']]


@schema
class Statistic(dj.Manual):
    definition = """
    statistic_id: serial
    ---
    -> StatisticType
    """

    class Graph(dj.Part):
        definition = """
        -> Statistic
        ---
        -> GraphStatisticName  # type of statistic that is computed on graph
        steps = NULL: int # length of resulting feature vector
        -> [nullable] EdgeAttribute # edge attribute
        avg: tinyint(1) # bool to define whether the statistic is averaged over locations or not
        ___
        UNIQUE INDEX(statistic_name,steps,edge_attr_id,avg)
        """

    def populate_graph_statistic(self,start=None):

        # define keys for graph feature calculation
        todo = [('4starMotif', 10, 1, 0),
                ('4starMotif', 10, 2, 0),
                ('4starMotif', 10, 1, 1),
                ('4starMotif', 10, 2, 1),
                ('maxDist', 10, 1, 1)]

        keys = Statistic().Graph().fetch()

        for key in keys:
            k = tuple(key)[1:]

            try:
                todo.pop(todo.index(k))
            except ValueError or IndexError:
                continue

        for k, key in enumerate(todo):
            si = self.fetch('statistic_id')

            if start:
                id = start + k
            else:
                if len(si):
                    id = np.max(si) + 1
                else:
                    id = 1

            self.insert1([id, 'graph'])
            key = list(key)
            key.insert(0, id)
            self.Graph().insert1(key)

    class Morphometry(dj.Part):
        definition = """
        -> Statistic
        ---
        statistic_name: varchar(50) # name of morphometric statistic
        """

    def populate_morphometry_statistic(self, start=None):
        primary_key = Morphometry().proj().fetch(limit=1, as_dict=True)[0]
        done = self.Morphometry().fetch('statistic_name')
        todo = Morphometry().fetch(limit=1, as_dict=True)[0]

        # get rid of keys that are calculated already
        for key in primary_key:

            todo.pop(key)

        for key in done:
            todo.pop(key)

        for k, key in enumerate(todo.keys()):
            si = self.fetch('statistic_id')
            if start:
                id = start + k
            else:

                if len(si):
                    id = np.max(si) + 1
                else:
                    id = 1

            self.insert1([id, 'morphometry'])
            key = [id, key]
            self.Morphometry().insert1(key)

    class MorphometryDistribution(dj.Part):
        definition = """
        -> Statistic
        ---
        statistic_name: varchar(100) # name of morphometric distribution
        """

    def populate_morphometryDist_statistic(self, start=None):

        keys = Statistic().MorphometryDistribution().fetch('statistic_name')
        todo = list(np.unique(MorphometryDistributionType().fetch('morph_dist_name')))

        # get rid of keys that are calculated already
        for key in keys:
            try:
                todo.pop(todo.index(key))
            except ValueError or IndexError:
                continue

        for k, key in enumerate(todo):
            si = self.fetch('statistic_id')

            if start:
                id = start + k
            else:

                if len(si):
                    id = np.max(si) + 1
                else:
                    id = 1

            self.insert1([id, 'morphometryDist'])
            key = [id, key]
            self.MorphometryDistribution().insert1(key)

    class DensityMap(dj.Part):
        definition = """
        -> Statistic
        ---
        statistic_name: varchar(100)    # name of the statistic
        dimensions: int # dimension of the density map
        -> DMParam
        -> DimensionalityReductionMethod
        ___
        UNIQUE INDEX(dmparam_id, dim_reduction_id)
        """

    def populate_density_statistic(self, start=None):

        def _make_tuples(self, key):

            z = copy.copy(key)
            d = dict()
            statistic_name = 'density_map_'

            if z['proj_axes'].find('0') > -1:
                statistic_name += 'x'
            if z['proj_axes'].find('1') > -1:
                statistic_name += 'y'
            if z['proj_axes'].find('2') > -1:
                statistic_name += 'z'

            if z['smooth']:
                statistic_name += '_smoothed'

            # if z['dim_reduction_id'] == 1:
            d['dimensions'] = z['nbins'] ** z['dim']
            statistic_name += '_no_reduction'

            # elif z['dim_reduction_id'] == 2:
            #    d['dimensions'] = DimensionalityReductionMethod().PCA().fetch1('n_components')
            #    statistic_name += '_pca'

            d['statistic_name'] = statistic_name
            d['dmparam_id'] = z['dmparam_id']
            d['dim_reduction_id'] = 1
            d['statistic_id'] = z['statistic_id']

            self.DensityMap().insert1(d, skip_duplicates=True)

        # get keys for density map parameter
        allkeys = (DMParam() - DMParam().Ignore()) * DimensionalityReductionMethod()

        todo = allkeys & allkeys.proj() - Statistic().DensityMap()  # --> nice!

        for k, key in enumerate(todo.fetch(as_dict=True)):
            si = self.fetch('statistic_id')

            if start:
                id = start + k
            else:
                if len(si):
                    id = np.max(si) + 1
                else:
                    id = 1
            self.insert1([id, 'density_map'], skip_duplicates=True)
            key['statistic_id'] = id
            _make_tuples(self, key)

    class Combined(dj.Part):
        definition = """
        ->master
        ---
        statistic_name: varchar(50) # name of the statistic to be computed
        involved_ids: blob # statistic ids that are supposed to be combined
        """

    def get_source_table(self):
        statistic_type = self.fetch('statistic_type')

        if statistic_type == 'density_map':
            T = SampledDensityMap()
        elif statistic_type == 'graph':
            T = Neuron()
        elif statistic_type == 'morphometry':
            T = Morphometry()
        elif statistic_type == 'morphometryDist':
            T = MorphometricDistribution()

        elif statistic_type == 'persistence':
            T = Persistence()
        else:
            raise ValueError('statistic type %s has no single source table' % statistic_type)

        return T

    class Persistence(dj.Part):
        definition = """
        -> master
        ---
        statistic_name: varchar(50)
        dimension: int
        bins: int
        gaussian_kernel_std: float
        """
            
@schema
class FeatureMap(dj.Computed):
    definition = """
    -> Dataset
    -> Statistic
    -> NeuronPart
    -> ReductionMethod

    """

    class Feature(dj.Part):
        definition = """
        -> FeatureMap
        -> Neuron
        ---
        feature: longblob # Feature representation of the cell
        """

    class Ignore(dj.Part):
        definition = """
        -> master
        ---
        error : varchar(200) # error message that lead to exclusion
        """

    @property
    def key_source(self):

        base = Dataset().proj() * NeuronPart().proj() * ReductionMethod().proj() * Statistic()
        all_ = (base & "part_id <4") - FeatureMap().Ignore()
        all_4 = (base & "part_id =4") - FeatureMap().Ignore()

        U = (all_.proj() + all_4.proj())

        key_source = U - (U * Statistic() & "statistic_type!= 'density_map' and statistic_type!= 'graph'"
                          & "reduction_id=1")
        return key_source

    def _make_tuples(self, key):

        z = copy.copy(key)
        print('Populating ', z)
        statistic_type = (Statistic() & z).fetch1('statistic_type')

        if z['part_id'] == 4:
            # this means one wants to combine the axonal and dendritic features separately. This only
            # works if the features are in the DB already. Otherwise they are not saved.

            z['part_id'] =2
            axonal_data = self.get_as_dataframe(z)

            z['part_id'] = 3
            dendritic_data = self.get_as_dataframe(z)

            data = axonal_data.append(dendritic_data)

            if not any((axonal_data.empty, dendritic_data.empty)):

                z['part_id'] = 4
                self.insert1(dict(z))

                # insert single features
                c_nums = data.columns.get_level_values('cell_id')
                f = data.values.T
                feature_data = []
                for k, c_num in enumerate(c_nums):
                    d = {'c_num': c_num, 'feature': f[k, :]}
                    d.update(key)
                    feature_data.append(d)
                self.Feature().insert(feature_data)
            else:
                warnings.warn('Either axonal or dendritic feature map of key %s is not in the database yet! '
                              'This feature won\'t be stored')

        else:
            if statistic_type == 'density_map':

                data = (SampledDensityMap() * Statistic().DensityMap() & z)

                # get all histograms that were created with the respective DM parameter
                hists = data.fetch('hist')
                if hists.shape[0] > 0:
                    c_nums = data.fetch('c_num').tolist()
                    H = []
                    for h in hists:
                        H.append(h.reshape((-1)))
                    H = np.array(H)

                    f = H  # numofcellsxfeature

                    self.insert1(dict(z))

                    # insert the features
                    feature_data = []
                    for k in range(len(c_nums)):
                        d = {'c_num': c_nums[k], 'feature': f[k, :]}
                        d.update(z)
                        feature_data.append(d)
                    self.Feature().insert(feature_data)

                else:
                    pass
            elif statistic_type == 'graph':
                # calculate Graph

                # get graph statistic parameter
                graph_params = ((Statistic().Graph() & z)*EdgeAttribute()).fetch(as_dict=True)[0]
                statType = graph_params['statistic_name']
                steps = graph_params['steps']
                weight = graph_params['attr_name']
                avg = graph_params['avg']

                # get all data
                T = (Neuron() & z)

                if T.fetch().shape[0] > 0:

                    self.insert1(dict(z))

                    for t in T.proj().fetch(as_dict=True):
                        if z['part_id'] == 1:
                            G = Neuron().get_tree(t)
                        elif z['part_id'] == 2:
                            G = (Neuron().Axon() & t).get_tree()
                        elif z['part_id'] == 3:
                            G = (Neuron().Dendrite() & t).get_tree()

                        if G:
                            f = _get_feature_vector(G, weight, statType, steps, avg=avg)
                            d = {'c_num': t['c_num'], 'feature': f}
                            d.update(z)

                            self.Feature().insert1(d)

                else:
                    pass
            elif statistic_type == 'morphometry':

                #get morphometry name
                statistic_name = (Statistic().Morphometry() & z).fetch1('statistic_name')
                c_nums, data = (Morphometry() & z).fetch('c_num', statistic_name)
                if data.size:

                    self.insert1(dict(z))

                    l = [{'c_num': c_nums[i], 'feature': [data[i]]} for i in range(np.max(c_nums.shape))]

                    for d in l:
                        d.update(z)

                    self.Feature().insert(l)

            elif statistic_type == 'morphometryDist':
                # get morphometry distribution  name
                statistic_name = (Statistic().MorphometryDistribution() & z).fetch1('statistic_name')
                c_nums, data = (MorphometricDistribution() & z & "morph_dist_name = '%s'" % statistic_name).fetch('c_num',
                                                                                                                  'data')
                if data.size:

                    self.insert1(dict(z))

                    l = [{'c_num': c_nums[i], 'feature': data[i].flatten()} for i in range(np.max(c_nums.shape))]

                    for d in l:
                        d.update(z)

                    self.Feature().insert(l)

            elif statistic_type == 'combined':
                ids = (Statistic().Combined() & key).fetch('involved_ids')[0]

                d = pd.DataFrame()
                for id in ids:
                    key['statistic_id'] = id
                    if self & key:
                        d = d.append(self.get_as_dataframe(key))
                    else:
                        raise ValueError('Feature Map for statistic id %s is not in database (yet?).' % str(id))

                if not d.empty:

                    self.insert1(dict(z))

                    c_nums = d.columns.get_level_values('cell_id').values
                    data = d.values.T

                    l = [{'c_num': c_nums[i], 'feature': [data[i]]} for i in range(np.max(c_nums.shape))]

                    for d in l:
                        d.update(z)

                    self.Feature().insert(l)
            elif statistic_type == 'persistence':
                t = (Statistic().Persistence() & z).fetch1('gaussian_kernel_std')
                bins = (Statistic().Persistence() & z).fetch1('bins')
                dim = (Statistic().Persistence() & z).fetch1('dimension')

                # get filter function
                name = (Statistic().Persistence() & z).fetch1('statistic_name')
                filter_func = dict(function_name=" ".join(name.split(" ")[2:]))

                xmax = np.ceil(dj.U().aggr(Persistence() & z & filter_func, n='max(birth)').fetch1('n'))
                ymax = np.ceil(dj.U().aggr(Persistence() & z & filter_func, n='max(death)').fetch1('n'))

                xmin = np.floor(dj.U().aggr(Persistence() & z & filter_func, n='min(birth)').fetch1('n'))
                ymin = np.floor(dj.U().aggr(Persistence() & z & filter_func, n='min(death)').fetch1('n'))

                params = dict(t=t, xmax=xmax, xmin=xmin, ymax=ymax, ymin=ymin, bins=bins, dim=dim)
                c_nums = np.unique((Persistence() & z).fetch('c_num'))

                if z['part_id'] == 1:
                    persistence = Persistence() & z & filter_func
                elif z['part_id'] == 2:
                    persistence = Persistence() & z & filter_func & "node_type=1 or node_type=2"
                elif z['part_id'] == 3:
                    persistence = Persistence() & z & filter_func & "node_type=1 or node_type=3"

                # insert the features
                feature_data = []
                try:

                    for c in c_nums:

                        d = dict(c_num=c)
                        f = (persistence & "c_num=%s" % c).get_persistence_image(**params)[0].reshape(-1)
                        d['feature'] = f
                        d.update(z)
                        feature_data.append(d)

                    self.insert1(z)
                    self.Feature().insert(feature_data)
                except LinAlgError as e:
                    print('Caught error ', e, ' the feature map can not be calculated. Key is inserted into Ignore table.')
                    self.insert1(z)
                    z['error'] = str(e)
                    self.Ignore().insert1(z)
            else:
                raise NotImplementedError
   
    def get_as_dataframe(self, key, restriction=None):

        if restriction is not None:
            indices = dj.AndList((key, restriction))
        else:
            indices = key
        fm = np.array([list(f) for f in (self.Feature()*Cell() & indices & self).fetch('feature')]).squeeze()
        keys = (self.Feature()*Cell() & indices & self).fetch('type', 'c_num', 'ds_id')
        num_of_features = int(fm.size/keys[1].shape[0])

        if num_of_features == 1:
            fm = fm.reshape(-1, 1)
        # create Dataframe
        tuples = list(zip(*keys))
        df_index = pd.MultiIndex.from_tuples(tuples, names=['type', 'cell_id', 'ds_id'])

        df = pd.DataFrame(fm.T, index=['F{0}'.format(ix) for ix in range(1, num_of_features+1)], columns=df_index)
        return df

    def get_dimension_of_feature(self, key):

        f = (self.Feature() & self & key).fetch(limit=1, as_dict=True)[0]['feature']
        s = np.max(f.shape)
        fm = [s]
        return fm
