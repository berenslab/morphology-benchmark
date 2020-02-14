
import pandas as pd
import utils.NeuronTree as nt
import numpy as np
import json
import copy
from sklearn.decomposition import PCA
import datajoint as dj
from scipy.spatial import ConvexHull

schema = dj.schema('agberens_morphologies', locals())

@schema
class Dataset(dj.Lookup):
    definition = """
    #table of data sets
    
    ds_id             :int            # unique data set id
    --- 
    species ="mouse"  :enum("mouse", "rat", "monkey", "human") # species that has been recorded in 
    location          :varchar(255)   # anatomical location of the recorded area
    link     :varchar(255)   # link to paper that published data set first
    description_short :varchar(100)   # citation
    description_long=""  :varchar(4055)  # free description of data set
    """

    contents = [{'ds_id': 1, 'location': 'retina', 'link': 'http://www.nature.com/nature/journal/v500/n7461/abs/nature12346.html', 'description_short': 'Helmstaedter et al. 2013', "description_long":""},
               {'ds_id': 2, 'location': 'V1 L1', 'link': 'http://science.sciencemag.org/content/sci/350/6264/aac9462.full.pdf?sid=c22cb829-41df-4654-be25-8afc3960496c', 'description_short': 'V1 Jiang et al. 2015', "description_long":""},
              {'ds_id': 3, 'location': 'V1 L23', 'link': 'http://science.sciencemag.org/content/sci/350/6264/aac9462.full.pdf?sid=c22cb829-41df-4654-be25-8afc3960496c', 'description_short': 'V1 Jiang et al. 2015', "description_long":""},
               {'ds_id': 4, 'location': 'V1 L5', 'link': 'http://science.sciencemag.org/content/sci/350/6264/aac9462.full.pdf?sid=c22cb829-41df-4654-be25-8afc3960496c', 'description_short': 'V1 Jiang et al. 2015', "description_long":""},
              {'ds_id': 5, 'location': 'V1 L4', 'link':'https://www.biorxiv.org/content/10.1101/507293v2', 'description_short': 'Scala et al., 2019', 'description_long': "Interneurons in layer 4 of primary visual cortex"},
               {'ds_id': 6, 'location': 'S1 L1', 'link': 'https://www.jneurosci.org/content/39/1/125', 'description_short': 'Schuman et al., 2019', 'description_long': 'Establishes 4 IN cell types in L1 of mouse barrel cortex'}]

@schema
class Cell(dj.Manual):
    definition = """ # data of one cell
    ->Dataset
    c_num  :int # cell number within dataset
    ---
    type   :varchar(25)   #type of the cell
    """

    class Ignore(dj.Part):
        definition = """
        -> Cell
        """


@schema
class SkeletonFile(dj.Imported):
    definition = """  # skeleton file of a cell
    -> Cell
    ---
    path: varchar(255) # path to file
    """

    class Ignore(dj.Part):
        definition = """
           -> SkeletonFile
           ---
           """

@schema
class SWC(dj.Imported):
    definition = """
    -> SkeletonFile
    n       : int
    ---
    type    : int
    x       : float
    y       : float
    z       : float
    radius  : float
    parent  : int
    """

    def get_standardized_swc(self, soma_radius=None, pca_rot=True, center_on_soma=True):

        swc = pd.DataFrame(self.fetch(as_dict=True))

        print('Converting to microns...')
        scaling = (ScalingToMicrons() & self).fetch1('scaling_factor')

        swc.update(swc['x']/scaling)
        swc.update(swc['y']/scaling)
        swc.update(swc['z']/scaling)
        swc.update(swc['radius']/scaling)

        if pca_rot:
            print('Rotating x and y into their frame of maximal extent...')
            pca = PCA(copy=True)
            pc = np.vstack((swc['x'], swc['y'])).T
            pca.fit(pc)
            result = np.matmul(pc, pca.components_.T)

            swc.update(pd.DataFrame(result, columns=['x', 'y']))

        # create one point soma when there is more than three soma points
        sp = swc[swc['type'] == 1]

        if sp.shape[0] > 3:
            print('There are more than 3 soma points. The radius of the soma is estimated...')

            # calculate the convex hull of soma points
            convex_hull = ConvexHull(sp[['x', 'y', 'z']].values, qhull_options='QJ')

            hull_points = convex_hull.points

            centroid = np.mean(hull_points, axis=0)

            distances_to_centroid = np.linalg.norm(hull_points - centroid, axis=1)
            rad = np.max(distances_to_centroid)

            # fix the parent connections
            connected_points = pd.concat([swc[swc['parent'] == n] for n in sp['n']])
            connected_points['parent'] = 1
            swc.update(connected_points)

            # delete old soma points
            swc = swc.drop(swc.index[sp['n'].values[:-1]])

            soma_dict = dict(
                zip(['n', 'type', 'x', 'y', 'z', 'radius', 'parent'], [int(1), int(1), centroid[0], centroid[1],
                                                                       centroid[2], rad, int(-1)]))

            swc.update(pd.DataFrame(soma_dict, index=[0]))
            swc = swc.sort_index()

        if center_on_soma:
            print('Centering data on soma ...')
            centroid = swc[swc['n'] == 1][['x', 'y', 'z']].values.reshape(-1)

            swc.update(swc[['x', 'y', 'z']] - centroid)

        if soma_radius:
            print('Setting all nodes to type soma that have a larger radius than %s microns...' % soma_radius)

            d = np.vstack((swc['radius'], swc['type'])).T
            d[d[:, 0] >= soma_radius, 1] = 1
            swc.update(pd.DataFrame(d[:, 1].astype(int), columns=['type']))

        return swc


@schema
class ScalingToMicrons(dj.Lookup):
    definition = """
    -> Dataset
    ---
    scaling_factor  : float #scaling factor to transform swc file coordinates into microns
    """

    contents = [dict(ds_id=1, scaling_factor=20.0),
                dict(ds_id=2, scaling_factor=1.0),
                dict(ds_id=3, scaling_factor=1.0),
                dict(ds_id=4, scaling_factor=1.0),
                dict(ds_id=5, scaling_factor=1.0)]

@schema
class ReductionMethod(dj.Lookup):
    definition = """
    # defines the construction type of a NeuronTree

    reduction_id  : int
    ---
    description: varchar(255)
    """
    contents = [[0, "No reduction"], [1, "minimal spanning tree"]]

    class Displacement(dj.Part):
        definition = """
        -> ReductionMethod
        ---
        disp:   float   # in microns. Displacement from edge that is tolerable to remove the node in question. See NeuronTree.reduce()
        """

    class EdgeLength(dj.Part):
        definition = """
        -> ReductionMethod
        ---
        epsilon   :   float   # in microns. The node in question is removed if it doesn't change the edge length by epsilon.
        """


@schema
class NeuronPart(dj.Lookup):
    definition = """
    part_id  :   int # part of the neuron
    ---
    description:    varchar(50)
    """

    contents = [[1, 'full'], [2, 'axon'], [3, 'dendrite']]




@schema
class Neuron(dj.Computed):
    definition = """
    -> SkeletonFile
    -> ReductionMethod
    ---
    soma_pos   : blob       # Soma position in microns
    soma_rad     : float    # Soma radius in microns
    """

    @property
    def key_source(self):
        return (SkeletonFile() - SkeletonFile().Ignore())*ReductionMethod()

    def _make_tuples(self, key):
        # fetch data and extend key
        # ...

        def to_list(G):
            N = []
            for n in (G.get_graph().nodes(data=True)):
                n_ = list()
                n_.append(int(n[0]))
                n[1]['pos'] = n[1]['pos'].tolist()
                n[1]['type'] = float(n[1]['type'])
                n_.append(n[1])
                N.append(n_)

            E = []
            for e in (G.get_graph().edges(data=True)):
                e_ = list()
                e_.append(int(e[0]))
                e_.append(int(e[1]))
                e_.append(e[2])
                E.append(e_)
            return N, E

        key = copy.copy(key)

        print('Populating:', key)

        construction_type = (ReductionMethod() & key).fetch1('reduction_id')
        sampling_dist=1

        if key['ds_id'] == 1:
            # if it's bipolar cells center on the IPL
            swc = (SWC() & key).get_standardized_swc(soma_radius=1, center_on_soma=False, pca_rot=False)
            IPL_depth = 31.374999999999996
            swc.update(swc['z'] - IPL_depth)

            # get soma x and y
            soma_idx = swc['type'] == 1
            soma_x = swc[soma_idx].mean()['x']
            soma_y = swc[soma_idx].mean()['y']

            swc.update(swc['x'] - soma_x)
            swc.update(swc['y'] - soma_y)

        else:
            swc = (SWC() & key).get_standardized_swc(pca_rot=False)
        soma = swc[swc['parent'] == -1][['x', 'y', 'z', 'radius']]
        G = nt.NeuronTree(swc=swc)
        
        if construction_type > 0:
            # TODO sth like createParameter set so we only have to pass args** to reduce method
            if construction_type == 1:
                G = G.get_mst()

        else:
            if key['ds_id'] > 1 and key['ds_id'] < 7:
                
                # resample tree data
                R = G.resample_tree(dist=sampling_dist)
                # smooth y direction in cortex data
                G = R.smooth_neurites(dim=1, window_size=21)
            
        self.insert1(dict(key, soma_pos=soma[['x', 'y', 'z']].values[0], soma_rad=soma['radius'].values[0]))

        A = G.get_axonal_tree()
        NA, EA = to_list(A)
        if len(EA):
            self.Axon().insert1(dict(key, node_list=json.dumps(NA), edge_list=json.dumps(EA)))
        else:
            print('No axon specified in swc file.')

        D = G.get_dendritic_tree()
        ND, ED = to_list(D)
        if len(ED):
            self.Dendrite().insert1(dict(key, node_list=json.dumps(ND), edge_list=json.dumps(ED)))
        else:
            print('No dendrite specified in swc file.')

    class Axon(dj.Part):
        definition = """
        -> Neuron
        ---
        node_list   : longblob
        edge_list   : longblob
        """

        def get_tree(self):
            t = self.fetch()
            if t.size:
                N = json.loads(t['node_list'][0][0])
                E = json.loads(t['edge_list'][0][0])

                for n in N:
                    n[1]['pos'] = np.array(n[1]['pos'])
                G = nt.NeuronTree(node_data=N, edge_data=E)
                return G
            else:
                return None

    class Dendrite(dj.Part):
        definition = """
        -> Neuron
        ---
        node_list   : longblob
        edge_list   : longblob
        """

        def get_tree(self):
            t = self.fetch()
            if t.size:
                N = json.loads(t['node_list'][0][0])
                E = json.loads(t['edge_list'][0][0])

                for n in N:
                    n[1]['pos'] = np.array(n[1]['pos'])
                G = nt.NeuronTree(node_data=N, edge_data=E)
                return G
            else:
                return None

    def get_tree(self, key):

        n = []
        e = []
        A = (self.Axon() & key).get_tree()
        if A:
            A = A.get_graph()
            n += A.nodes(data=True)
            e += A.edges(data=True)

        D = (self.Dendrite() & key).get_tree()
        if D:
            D = D.get_graph()
            n += D.nodes(data=True)
            e += D.edges(data=True)

        G = nt.NeuronTree(node_data=n, edge_data=e)
        return G


@schema
class VolumeDataStats(dj.Manual):
    definition = """ # data that describes general properties about a volume data set
    -> Dataset
    ---
    voxel_volume: longblob # x,y,z extend of volume in number of voxel
    voxel_dim: longblob # x,y, z extend of one single voxel in nm
    """
    contents = [dict(ds_id=1, voxel_volume=[2731,2475,1067], voxel_dim=[50,50,50])]


@schema
class VolumeData(dj.Imported):
    definition = """   # soma centered and rotated volume data from one cell
    -> Cell
    -> VolumeDataStats
    ---
    soma_pos: longblob # position of the soma in recording volume
    rotation: longblob # rotation matrix to make recording volume upright
    data: longblob # 3D point cloud representing the recorded neuron
    """


