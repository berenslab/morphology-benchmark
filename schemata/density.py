
from . import cell
from utils.utils import smooth_gaussian

import utils.NeuronTree as nt

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime
import datajoint as dj
import copy


schema = dj.schema('agberens_morphologies', locals())


@schema
class SkeletonPointSamplingMethod(dj.Lookup):
    definition = """
    # sampling properties defined for Neuron tree point cloud generation
    sampling_id: int
    ---
    """
    contents = [[1], [2], [3]]

    class Naive(dj.Part):
        definition = """
        -> SkeletonPointSamplingMethod
        ---
        sampling_dist: float # sampling distance along skeleton in microns
        """

    class Ignore(dj.Part):
        definition = """
        -> SkeletonPointSamplingMethod
        ---
        """

@schema
class Range(dj.Manual):
    definition= """
    -> cell.Dataset
    -> cell.NeuronPart
    ---
    min: longblob
    max: longblob
    """

@schema
class DMParam(dj.Lookup):
    definition = """ # parameters for density map calculation of neuronal trees
     dmparam_id: serial # id of the parameter set
     ---
     dim: int         # dimension of density map
     proj_axes: enum('0','01','02','1','10','12','2','20','21', 'None') # defines the axes that the point cloud is projected to when no decomposition was specified
     nbins=100: int       # number of bins used for each dimension
     fixed_range: boolean # determines whether a range is fixed or not
     smooth: boolean  # determines whether the density map is smoothed with a gaussian filter afterwards
     sigma: int # sigma for Gaussian smoothing
     ___
     UNIQUE INDEX(dim, proj_axes,nbins, fixed_range, smooth, sigma)
     """

    class Ignore(dj.Part):
        definition = """
        -> DMParam
        ---
        """


@schema
class SampledDensityMap(dj.Computed):
    definition = """
    # density map created from point clouds based on Neuronal skeleton files
    -> cell.Neuron
    -> cell.NeuronPart
    -> SkeletonPointSamplingMethod
    -> DMParam
    ---
    hist: longblob # histogram
    edges: longblob # bins the histogram is defined over
    start_time: timestamp
    end_time: timestamp
    """

    @property
    def key_source(self):
        return (cell.Neuron() & 'part_id < 4')* cell.NeuronPart() * (DMParam() - DMParam().Ignore()) * \
               (SkeletonPointSamplingMethod() - SkeletonPointSamplingMethod().Ignore())

    def plot_density(self):

        H = self.fetch1('hist')
        edges = self.fetch1('edges')

        if edges.shape[0] == 1:
            plt.plot(edges[0][:-1], H)
            plt.grid('off')
        elif edges.shape[0] == 2:
            sns.heatmap(H, xticklabels=False, yticklabels=False)

        else:
            raise NotImplementedError

    def _project_data(self, dim, proj_axes, data):

        p_a = proj_axes
        if dim == 2:
            if len(p_a) < 2:
                print('Invalid parameter setting: The passed projection axes {0} do not \
                      fit with the dimension of the projection {1}'.format(p_a, dim))
                (DMParam() & "dim={0}".format(dim) & "proj_axes={0}".format(p_a)).delete()
            else:
                indices = '012'
                for ix in range(len(p_a)):
                    indices = indices.replace(p_a[ix], '')
                deleted_axis = int(indices)
                ax = [0, 1, 2]
                ax.remove(deleted_axis)
                result = data[:, ax]

        elif dim == 1:
            if len(p_a) > 1:
                print('Invalid parameter setting: The passed projection axes {0} do not \
                      fit with the dimension of the projection {1}'.format(p_a, dim))
                (DMParam() & "dim={0}".format(dim) & "proj_axes={0}".format(p_a)).delete()
            else:
                ax = int(p_a)
                result = data[:, ax]
        else:
            result = data

        return result

    def _make_tuples(self, key):
        # fetch data and extend key
        # ...
        key = copy.copy(key)
        print('Populating:', key)

        # get Neuron
        if key['part_id'] == 1:
            neuron = cell.Neuron().get_tree(key)
        elif key['part_id'] == 2:
            neuron = (cell.Neuron().Axon() & key).get_tree()
        elif key['part_id'] == 3:
            neuron = (cell.Neuron().Dendrite() & key).get_tree()

        if neuron:
            # get point cloud
            dist = (SkeletonPointSamplingMethod().Naive() & key).fetch1('sampling_dist')
            pc = nt.NeuronTree.resample_nodes(neuron.get_graph(), dist)
            
            params = (DMParam() & key).fetch()
            # for normalization
            r = (Range() & key).fetch()
            
            pc = (pc - r['min'][0]) / (r['max'][0] - r['min'][0])

            start = datetime.datetime.now()
            
            # Rotate the point cloud
            if params['smooth'][0] == -1:
                pc_ = copy.copy(pc)
                angles = list(range(36,360,36))
                for k,a in enumerate(angles):

                    theta = a*np.pi/180
                    R = np.eye(3,3)
                    R[0,0] = np.cos(theta)
                    R[1,1] = np.cos(theta)
                    R[0,1] = - np.sin(theta)
                    R[1,0] = np.sin(theta)

                    X = np.matmul(R, pc.T)
                    pc_ = np.vstack((pc_,X.T))

                data = self._project_data(params['dim'], params['proj_axes'][0], pc_)
                indx = data[:,0] >=0
                data = data[indx,:]

                params['smooth'] = 1
                range_ = [[-.1,1.1], [-1.1,1.1]]
            else:
                range_ = [[-.1, 1.1]] * params['dim'][0]
                data = self._project_data(params['dim'], params['proj_axes'][0], pc)

            H, edges = np.histogramdd(data, bins=(params['nbins'][0],) * params['dim'][0],
                                      range=range_, normed=True)

            # perform smoothing
            if params['smooth']:
                print('Smoothing the histogram with sigma ', params['sigma'][0])
                H = smooth_gaussian(H, dim=params['dim'], sigma=params['sigma'][0])

            stop = datetime.datetime.now()
            self.insert1(dict(key, hist=H, edges=edges, start_time=start, end_time=stop))


