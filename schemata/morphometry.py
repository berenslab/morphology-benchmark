from .cell import Neuron, NeuronPart
from copy import copy
import networkx as nx
import numpy as np
import sys

import matplotlib.pyplot as plt
import seaborn as sns

import datajoint as dj

schema = dj.schema('agberens_morphologies', locals())
sys.setrecursionlimit(10000)

@schema
class MorphometryDistributionType(dj.Lookup):
    definition = """
    morph_dist_name : varchar(100) # name of the morphometric distribution
    ---
    description: varchar(200)
    """
    
    contents = [['root angle', 'root angle distribution of axis angles'], ['root angle euler', 'root angle distribution of euler angles'],
                ['sholl intersection xy', '2D Sholl intersection profile in xy plane'], 
                ['sholl intersection xz', '2D Sholl intersection profile in xz plane'], 
                ['sholl intersection yz', '2D Sholl intersection profile in yz plane'], 
                ['thickness', 'distribution of thickness '],
                ['thickness vs path dist', 'distribution of thickness against path distance from soma'],
                ['thickness vs branch order', 'distribution of thickness against branch order from soma'],
                ['branch order', 'distribution of branching orders'],
                ['path angle', 'distribution of the angles along the neural path '],
                ['path angle vs path dist', 'distribution of the angles along the neural path '
                                            'against path distance from soma'],
                ['path angle vs branch order', 'distribution of the angles along the neural path against branch'
                                               'order from soma'],
                ['branch angle', 'distribution of the branch angles'],
                ['branch angle vs path dist', 'distribution of the branch angles against the path distance from soma'],
                ['branch angle vs branch order', 'distribution of the branch angles against branch order'],
                ['vertical dist to soma', 'distribution of the vertical distance to soma'],
                ['segment length', 'distribution of segment length'],
                ['segment path length vs branch order', 'distribution of segment path length as function of branch order']]

@schema
class Morphometry(dj.Computed):
    definition = """
    -> Neuron
    -> NeuronPart
    ---
    branch_points:  int # number of branch points
    tips:           int # number of tips
    height:         float   # height [z-extension] of the Neuron in microns
    width:          float   # width [x-extension] of the neuron
    depth:          float   # depth [y-extension] of the neuron
    stems:          int     # number of branches from cell body
    avg_thickness:  float   # average radius of all neurites (without soma)
    max_thickness:  float   # maximal radius of all neurites (without soma). Only stored for normalization purposes
    total_length:   float   # summed total path length of all branches
    total_surface:  float   # total surface area
    total_volume:   float   # total volume
    max_path_dist_to_soma: float    # maximal path length to soma
    max_branch_order:   int  # maximal branch order
    max_segment_path_length:    float # length of longest segment path length (from one branch point to the other) in microns. Only stored for normalization purposes.       
    median_intermediate_segment_pl: float # median segment path length of intermediate segments
    median_terminal_segment_pl:    float # median segment path length of terminal segments                                                
    min_path_angle:     float   # minimal path angle along a neurite
    median_path_angle:    float  # median path angle along a neurite
    max_path_angle:     float   # 99.5 percentile of path angle along a neurite
    min_tortuosity:     float   # log(minimal tortuosity), measure of curvature. The ratio between path length and euclidean distance.
    median_tortuosity:    float   # log(median tortuosity), measure of curvature. The ratio between path length and euclidean distance.
    max_tortuosity:     float   # log(99.5 percentile of tortuosity), measure of curvature. The ratio between path length and euclidean distance.
    min_branch_angle:   float   # minimal branch angle  
    mean_branch_angle:  float   # mean branch angle
    max_branch_angle:   float   # maximal branch angle
    max_degree:         float   # maximal branching degree within neuron (soma excluded)
    tree_asymmetry:     float   # tree asymmetry index 
    """
    @property
    def key_source(self):
        return (Neuron() * NeuronPart()) & "reduction_id = 0" & "part_id < 4"

    def _make_tuples(self, key):
        z = copy(key)
        print('Populating %s' % z)

        if z['part_id'] == 1:
            T = Neuron().get_tree(z)
        elif z['part_id'] == 2:
            T = (Neuron().Axon() & z).get_tree()
        elif z['part_id'] == 3:
            T = (Neuron().Dendrite() & z).get_tree()

        if T:
            z['branch_points'] = T.get_branchpoints().size
            extend = T.get_extend()

            z['width'] = extend[0]
            z['depth'] = extend[1]
            z['height'] = extend[2]

            tips = T.get_tips()

            z['tips'] = tips.size

            z['stems'] = len(T.edges(1))

            z['total_length'] = np.sum(list(nx.get_edge_attributes(T.get_graph(), 'path_length').values()))
            # get all radii
            radii = nx.get_node_attributes(T.get_graph(), 'radius')
            # delete the soma
            radii.pop(T.get_root())
            z['avg_thickness'] = np.mean(list(radii.values()))
            z['max_thickness'] = np.max(list(radii.values()))

            z['total_surface'] = np.sum(list(T.get_surface().values()))
            z['total_volume'] = np.sum(list(T.get_volume().values()))

            z['max_path_dist_to_soma'] = np.max(T.get_distance_dist()[1])
            z['max_branch_order'] = np.max(list(T.get_branch_order().values()))

            path_angles = []
            for p1 in T.get_path_angles().items():
                if p1[1].values():
                    path_angles += list(list(p1[1].values())[0].values())

            z['max_path_angle'] = np.percentile(path_angles,99.5)
            z['min_path_angle'] = np.min(path_angles)
            z['median_path_angle'] = np.median(path_angles)

            R = T.get_mst()
            segment_length = R.get_segment_length()
            terminal_segment_pl = [item[1] for item in segment_length.items() if item[0][1] in tips]
            intermediate_segment_pl = [item[1] for item in segment_length.items() if item[0][1] not in tips]

            z['max_segment_path_length'] = np.max(list(segment_length.values()))
            z['median_intermediate_segment_pl'] = np.median([0] + intermediate_segment_pl)
            z['median_terminal_segment_pl'] = np.median(terminal_segment_pl)

            tortuosity = [e[2]['path_length'] / e[2]['euclidean_dist'] for e in R.edges(data=True)]

            z['max_tortuosity'] = np.log(np.percentile(tortuosity,99.5))
            z['min_tortuosity'] = np.log(np.min(tortuosity))
            z['median_tortuosity'] = np.log(np.median(tortuosity))

            branch_angles = R.get_branch_angles()
            z['max_branch_angle'] = np.max(branch_angles)
            z['min_branch_angle'] = np.min(branch_angles)
            z['mean_branch_angle'] = np.mean(branch_angles)

            # get maximal degree within data
            z['max_degree'] = np.max([item[1] for item in R.get_graph().out_degree().items() if item[0] != R.get_root()])

            # get tree asymmetry
            weights, psad = R.get_psad()
            if np.sum(list(weights.values())) != 0:
                z['tree_asymmetry'] = np.sum([weights[k]*psad[k] for k in psad.keys()])/np.sum(list(weights.values()))
            else:
                z['tree_asymmetry'] = 0
                
            self.insert1(z)



@schema 
class MorphometricDistribution(dj.Computed):
    definition = """
    -> Morphometry
    -> MorphometryDistributionType
    ---
    data: longblob
    bin_edges: mediumblob
    """

    def _make_tuples(self, key):

        z = copy(key)
        print('Populating %s' % key)
        
        if z['part_id'] == 1:
            T = Neuron().get_tree(z)
        elif z['part_id'] == 2:
            T = (Neuron().Axon() & z).get_tree()
        elif z['part_id'] == 3:
            T = (Neuron().Dendrite() & z).get_tree()

        if T:
            # root angle distribution
            if z['morph_dist_name'] == 'root angle':

                hist, bin_edges = T.get_root_angle_dist(bins=20, normed=True)
                z['data'] = hist
                z['bin_edges'] = dict(zip([str(k) for k in range(len(bin_edges))], bin_edges))
                self.insert1(z)

            elif z['morph_dist_name'] == 'root angle euler':
                hist, bin_edges = T.get_root_angle_dist(angle_type='euler', bins=20, normed=True)
                z['data'] = hist
                z['bin_edges'] = dict(zip([str(k) for k in range(len(bin_edges))], bin_edges))
                self.insert1(z)

            elif z['morph_dist_name'] == 'thickness':
                z_ = copy(z)
                del z_['c_num']
                max_thickness = int(np.floor(np.max((Morphometry() & z_).fetch('max_thickness'))))

                hist, edges = T.get_thickness_dist(bins=30, range=(0, max_thickness), normed=True)
                z['data'] = hist
                z['bin_edges'] = dict(zip([str(k) for k in range(len(edges))], edges))
                self.insert1(z)

            elif z['morph_dist_name'] == 'thickness vs branch order':
                z_ = copy(z)
                del z_['c_num']
                max_thickness = int(np.floor(np.max((Morphometry() & z_).fetch('max_thickness'))))
                max_bo = np.max((Morphometry() & z_).fetch('max_branch_order'))

                hist, edges = T.get_thickness_dist(bins=[max_bo, 30], range=((0, max_bo), (0, max_thickness)),
                                                   dist_measure='branch_order', normed=True)

                z['data'] = hist
                z['bin_edges'] = dict(zip([str(k) for k in range(len(edges))], edges))
                self.insert1(z)

            elif z['morph_dist_name'] == 'thickness vs path dist':
                z_ = copy(z)
                del z_['c_num']
                max_thickness = int(np.floor(np.max((Morphometry() & z_).fetch('max_thickness'))))
                max_path = np.max((Morphometry() & z_).fetch('max_path_dist_to_soma'))

                hist, edges = T.get_thickness_dist(bins=[20, 30], range=((0, max_path/2), (0, max_thickness)),
                                                   dist_measure='path_from_soma', normed=True)

                z['data'] = hist
                z['bin_edges'] = dict(zip([str(k) for k in range(len(edges))], edges))
                self.insert1(z)

            elif z['morph_dist_name'] == 'sholl intersection xy':
                data, bins = T.get_sholl_intersection_profile(centroid='soma')
                z['data'] = data
                z['bin_edges'] = bins
                self.insert1(z)

            elif z['morph_dist_name'] == 'sholl intersection xz':
                data, bins = T.get_sholl_intersection_profile(proj='xz', centroid='soma')
                z['data'] = data
                z['bin_edges'] = bins
                self.insert1(z)

            elif z['morph_dist_name'] == 'sholl intersection yz':
                data, bins = T.get_sholl_intersection_profile(proj='yz', centroid='soma')
                z['data'] = data
                z['bin_edges'] = bins
                self.insert1(z)

            elif z['morph_dist_name'] == 'branch order':
                z_ = copy(z)
                del z_['c_num']
                max_bo = np.max((Morphometry() & z_).fetch('max_branch_order'))

                hist, edges = T.get_branch_order_dist(bins=max_bo, range=(0, max_bo), normed=True)
                z['data'] = hist
                z['bin_edges'] = dict(zip([str(k) for k in range(len(edges))], edges))
                self.insert1(z)

            elif z['morph_dist_name'] == 'branch angle':

                hist, edges = T.get_branch_angle_dist(bins=20, normed=True)
                z['data'] = hist
                z['bin_edges'] = dict(zip([str(k) for k in range(len(edges))], edges))
                self.insert1(z)
            elif z['morph_dist_name'] == 'branch angle vs branch order':
                z_ = copy(z)
                del z_['c_num']
                max_bo = np.max((Morphometry() & z_).fetch('max_branch_order'))

                hist, edges = T.get_branch_angle_dist(dist_measure='branch_order', bins=[max_bo, 20], range=((0, max_bo),
                                                                                                             (0, 180)),
                                                      normed=True)
                z['data'] = hist
                z['bin_edges'] = dict(zip([str(k) for k in range(len(edges))], edges))
                self.insert1(z)
            elif z['morph_dist_name'] == 'branch angle vs path dist':
                z_ = copy(z)
                del z_['c_num']
                max_path = np.max((Morphometry() & z_).fetch('max_path_dist_to_soma'))

                hist, edges = T.get_branch_angle_dist(dist_measure='path_from_soma', bins=[20, 20], range=((0, max_path/2),
                                                                                                           (0, 180)),
                                                      normed=True)
                z['data'] = hist
                z['bin_edges'] = dict(zip([str(k) for k in range(len(edges))], edges))
                self.insert1(z)
            elif z['morph_dist_name'] == 'path angle':
                hist, edges = T.get_path_angle_dist(bins=20, normed=True)
                z['data'] = hist
                z['bin_edges'] = dict(zip([str(k) for k in range(len(edges))], edges))
                self.insert1(z)
            elif z['morph_dist_name'] == 'path angle vs branch order':
                z_ = copy(z)
                del z_['c_num']
                max_bo = np.max((Morphometry() & z_).fetch('max_branch_order'))

                hist, edges = T.get_path_angle_dist(dist_measure='branch_order', bins=[max_bo, 20], range=((0, max_bo),
                                                                                                           (0, 180)),
                                                    normed=True)
                z['data'] = hist
                z['bin_edges'] = dict(zip([str(k) for k in range(len(edges))], edges))
                self.insert1(z)
            elif z['morph_dist_name'] == 'path angle vs path dist':
                z_ = copy(z)
                del z_['c_num']
                max_path = np.max((Morphometry() & z_).fetch('max_path_dist_to_soma'))

                hist, edges = T.get_path_angle_dist(dist_measure='path_from_soma', bins=[20, 20], range=((0, max_path / 2),
                                                                                                         (0, 180)),
                                                    normed=True)
                z['data'] = hist
                z['bin_edges'] = dict(zip([str(k) for k in range(len(edges))], edges))
                self.insert1(z)
            elif z['morph_dist_name'] == 'vertical dist to soma':

                z_ = copy(z)
                del z_['c_num']
                max_path = np.max((Morphometry() & z_).fetch('max_path_dist_to_soma'))

                hist, edges = T.get_distance_dist(bins=20, range=(0, max_path), normed=True)

                z['data'] = hist
                z['bin_edges'] = dict(zip([str(k) for k in range(len(edges))], edges))
                self.insert1(z)

            elif z['morph_dist_name'] == 'segment length':

                z_ = copy(z)
                del z_['c_num']
                max_segment_length = np.max((Morphometry() & z_).fetch('max_segment_path_length'))

                hist, edges = T.get_segment_length_dist(bins= 20,range=(0,max_segment_length), normed=True)
                z['data'] = hist
                z['bin_edges'] = dict(zip([str(k) for k in range(len(edges))], edges))
                self.insert1(z)
            elif z['morph_dist_name'] == 'segment path length vs branch order':
                z_ = copy(z)
                del z_['c_num']
                max_segment_length = np.max((Morphometry() & z_).fetch('max_segment_path_length'))
                max_bo = np.max((Morphometry() & z_).fetch('max_branch_order'))

                hist, edges = T.get_segment_length_dist(branch_order=True, bins=20,
                                                        range=[(0, max_bo), (0, max_segment_length)],
                                                        normed=True)

                z['data'] = hist
                z['bin_edges'] = dict(zip([str(k) for k in range(len(edges))], edges))
                self.insert1(z)

            else:
                raise NotImplementedError('The distribution type %s is not implemented!' % z['morph_dist_name'])

    def plot(self, key):

        assert len((self & key)) == 1
        hist = (self & key).fetch1('data')
        edges = (self & key).fetch1('bin_edges')
        if len(hist.shape) == 1:
            plt.figure()
            plt.plot(edges['0'][0][:-1], hist)
            sns.despine()
            plt.xlabel(key['morph_dist_name'])
            plt.ylabel('P(x)')

        elif len(hist.shape) == 2:
            plt.figure()

            if key['morph_dist_name'] == 'path angle vs branch order':
                xlabel = 'path angle [deg]'
                ylabel = 'branch order'

            elif key['morph_dist_name'] == 'path angle vs path dist':
                xlabel = 'path angle [deg]'
                ylabel = 'dist from soma [microns]'

            elif key['morph_dist_name'] == 'thickness vs path dist':
                xlabel = 'thickness [microns]'
                ylabel = 'dist from soma [microns]'

            elif key['morph_dist_name'] == 'thickness vs branch order':
                xlabel = 'thickness [microns]'
                ylabel = 'branch order'

            elif key['morph_dist_name'] == 'branch angle vs path dist':
                xlabel = 'branch angle [deg]'
                ylabel = 'dist from soma [microns]'

            elif key['morph_dist_name'] == 'branch angle vs branch order':
                xlabel = 'branch angle [deg]'
                ylabel = 'branch order'
            elif key['morph_dist_name'] == 'segment path length vs branch order':
                xlabel = 'segment path length'
                ylabel = 'branch order'

            plt.imshow(hist.T, extent=[0, edges['0'][0][-1], 0, edges['1'][0][-1]])
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.colorbar()

        elif len(hist.shape) == 3:
            x, y, z = np.meshgrid(edges['0'][0], edges['1'][0], edges['2'][0], indexing='ij')
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter3D(x, y, z, s=hist.flatten())

            plt.xlabel('x')
            plt.ylabel('y')
            plt.zlabel('z')








