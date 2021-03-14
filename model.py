import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
import torch_scatter
import networkx as nx

""" -------- Infrastructures -------- """

class Lattice(object):
    """ Hosts lattice information and construct causal graph
        
        Args:
        size: number of size along one dimension (assuming square/cubical lattice)
        dimension: dimension of the lattice
        max_distance: the maximum distance to construct causal relationships
    """
    def __init__(self, size:int, dimension:int, max_distance = None):
        self.center = 2
        self.size = size
        self.dimension = dimension
        self.shape = [size]*dimension
        self.sites = size**dimension
        self.tree_depth = self.sites.bit_length()
        self.node_init()
        self.node_centers_init()
        self.graph_init(max_distance)
        
    def __repr__(self):
        return 'Lattice({} grid with tree depth {})'.format(
                    'x'.join(str(L) for L in self.shape),
                     self.tree_depth)
    
    def node_init(self):
        """ Node initialization, calculate basic node information
            for other methods in this class to work.
            Called by class initialization. """
        self.node_index = torch.zeros(self.sites, dtype=torch.long)
        def partition(rng: torch.Tensor, dim: int, ind: int, lev: int):
            if rng[dim].sum()%2 == 0:
                mid = rng[dim].sum()//2
                rng1 = rng.clone()
                rng1[dim, 1] = mid
                rng2 = rng.clone()
                rng2[dim, 0] = mid
                partition(rng1, (dim + 1)%self.dimension, 2*ind, lev+1)
                partition(rng2, (dim + 1)%self.dimension, 2*ind + 1, lev+1)
            else:
                self.node_index[ind-self.sites] = rng[:,0].dot(self.size**torch.arange(0,self.dimension).flip(0))
        partition(torch.tensor([[0, self.size]]*self.dimension), 0, 1, 1)
        
    def node_centers_init(self):
        self.node_levels = torch.ones(1, dtype=torch.int)
        self.node_centers = torch.ones(1, self.dimension) * self.center
        def next_layer(num_layers, last_layer, dim = 0, scaling = 1, level = 2):
            if num_layers > 1:
                right = last_layer.clone()
                left = last_layer.clone()
                right[:, dim] = right[:, dim] + 1 * scaling
                left[:, dim] = left[:, dim] - 1 * scaling
                new_layer = torch.cat((left, right))
                self.node_centers = torch.cat((self.node_centers, new_layer))
                self.node_levels = torch.cat((self.node_levels, torch.ones(len(new_layer)) * level))
                level += 1
                dim = (dim + 1) % self.dimension
                next_layer(num_layers - 1, new_layer, dim , scaling if dim else scaling * 0.5, level)
        next_layer(self.tree_depth - 1, self.node_centers)
        self.node_levels = torch.cat((torch.zeros(1, dtype=torch.int), self.node_levels))
        self.node_centers = torch.cat((torch.zeros(1, self.dimension), self.node_centers))
        
    def graph_init(self, max_distance):
        graph = self.causal_graph(max_distance)
        graph = graph.to_sparse()
        self.graph = ExpandGraph((graph.indices(), graph.values()))
        self.max_depth = self.graph.get_depths().max()
        
    def _left_right_tree_edges(self, n):
        nodes = iter(range(2 ** n - 1))
        parents = [next(nodes)]
        new_parents = []
        while parents:
            for i in range(2):
                for parent in parents:
                    try:
                        child = next(nodes)
                        new_parents.append(child)
                        yield parent, child
                    except StopIteration:
                        break
            parents = new_parents
            new_parents = []
            
    def _autoregressive_pairs(self, n):
        num_nodes = 2 ** n - 1
        for source in range(num_nodes):
            for target in range(source, num_nodes):
                yield source, target

    def generate_left_right_tree(self, n):
        G = nx.empty_graph(2 ** n - 1, nx.DiGraph)
        G.add_edges_from(self._left_right_tree_edges(n))
        return G
        
    def torus_distances(self):
        col_i = self.node_centers[1:].unsqueeze(0)
        col_j = self.node_centers[1:].unsqueeze(1)
        differences = col_i - col_j
        torus_differences = (differences + self.center) % (self.center * 2) - self.center
        torus_distance_squared = torch.sum(torus_differences ** 2, dim = 2)
        rescaled_torus_distance = torus_distance_squared * 2 ** self.node_levels[1:]
        return rescaled_torus_distance
    
    def relationships(self):
        pairs = self._autoregressive_pairs(self.tree_depth - 1)
        binary_graph = self.generate_left_right_tree(self.tree_depth - 1)
        lca_iter = nx.all_pairs_lowest_common_ancestor(binary_graph, pairs = pairs)
        cords = [[], []]
        lowest_common_ancestor = []
        
        for cord, lca in lca_iter:
            cords[0].append(cord[0])
            cords[1].append(cord[1])
            lowest_common_ancestor.append(lca)

        tensor_cords = torch.tensor(cords).type(torch.long)
        tensor_lowest_common_ancestor = torch.tensor(lowest_common_ancestor).type(torch.long)
        cord_levels = self.node_levels[1:][tensor_cords]
        lowest_common_ancestor_levels = self.node_levels[1:][tensor_lowest_common_ancestor]
        relative_levels = cord_levels - lowest_common_ancestor_levels
        relationship_types = relative_levels[0] * (self.tree_depth - 1) + relative_levels[1]
        _ , relationship_types = relationship_types.unique(return_inverse = True)
        relationship_types += 1
        return torch.sparse_coo_tensor(tensor_cords, relationship_types).to_dense()
    
    def causal_graph(self, max_distance = None):
        
        relationship = self.relationships()
        if max_distance == None:
            return relationship
        distances = self.torus_distances()
        relationship[distances > max_distance] = 0
        return relationship
    
class AutoregressiveLattice(Lattice):
    """ Hosts lattice information and constructs a desnse autoregressive causal graph
        
        Args:
        size: number of size along one dimension (assuming square/cubical lattice)
        dimension: dimension of the lattice
    """
    def __init__(self, size:int, dimension:int, extend = 0):
        super().__init__(size, dimension)
        self.extend = extend
    def causal_graph(self):
        graph = torch.stack((torch.arange(1,self.sites + self.extend), torch.arange(1,self.sites + self.extend),
                             torch.zeros(self.sites - 1 + self.extend, dtype=int)))
        for i in range(1, self.sites - 1 + self.extend):
            graph = torch.cat((graph, torch.stack((i * torch.ones(self.sites - (i + 1) + self.extend, dtype = int),
                                                   torch.arange(i + 1, self.sites + self.extend),
                                                   torch.ones(self.sites - (i + 1) + self.extend, dtype = int)))), 1)
        return graph
    
class Graph(object):
    
    def __init__(self, connection_graph, encode = False):
        """Constructs a causal graph from a connection_graph structure
        
        args:
        connection_graph: a graph with connections and their connection types
        encode: a boolean value indicating wether to encode the connection types"""
        
        assert len(connection_graph[0][0]) == len(connection_graph[1]), "Size Mismatch"
        
        if len(connection_graph[0][0]) != 0:
            self.edge_types = len(connection_graph[1].unique())
            self.max_node = connection_graph[0].max().item()
            if encode:
                encoded_connection_types = self.encode_connection_types(connection_graph[1])
                connection_graph = (connection_graph[0], encoded_connection_types)
        self.connection_graph = connection_graph
        self.num_edges = len(connection_graph[0][0])
        
    @classmethod
    def from_python_lists(cls, in_connections, out_connections, connection_types, encode = False):
        """Constructs a causal graph from specified arguments
        
        args:
        in_connections: a list of where connections start
        out_connections: a list of where connections end
        connection_types: a list of types of connection
        encode: a boolean value indicating wether to encode the connection types"""
        
            
        obj = cls((torch.tensor([in_connections, out_connections]), torch.tensor(connection_types)), encode)
        return obj
        
    def encode_connection_types(self, connection_types):
        """encodes connection_types as consecutive numbers from start
        
        args:
        connection_types: list of the different connection types
        start: integer indicating where to start the encoding"""
        types = connection_types.unique()
        if min(types) < 0 or max(types) >= len(types):
            for encoding, connection_type in enumerate(types):
                connection_types[connection_types == connection_type] = encoding
        return connection_types
    
    def expand_features(self, in_expansion=1, out_expansion=1):
        #expansion of the in features
        out_features = self.connection_graph[0][1].repeat(in_expansion)
        in_features = self.connection_graph[0][0] * in_expansion
        in_features = in_features + torch.arange(0, in_expansion).unsqueeze(1)
        in_features = in_features.flatten()
        weights = self.connection_graph[1] * in_expansion
        weights = weights + torch.arange(0, in_expansion).unsqueeze(1)
        weights = weights.flatten()
    
        #expansion of the out features
        in_features = in_features.repeat(out_expansion)
        out_features = out_features * out_expansion
        out_features = out_features + torch.arange(0, out_expansion).unsqueeze(1)
        out_features = out_features.flatten()
        weights = weights * out_expansion
        weights = weights + torch.arange(0, out_expansion).unsqueeze(1)
        weights = weights.flatten()
        connection_graph = (torch.stack((in_features, out_features)), weights)
        return Graph(connection_graph, encode = True)
    
    def get_adj_matrix(self):
        sparse_graph = self.inverse_connections().sparse_graph()
        adj_size = self.max_node + 1
        return torch.sparse.FloatTensor(sparse_graph[0], torch.ones(self.num_edges),
                                        torch.Size([adj_size, adj_size])).to_dense().int()
    
    def get_depths(self):                                                                            
        """creates a list of the causal depths if there are no loops in the graph otherwise returns None"""
        depths = torch.zeros(self.max_node + 1, 1, dtype=torch.int32)
        depth_determiner = torch.ones(self.max_node + 1, 1, dtype=torch.int32)
        adj_matrix = self.remove_self_loops().get_adj_matrix()
        """torch.set_printoptions(profile="full")
        print("ADJ:", adj_matrix)
        torch.set_printoptions(profile="default")"""
        
        prev_parents = self.max_node + 1
        num_parents = self.max_node + 1
        while num_parents:
            depth_determiner = (adj_matrix.mm(depth_determiner) > 0).int()
            num_parents = sum(depth_determiner)
            if num_parents == prev_parents:
                return None
            prev_parents = num_parents
            depths += depth_determiner
        return depths.flatten()
    
    def get_depth_assignment(self):
        """creates a list grouping the different depths of each site"""
        depths = self.get_depths()
        if depths is None:
            return None
        depth_assignment = [[] for _ in range(max(depths) + 1)]
        for i, depth in enumerate(depths):
            depth_assignment[depth].append(i)
        return depth_assignment
    
    def get_edges(self):
        """creates a set of connections with out connection type"""
        return self.connection_graph[0]
    
    def get_max_nodes(self):
        max_row = self.connection_graph[0][0].max().item()
        max_column = self.connection_graph[0][1].max().item()
        return max_row, max_column
            
    def inverse_connections(self):
        """inverts directionality of edges in the graph"""
        inverse_connection_graph = (self.connection_graph[0].roll(1, 0) , self.connection_graph[1])
        return Graph(inverse_connection_graph)
        
    def is_self_loop(self):
        """checks if graph has any self loops"""
        return torch.all(self.connection_graph[0][0] != self.connection_graph[0][1])
    
    def remove_self_loops(self):
        """removes all self loops from the graph"""
        not_self_loops = self.connection_graph[0][0] != self.connection_graph[0][1]
        return Graph(self.take_connections(not_self_loops))
    
    def select_connections(self, *args, out = False):
        """creates a graph with specified in or out features
        
        args: connections to select
        out: boolean indicating to select by in feature or out feature
        """
        graph_selection = torch.tensor([node.item() in args for node in self.connection_graph[0][1 if out else 0]])
        
        return Graph(self.take_connections(graph_selection))
        
    def set_connection_graph(self, connection_graph):
        """updates connection_graph, num_edges, and weights based on input connection graph
        
        args:
        connection_graph: connection graph to set self.connection_graph to"""
        self.connection_graph = connection_graph
        self.num_edges = len(self.connection_graph[1])
        self.edge_types = len(torch.unique(self.connection_graph[1]))
        self.max_node = max(max(self.connection_graph[0][0]), max(self.connection_graph[0][1])).item()
        
    def sparse_graph(self):
        """creates a sparse graph for pytorch sparse tensors"""
            
        return self.connection_graph[0], self.connection_graph[1]
    
    def take_connections(self, bool_tensor):
        selected_graph = (self.connection_graph[0][:, bool_tensor], self.connection_graph[1][bool_tensor])
        return selected_graph
    
    def __repr__(self):
        """representation of connection_graph sorted by connection_type"""
        return "{0}\n{1}".format(self.connection_graph[0], self.connection_graph[1])
    
class ExpandGraph(Graph):
    
    def __init__(self, connection_graph, encode = False, feats = (1, 1)):
        super().__init__(connection_graph, encode)
        self.feats = feats
        
    def expand_features(self, in_expansion=1, out_expansion=1):
        #expansion of the in features
        out_features = self.connection_graph[0][1].repeat(in_expansion)
        in_features = self.connection_graph[0][0] * in_expansion
        in_features = in_features + torch.arange(0, in_expansion).unsqueeze(1)
        in_features = in_features.flatten()
        weights = self.connection_graph[1] * in_expansion
        weights = weights + torch.arange(0, in_expansion).unsqueeze(1)
        weights = weights.flatten()
    
        #expansion of the out features
        in_features = in_features.repeat(out_expansion)
        out_features = out_features * out_expansion
        out_features = out_features + torch.arange(0, out_expansion).unsqueeze(1)
        out_features = out_features.flatten()
        weights = weights * out_expansion
        weights = weights + torch.arange(0, out_expansion).unsqueeze(1)
        weights = weights.flatten()
        connection_graph = (torch.stack((in_features, out_features)), weights)
        return ExpandGraph(connection_graph, encode = True, feats = (in_expansion, out_expansion))
    
    def remove_self_loops(self):
        bound = self.connection_graph[0][0] // self.feats[0]
        is_self_loop = torch.le(bound * self.feats[1], self.connection_graph[0][1])
        is_self_loop = (is_self_loop) & (torch.lt(self.connection_graph[0][1], (bound + 1) * self.feats[1]))
        return ExpandGraph(self.take_connections(~is_self_loop), feats = self.feats)

class Group(object):
    """Represent a group, providing multiplication and inverse operation.
    
    Args:
    mul_table: multiplication table as a tensor, e.g. Z2 group: tensor([[0,1],[1,0]])
    """
    def __init__(self, mul_table: torch.Tensor):
        super(Group, self).__init__()
        self.mul_table = mul_table
        self.order = mul_table.size(0) # number of group elements
        gs, ginvs = torch.nonzero(self.mul_table == 0, as_tuple=True)
        self.inv_table = torch.gather(ginvs, 0, gs)
        self.val_table = None
    
    def __iter__(self):
        return iter(range(self.order))
    
    def __repr__(self):
        return 'Group({} elements)'.format(self.order)
    
    def inv(self, input: torch.Tensor):
        return torch.gather(self.inv_table.expand(input.size()[:-1]+(-1,)), -1, input)
    
    def mul(self, input1: torch.Tensor, input2: torch.Tensor):
        output = input1 * self.order + input2
        return torch.gather(self.mul_table.flatten().expand(output.size()[:-1]+(-1,)), -1, output)
    
    def prod(self, input, dim: int, keepdim: bool = False):
        input_size = input.size()
        flat_mul_table = self.mul_table.flatten().expand(input_size[:dim]+input_size[dim+1:-1]+(-1,))
        output = input.select(dim, 0)
        for i in range(1, input.size(dim)):
            output = output * self.order + input.select(dim, i)
            output = torch.gather(flat_mul_table, -1, output)
        if keepdim:
            output = output.unsqueeze(dim)
        return output
    
    def val(self, input, val_table = None):
        if val_table is None:
            val_table = self.default_val_table()
        elif len(val_table) != self.order:
            raise ValueError('Group function value table must be of the same size as the group order, expect {} got {}.'.format(self.order, len(val_table)))
        return torch.gather(val_table.expand(input.size()[:-1]+(-1,)), -1, input)

    def default_val_table(self):
        if self.val_table is None:
            self.val_table = torch.zeros(self.order)
            self.val_table[0] = 1.
        return self.val_table

class SymmetricGroup(Group):
    """ Represent a permutation group """
    def __init__(self, n: int):
        self.elements = list(itertools.permutations(range(n), n))
        index = {g:i for i, g in enumerate(self.elements)}
        mul_table = torch.empty([len(self.elements)]*2, dtype=torch.long)
        for g1 in self.elements:
            for g2 in self.elements:
                g = tuple(g1[a] for a in g2)
                mul_table[index[g1], index[g2]] = index[g]
        super(SymmetricGroup, self).__init__(mul_table)

    def default_val_table(self):
        if self.val_table is None:
            def cycle_number(g):
                if len(g) == 0:
                    return 0
                elif g[0] == 0:
                    return cycle_number(tuple(a - 1 for a in g[1:])) + 1
                else:
                    return cycle_number(tuple(g[0] - 1 if a == 0 else a - 1 for a in g[1:]))
            self.val_table = torch.tensor([cycle_number(g) for g in self.elements], dtype=torch.float)
        return self.val_table


""" -------- Energy Model -------- """

class EnergyTerm(nn.Module):
    """ represent an energy term"""
    strength = 1.
    group = None
    lattice = None
    def __init__(self):
        super(EnergyTerm, self).__init__()
        
    def __mul__(self, other):
        self.strength *= other
        return self
    
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * (-1)
    
    def __add__(self, other):
        if isinstance(other, EnergyTerm):
            return EnergyTerms([self, other])
        elif isinstance(other, EnergyTerms):
            return other.append(self)
        
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (- other)
    
    def __rsub__(self, other):
        return (- self) + other
    
    def extra_repr(self):
        return '{}'.format(self.strength)
        
    def on(self, group: Group = None, lattice: Lattice = None):
        self.group = group
        self.lattice = lattice
        return self
        
    def forward(self):
        if self.group is None:
            raise RuntimeError('A group structure has not been linked before forward evaluation of the energy term. Call self.on(group = group) to link a Group.')
        if self.lattice is None:
            raise RuntimeError('A lattice system has not been linked before forward evaluation of the energy term. Call self.on(lattice = lattice) to link a Lattice.')

class EnergyTerms(nn.ModuleList):
    """ represent a sum of energy terms"""
    def __init__(self, *arg):
        super(EnergyTerms, self).__init__(*arg)
    
    def __mul__(self, other):
        for term in self:
            term = term * other
        return self
        
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * (-1)
    
    def on(self, group: Group = None, lattice: Lattice = None):
        for term in self:
            term.on(group, lattice)
        return self
    
    def forward(self, input):
        return sum(term(input) for term in self)

class OnSite(EnergyTerm):
    """ on-site energy term """
    def __init__(self, val_table = None):
        super(OnSite, self).__init__()
        self.val_table = val_table
    
    def extra_repr(self):
        if not self.val_table is None:
            return 'G -> {}'.format((self.val_table * self.strength).tolist())
        else:
            return super(OnSite, self).extra_repr()
    
    def forward(self, input):
        super(OnSite, self).forward()
        dims = tuple(range(-self.lattice.dimension,0))
        energy = self.group.val(input, self.val_table) * self.strength
        return energy.sum(dims)
    
class TwoBody(EnergyTerm):
    """ two-body interaction term """
    def __init__(self, val_table = None, shifts = None):
        super(TwoBody, self).__init__()
        self.val_table = val_table
        self.shifts = shifts
        
    def extra_repr(self):
        if not self.val_table is None:
            return 'G -> {} across {}'.format(
                (self.val_table * self.strength).tolist(),
                self.shifts if not self.shifts is None else '(0,...)')
        elif not self.shifts is None:
            return '{} across {}'. format(
                self.strength,
                self.shifts)
        else:
            return super(TwoBody, self).extra_repr()
        
    def forward(self, input):
        super(TwoBody, self).forward()
        dims = tuple(range(-self.lattice.dimension,0))
        if self.shifts is None:
            self.shifts = (0,)*self.lattice.dimension
        rolled = self.group.inv(input.roll(self.shifts, dims))
        coupled = self.group.mul(rolled, input)
        energy = self.group.val(coupled, self.val_table) * self.strength
        return energy.sum(dims)

class EnergyModel(nn.Module):
    """ Energy mdoel that describes the physical system. Provides function to evaluate energy.
    
        Args:
        energy: lattice Hamiltonian in terms of energy terms
        group: a specifying the group on each site
        lattice: a lattice system containing information of the group and lattice shape
    """
    def __init__(self, energy: EnergyTerms, group: Group, lattice: Lattice):
        super(EnergyModel, self).__init__()
        self.group = group
        self.lattice = lattice
        self.update(energy)
    
    def extra_repr(self):
        return '(group): {}\n(lattice): {}'.format(self.group, self.lattice) + super(EnergyModel, self).extra_repr()
        
    def forward(self, input):
        return self.energy(input)

    def update(self, energy):
        self.energy = energy.on(self.group, self.lattice)

""" -------- Transformations -------- """

class HaarTransform(dist.Transform):
    """ Haar wavelet transformation (bijective)
        transformation takes real space configurations x to wavelet space encoding y
    
        Args:
        group: a group structure for each unit
        lattice: a lattice system containing information of the group and lattice shape
    """
    def __init__(self, group: Group, lattice: Lattice):
        super(HaarTransform, self).__init__()
        self.group = group
        self.lattice = lattice
        self.bijective = True
        self.make_wavelet()
        
    # construct Haar wavelet basis
    def make_wavelet(self):
        self.wavelet = torch.zeros(torch.Size([self.lattice.sites, self.lattice.sites]), dtype=torch.int)
        self.wavelet[0] = 1
        for z in range(1,self.lattice.tree_depth):
            block_size = 2**(z-1)
            for q in range(block_size):
                node_range = 2**(self.lattice.tree_depth-1-z) * torch.tensor([2*q+1,2*q+2])
                nodes = torch.arange(*node_range)
                sites = self.lattice.node_index[nodes]
                self.wavelet[block_size + q, sites] = 1 
                
    def _call(self, z):
        x = self.group.prod(z.unsqueeze(-1) * self.wavelet, -2)
        return x.view(z.size()[:-1]+torch.Size(self.lattice.shape))
    
    def _inverse(self, x):
        y = x.flatten(-self.lattice.dimension)[...,self.lattice.node_index]
        def renormalize(y):
            if y.size(-1) > 1:
                y0 = y[...,0::2]
                y1 = y[...,1::2]
                return torch.cat((renormalize(y0), self.group.mul(self.group.inv(y0), y1)), -1)
            else:
                return y
        z = renormalize(y)
        return z
    
    def log_abs_det_jacobian(self, x, y):
        return torch.tensor(0.)

class TwoDimTransform(dist.Transform):
    """ One dimensional to two dimensional transform (bijective)
        transformation takes from a one dimensional structure to a two dimensional structure
    
        Args:
        group: a group structure for each unit
        lattice: a lattice system containing information of the group and lattice shape
    """
    
    def __init__(self, group: Group, lattice: Lattice):
        super(TwoDimTransform, self).__init__()
        self.group = group
        self.lattice = lattice
        self.bijective = True
        
    def _call(self, z):
        return z.view(z.size()[:-1]+torch.Size(self.lattice.shape))
    
    def _inverse(self, x):
        return x.flatten(-self.lattice.dimension)[...,self.lattice.node_index]
    
    def log_abs_det_jacobian(self, x, y):
        return torch.tensor(0.)
    
class OneHotCategoricalTransform(dist.Transform):
    """Convert between one-hot and categorical representations.
    
    Args:
    num_classes: number of classes."""
    def __init__(self, num_classes: int):
        super(OneHotCategoricalTransform, self).__init__()
        self.num_classes = num_classes
        self.bijective = True
    
    def _call(self, x):
        # one-hot to categorical
        return x.max(dim=-1)[1]
    
    def _inverse(self, y):
        # categorical to one-hot
        return F.one_hot(y, self.num_classes).to(dtype=torch.float)
    
    def log_abs_det_jacobian(self, x, y):
        return torch.tensor(0.)

""" -------- Base Distribution -------- """

class GraphConv(nn.Module):
    """ Graph Convolution layer 
        
        Args:
        graph: tensor of shape [3, num_edges] 
               specifying (source, target, type) along each column
        in_features: number of input features (per node)
        out_features: number of output features (per node)
        bias: whether to learn an edge-depenent bias
        self_loop: whether to include self loops in message passing
    """
    def __init__(self, lattice, in_features: int, out_features: int,
                 bias: bool = True, self_loop: bool = True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = bias
        else:
            self.register_parameter('bias', None)
        self.edge_types = None
        self.self_loop = self_loop
        self.lattice_sites = lattice.sites
        self.update_graph(lattice.graph)
        self.conv_size = (self.lattice_sites * out_features, self.lattice_sites * in_features)

    def update_graph(self, graph):
        # update the graph, adding new linear maps if needed
        if not self.self_loop:
            graph = graph.remove_self_loops()#removes any self_loops according to boolean
        self.graph = graph.expand_features(self.in_features, self.out_features)#expands the graph features
        self.graph = self.graph.inverse_connections()
        self.depth_assignment = graph.get_depth_assignment()
        self.forwarding_graphs_init()
        edge_types = self.graph.edge_types
        if edge_types != self.edge_types:
            self.weight = nn.Parameter(torch.Tensor(edge_types))
            if self.bias is not None:
                self.bias = nn.Parameter(torch.Tensor(edge_types, self.out_features))
            self.reset_parameters()
        self.edge_types = edge_types
        return self
    
    def forwarding_graphs_init(self):
        self.forwarding_graphs = []
        for depth in self.depth_assignment:
            self.forwarding_graphs.append(self.graph.select_connections(*depth))

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.lattice_sites * self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def extra_repr(self):
        return 'edge_types={}, in_features={}, out_features={}, bias={}, self_loop={}'.format(
            self.edge_types, self.in_features, self.out_features, self.bias is not None, self.self_loop)

    def forward(self, input, depth = None):
        input = input.flatten(1).t()
        if depth == None:
            signal, edge_type = self.graph.sparse_graph()
            weights = torch.gather(self.weight, 0, edge_type)
            conv = torch.sparse_coo_tensor(signal, weights, size = self.conv_size)
            output = torch.sparse.mm(conv, input)
        else:
            signal, edge_type = self.forwarding_graphs[depth].sparse_graph()
            weights = torch.gather(self.weight, 0, edge_type)
            conv = torch.sparse_coo_tensor(signal, weights, size = self.conv_size)
            output = torch.sparse.mm(conv, input)
        output = output.t().unflatten(1, (self.lattice_sites, self.out_features))
        #if self.bias:
        
        return output

class AutoregressiveModel(nn.Module, dist.Distribution):
    """ Represent a generative model that can generate samples and evaluate log probabilities.
        
        Args:
        lattice: lattice system
        features: a list of feature dimensions for all layers
        nonlinearity: activation function to use 
        bias: whether to learn the bias
    """
    
    def __init__(self, lattice: Lattice, features, nonlinearity: str = 'Tanh', bias: bool = True):
        super(AutoregressiveModel, self).__init__()
        self.lattice = lattice
        self.nodes = lattice.sites
        self.max_depth = lattice.max_depth
        self.features = features
        dist.Distribution.__init__(self, event_shape=torch.Size([self.nodes, self.features[0]]))
        self.has_rsample = True
        #self.graph = self.lattice.graph
        self.layers = nn.ModuleList()
        for l in range(1, len(self.features)):
            if l == 1: # the first layer should not have self loops
                self.layers.append(GraphConv(self.lattice, self.features[0], self.features[1], bias, self_loop = False))
            else: # remaining layers are normal
                self.layers.append(nn.LayerNorm([self.features[l - 1]]))
                self.layers.append(getattr(nn, nonlinearity)()) # activatioin layer
                self.layers.append(GraphConv(self.lattice, self.features[l - 1], self.features[l], bias))

    def update_graph(self, graph):
        # update graph for all GraphConv layers
        self.graph = graph
        for layer in self.layers:
            if isinstance(layer, GraphConv):
                layer.update_graph(graph)
        return self

    def forward(self, input):
        output = input
        for layer in self.layers: # apply layers
            output = layer(output)
        return output # logits
    
    def log_prob(self, sample):
        logits = self(sample) # forward pass to get logits
        return torch.sum(sample * F.log_softmax(logits, dim=-1), (-2,-1))

    def sampler(self, logits, dim=-1): # simplified from F.gumbel_softmax
        gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        gumbels += logits.detach()
        index = gumbels.max(dim, keepdim=True)[1]
        return torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)

    def _sample(self, sample_size: int, sampler = None):
        if sampler is None: # if no sampler specified, use default
            sampler = self.sampler
        # create a list of tensors to cache layer-wise outputs
        cache = [torch.zeros(sample_size, self.nodes, self.features[0])]
        for layer in self.layers:
            if isinstance(layer, GraphConv): # for graph convolution layers
                features = layer.out_features # features get updated
            cache.append(torch.zeros(sample_size, self.nodes, features))
        # cache established. start by sampling node 0.
        # assuming global symmetry, node 0 is always sampled uniformly
        cache[0][..., 0, :] = sampler(cache[0][..., 0, :])
        # start autoregressive sampling
        for j in range(1, self.max_depth + 1): # iterate through nodes 1:all
            for l, layer in enumerate(self.layers):
                if isinstance(layer, GraphConv): # for graph convolution layers
                    if l==0: # first layer should forward from previous node
                        cache[l + 1] += layer(cache[l], j - 1)#possibly change to not be in place operation
                    else: # remaining layers forward from this node
                        cache[l + 1] += layer(cache[l], j)
                else: # for other layers, only update node j (other nodes not ready yet)
                    src = layer(cache[l][..., [j], :])
                    index = torch.tensor(j).view([1]*src.dim()).expand(src.size())
                    cache[l + 1] = cache[l + 1].scatter(-2, index, src)#scatter incorrect
            # the last cache hosts the logit, sample from it 
            cache[0][..., j, :] = sampler(cache[-1][..., j, :])
        return cache # cache[0] hosts the sample
    
    def sample(self, sample_size=1):
        with torch.no_grad():
            cache = self._sample(sample_size)
        return cache[0]
    
    def rsample(self, sample_size=1, tau=None, hard=False):
        # reparametrized Gumbel sampling
        if tau is None: # if temperature not given
            tau = 1/(self.features[-1]-1) # set by the out feature dimension
        cache = self._sample(sample_size, lambda x: F.gumbel_softmax(x, tau, hard))
        return cache[0]

    def sample_with_log_prob(self, sample_size=1):
        cache = self._sample(sample_size)
        sample = cache[0]
        logits = cache[-1]
        log_prob = torch.sum(sample * F.log_softmax(logits, dim=-1), (-2,-1))
        return sample, log_prob


""" -------- Model Interface -------- """

class HolographicPixelGCN(nn.Module, dist.TransformedDistribution):
    """ Combination of hierarchical autoregressive and flow-based model for lattice models.
    
        Args:
        energy: a energy model to learn
        hidden_features: a list of feature dimensions of hidden layers
        nonlinearity: activation function to use 
        bias: whether to learn the additive bias in heap linear layers
    """
    def __init__(self, energy: EnergyModel, hidden_features, nonlinearity: str = 'Tanh', bias: bool = True):
        super(HolographicPixelGCN, self).__init__()
        self.energy = energy
        self.group = energy.group
        self.lattice = energy.lattice
        self.haar = HaarTransform(self.group, self.lattice)
        self.onecat = OneHotCategoricalTransform(self.group.order)
        features = [self.group.order] + hidden_features + [self.group.order]
        auto = AutoregressiveModel(self.lattice, features, nonlinearity, bias)
        dist.TransformedDistribution.__init__(self, auto, [self.onecat, self.haar])
        self.transform = dist.ComposeTransform(self.transforms)
        
    def free_energy(self, x):
        dims = tuple(range(-self.lattice.dimension,0))
        translation = itertools.product(*([range(self.lattice.size)]*self.lattice.dimension))
        log_probs = torch.stack([self.log_prob(x.roll(shifts=shifts, dims=dims)) for shifts in translation])
        """
        x = torch.rot90(x, 1, [-1, -2])
        x = torch.transpose(x, -1, -2)
        translation = itertools.product(*([range(self.lattice.size)]*self.lattice.dimension))
        transposed_log_probs = torch.stack([self.log_prob(x.roll(shifts=shifts, dims=dims)) for shifts in translation])
        log_probs = torch.cat((log_probs, transposed_log_probs), -2)
         """
        log_prob = log_probs.logsumexp(0) - math.log(2 * self.lattice.sites)
        return self.energy(x) + log_prob

class NoHaarHolographicPixelGCN(nn.Module, dist.TransformedDistribution):
    """ Combination of hierarchical autoregressive and flow-based model for lattice models with out a Haar transformation.
    
        Args:
        energy: a energy model to learn
        hidden_features: a list of feature dimensions of hidden layers
        nonlinearity: activation function to use 
        bias: whether to learn the additive bias in heap linear layers
    """
    def __init__(self, energy: EnergyModel, hidden_features, nonlinearity: str = 'Tanh', bias: bool = True, rotate: bool = False):
        super(NoHaarHolographicPixelGCN, self).__init__()
        self.energy = energy
        self.group = energy.group
        self.lattice = energy.lattice
        self.twodim = TwoDimTransform(self.group, self.lattice)
        self.onecat = OneHotCategoricalTransform(self.group.order)
        features = [self.group.order] + hidden_features + [self.group.order]
        auto = AutoregressiveModel(self.lattice, features, nonlinearity, bias)
        dist.TransformedDistribution.__init__(self, auto, [self.onecat, self.twodim])
        self.transform = dist.ComposeTransform(self.transforms)
        
    def free_energy(self, x):
        dims = tuple(range(-self.lattice.dimension,0))
        translation = itertools.product(*([range(self.lattice.size)]*self.lattice.dimension))
        log_probs = torch.stack([self.log_prob(x.roll(shifts=shifts, dims=dims)) for shifts in translation])
        """
        x = torch.rot90(x, 1, [-1, -2])
        x = torch.transpose(x, -1, -2)
        translation = itertools.product(*([range(self.lattice.size)]*self.lattice.dimension))
        transposed_log_probs = torch.stack([self.log_prob(x.roll(shifts=shifts, dims=dims)) for shifts in translation])
        log_probs = torch.cat((log_probs, transposed_log_probs), -2)
        """
        log_prob = log_probs.logsumexp(0) - math.log(self.lattice.sites)
        return self.energy(x) + log_prob
