#%reset -f

import numpy as np
from geometry_alg import RT, Graph, find_elements, solid_angle_tetra
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class concave_hull_3D(object):
    def __init__(self, 
                 name = None, 
                 points = None, 
                 weights = None,
                 alpha = -1,
                 alpha_step = 0.1):
        """
        Initialize a 3D concave hull class. Use self.construct_conchull to compute concave hull after initialization.
        Parameters:
            points - np.array of shape (n,3) points.
            weights - weights for the points, applicable in weighted alpha shapes (regular triangulation)
            alpha - alpha value. 
                    alpha = np.inf: compute convex hull
                    alpha = -1 (default): compute an optimal concave hull that only have one connected component
                    alpha = 0: compute an optimal concave hull that encloses all the points
                    otherwise: compute a concave hull using the provided alpha radius
            alpha_step - steps for tuning alpha radius if alpha = 0
        """
        self.name = name
        self.points = points
        self.weights = weights
        self.alpha = alpha
        self.alpha_step = alpha_step
        self.alpha_optimal = 0
        self.tri_trim = []
        self.Vertices = []
        self.Edges = []
        self.Triangles = []
    
    def find_circum_radii_cm(self, triang):
        """
        Compute the radii of circumspheres for each tetrahedron in a triangulation, 
        using the Miroslav Fiedler's method.
        Parameters:
            triang - a scipy.spatial.Delaunay or RT class object
        returns
            the list of radii of circumspheres for each tetrahedron in triang
        """
        r = []
        for t in triang.simplices:
            ones_v = np.ones(shape = (4, 1))
            ones_h = np.array([0, 1, 1, 1, 1])
            tpos = np.take(self.points, t, axis = 0)
            cm = np.vstack((ones_h, 
                            np.concatenate((ones_v, cdist(tpos, tpos)), axis = 1)))
            m = -2 * np.linalg.inv(cm)
            r.append(1/2 * np.sqrt(m[0, 0]))
        return np.array(r)

    def construct_conchull(self, alpha = None, alpha_step = None):
        """
        Compute the concave hull (alpha shape) of a set of 3D points based on Delaunay triangulation.
        Parameters:
            alpha - alpha value. if equals None, self.alpha will be used.
                    alpha = np.inf: compute convex hull
                    alpha = -1 (default): compute an optimal concave hull that only have one connected component
                    alpha = 0: compute an optimal concave hull that encloses all the points
                    otherwise: compute a concave hull using the provided alpha radius
        """
        print('Constructing concave hull (alpha shape) for the molecule...')
        if alpha is not None:
            self.alpha = alpha
        if alpha_step is not None:
            self.alpha_step = alpha_step
            
        # Build Delaunay or regular triangulation and find radii of the circumspheres for each tetrahedron        
        if self.weights is None:
            Triangulation = Delaunay(self.points)
        else:
            Triangulation = RT(points = self.points, weights = self.weights)
        r = self.find_circum_radii_cm(Triangulation)
        
        if self.alpha == np.inf:
            self.tri_trim = Triangulation.simplices
            self.Vertices, self.Edges, self.Triangles = find_elements(self.tri_trim)
        elif self.alpha == -1:
            alpha_try = np.sort(np.unique(r))[0]
            sign = 0
            while sign == 0:
                alpha_try = alpha_try + self.alpha_step
                tri_trim_try = Triangulation.simplices[r < alpha_try, :]
                ver, edg, tri = find_elements(tri_trim_try)
                graph = Graph(V = ver.shape[0])
                for edge in range(edg.shape[0]):
                    i, j = edg[edge]
                    ind1 = np.where(ver == i)[0][0]
                    ind2 = np.where(ver == j)[0][0]
                    graph.addEdge(ind1, ind2)
                cc_num = graph.NumOfConnectedComponents()
                #print('Tuning alpha value: %lf, number of connected components: %lf, number of enclosed points: %d (%d)' % (alpha_try, cc_num, ver.shape[0], self.points.shape[0]))
                if cc_num == 1:
                    self.tri_trim = tri_trim_try
                    self.alpha_optimal = alpha_try
                    self.Vertices, self.Edges, self.Triangles = ver, edg, tri
                    sign = 1            
        elif self.alpha == 0:
            alpha_try = np.sort(np.unique(r))[0]
            num_enc_pts = 1
            num_enc_pts_old = 0            
            while num_enc_pts >= num_enc_pts_old:
                alpha_try = alpha_try + self.alpha_step
                tri_trim_try = Triangulation.simplices[r < alpha_try, :]
                ver, edg, tri = find_elements(tri_trim_try)
                #print('Tuning alpha value: %lf, number of enclosed points: %d (%d)' % (alpha_try, ver.shape[0], self.points.shape[0]))
                if ver.shape[0] >= num_enc_pts:
                    num_enc_pts_old = num_enc_pts
                    num_enc_pts = ver.shape[0]
                    self.alpha_optimal = alpha_try
                else:
                    num_enc_pts = 0                
            self.tri_trim = Triangulation.simplices[r < self.alpha_optimal, :]
            self.Vertices, self.Edges, self.Triangles = find_elements(self.tri_trim)
        else: 
            if np.all(r > self.alpha):
                print("Alpha radius is set too small, no connected regions found!")
            else:
                # trim the original triangulation
                self.tri_trim = Triangulation.simplices[r < self.alpha, :]
                self.Vertices, self.Edges, self.Triangles = find_elements(self.tri_trim)

    def compute_solid_angles(self, query = None):
        """
        Compute the solid angles of a list of points.
        Parameters:
            query - point indices in self.points
        return
            a dicitonary of the (index, solid angle) pairs
        """
        sa_dict = {}
        aids = range(self.points.shape[0]) if query is None else query
        if self.Triangles != []:
            for point_ind in aids:    
                (tetras, tops) = np.where(self.tri_trim == point_ind)
                if len(tetras) > 0:
                    sol_angs = []
                    for tetra, top in zip(tetras, tops):
                        sol_angs.append(solid_angle_tetra(self.points[self.tri_trim[tetra]], top))
                    sa = np.cos(np.sum(sol_angs) / 4)
                    sa_dict[point_ind] = sa
                else:
                    sa_dict[point_ind] = None
        return sa_dict
    
    def plot_concave_hull(self, indices = None):
        """
        Plot the whole or partial concave hull of the points with indices. 
        """
        fig = plt.figure()        
        ax = fig.add_subplot(1, 1, 1, projection = '3d')
        ax.plot_trisurf(self.points[:,0], self.points[:,1], self.points[:,2],
                        triangles = self.Triangles, cmap = plt.cm.Blues, edgecolor = 'b')
        plt.show()

