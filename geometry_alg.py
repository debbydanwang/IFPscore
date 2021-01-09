#%reset -f

import numpy as np
from collections import defaultdict
from scipy.spatial import ConvexHull


class Graph: 
    def __init__(self, V): 
        """
        Initialize a graph by the number of vertices (V).
        """       
        self.V = V 
        self.adj = [[] for i in range(V)] 
  
    def DFS(self, temp, v, visited): 
        """
        Depth-first search.
        """  
        # Mark the current vertex as visited 
        visited[v] = True  
        # Store the vertex to list 
        temp.append(v)   
        # Repeat for all vertices adjacent to this vertex v 
        for i in self.adj[v]: 
            if visited[i] == False:                   
                # Update the list 
                temp = self.DFS(temp, i, visited) 
        return temp 
  
    def addEdge(self, v, w): 
        """
        Add an undirected edge between vertices v and w (indices)
        """
        self.adj[v].append(w) 
        self.adj[w].append(v) 
  
    def NumOfConnectedComponents(self): 
        """
        Compute connected components in an undirected graph and return the number of connected components.
        """
        visited = [] 
        cc = [] 
        for i in range(self.V): 
            visited.append(False) 
        for v in range(self.V): 
            if visited[v] == False: 
                temp = [] 
                cc.append(self.DFS(temp, v, visited)) 
        return len(cc) 
  
def get_lifted_points(coor_wp):
    """Form a set of lifted points from a weighted point set.
    Parameters:
        coor_wp - coordinates of weighted points in 3D, format: (x, y, z, weight)
    Returns the lifted point set with format: (x, y, z, x^2 + y^2 + z^2 - weight).
    """
    lifted = coor_wp
    lifted[:, 3] = coor_wp[:, 0] ** 2 + coor_wp[:, 1] ** 2 + coor_wp[:, 2] ** 2 - coor_wp[:, 3]
    return lifted

class RT(object):
    def __init__(self, points, weights):
        """
        Initialize and compute the regular triangulation (RT) of a set of 3D weighted points (S) using the higher-dimension embedding.
        RT(S) is obtained as a vertical projection of the lower facets of the convex hull of the lifted points (S+), namely
        RT(S) = proj(CH(S+)_lowerfacets).
        See Qhull as an example in [1] C. Bradford Barber, David P. Dobkin, and Hannu Huhdanpaa. The quickhull algorithm 
        for convex hulls. ACM Trans. Math. Softw., 22(4):469–483, 1996.
        Parameters:
            weighted_points - a set of weighted points, format as (x, y, z, weight)
        """
        self.points = points
        self.pweights = weights
        weighted_points = np.column_stack((points, weights))
        # Building convex hull of the lifted points 
        lifted_points = get_lifted_points(weighted_points)
        hull = ConvexHull(points = lifted_points)
        # For each facet on the convex hull, its hyperplane is defined as A.x + B.y + C.z + D.weight + E = 0, namely weight = (A.x + B.y + C.z + E) / (-D),
        # compute the new weights for all vertices on the convex hull based on each facet hyperplane equation: new_weight_ij = (Ai.xj + Bi.yj + Ci.zj + Ei) / (-Di),
        # if weight_ij >= new_weight_ij for all vertices (j) and Di != 0, then facet i is a lower facet (the convex hull is vertically above it and it is non-vertical),
        # collecting all such facets i (tetrahedra) and vertically projecting them to 3D gives RT(S).
        pts = np.column_stack((np.take(hull.points, hull.vertices, axis = 0), 
                               np.ones(hull.vertices.shape[0])))
        pts_for_dotprod = pts[:, [0, 1, 2, 4]]
        weight = pts[:, 3]
        facet_num = hull.equations.shape[0]
        vertex_num = pts.shape[0]
        weight_comp = np.ones(shape = (facet_num, vertex_num)) * np.inf
        for facet in np.arange(facet_num):
            para = hull.equations[facet]
            if para[3] != 0: # the hyperplane is vertical if para[3] == 0
                for vet in np.arange(vertex_num):
                    weight_comp[facet, vet] = np.dot(np.delete(para, 3), pts_for_dotprod[vet]) / (-para[3])     
        lower_facets = []
        for facet in np.arange(facet_num):
            if np.all(weight >= weight_comp[facet]):
                lower_facets += [facet]
        self.simplices = np.take(hull.simplices, lower_facets, axis = 0)
        
def find_elements(tetra):
    """
    Find the elements of a (trimmed) triangulation, namely the vertices, edges and triangles.
    Parameters:
        tetra - np.array of shape (n,4), each row is a tetrahedron represented by a 4-tuple of vertex indices
    Return:
        vertices, edges and triangles in the triangulation
    """
    # Find triangles in a triangulation 
    TriComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    Triangles = tetra[:, TriComb].reshape(-1, 3)
    Triangles = np.sort(Triangles, axis = 1)
    # Remove overlapping triangles
    TrianglesDict = defaultdict(int)
    for tri in Triangles:
        TrianglesDict[tuple(tri)] += 1
    Triangles = np.array([tri for tri in TrianglesDict if TrianglesDict[tri] == 1])
    # Edges
    EdgeComb = np.array([(0, 1), (0, 2), (1, 2)])
    Edges = Triangles[:, EdgeComb].reshape(-1, 2)
    Edges = np.sort(Edges, axis = 1)
    Edges = np.unique(Edges, axis = 0)  
    # Vertices
    Vertices = np.unique(Edges)
    return Vertices, Edges, Triangles

def solid_angle_tetra(points, top):
    """
    Compute the solid angle subtended by a tetrahedron at one of its vertex.
    Parameters:
        points - np.array of shape (4,3), denoting the 4 vertices of a tetrahedron
        top - the index of vertex where the solid angle is computed
    Return:
        solid angle at the top vertex in the tetrahedron
    Example:
        points = np.array([[1,2,3], [1,3,5], [2,4,6], [1,2,1]])
        top = 0
        solid_angle_tetra(points, top) # output: 0.86
    """
    origin = np.expand_dims(points[top], axis = 0)
    triangle = np.delete(points, top, axis = 0)
    vectors = triangle - origin
    vectors_for_crosprod1 = np.take(vectors, [0, 0, 1, 1, 2, 2], axis = 0)
    vectors_for_crosprod2 = np.take(vectors, [1, 2, 0, 2, 0, 1], axis = 0)
    normals = np.cross(vectors_for_crosprod1, vectors_for_crosprod2)
    sol_ang = compute_angle_between(normals[0], normals[1]) + compute_angle_between(normals[2], normals[3]) + compute_angle_between(normals[4], normals[5]) - np.pi
    return sol_ang    
    
def compute_angle_between(v1, v2):
    """
    Compute the angle between two vectors v1 and v2 according to w = arccos(v1.v2/|v1||v2|)
    return w
    """
    dot_prod = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2) 
    return np.arccos(dot_prod / norms)

def find_plane_of_triangle(triangle_points):
    """
    Find the plane equation of a triangle ABC, according to the normal(p1, p2, p3) = CA x BA and offset p4 = -(p1.xA + p2.yA + p3.zA):
        p1.x + p2.y + p3.z + p4 = 0
    Parameters:
        triangle_points - coordinates of the three vertices of triangle, shape of 3 x 3
    returns an array of [normal, offset]
    """
    vec1 = triangle_points[2] - triangle_points[0]
    vec2 = triangle_points[1] - triangle_points[0]
    normal = np.cross(vec1, vec2)
    anchor = -np.dot(normal, triangle_points[0])
    return (normal, anchor)
