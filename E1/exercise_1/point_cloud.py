"""Triangle Meshes to Point Clouds"""
import numpy as np
from numpy.core.fromnumeric import size


def sample_point_cloud(vertices, faces, n_points):
    """
    Sample n_points uniformly from the mesh represented by vertices and faces
    :param vertices: Nx3 numpy array of mesh vertices
    :param faces: Mx3 numpy array of mesh faces
    :param n_points: number of points to be sampled
    :return: sampled points, a numpy array of shape (n_points, 3)
    """

    # ###############
    # Calculate the areas to find the probability distribution
    areas = [] # Holds the triangle areas
    for face in faces:
        [corner_1,corner_2,corner_3] = vertices[face]
        area = 0.5*np.linalg.norm(np.cross(corner_2-corner_1,corner_3-corner_1))
        areas.append(area)
    areas = np.asarray(areas)
    probabilities = areas/areas.sum()
    weighted_random_triangle_indices = np.random.choice(range(len(areas)), size=n_points,p=probabilities)
    
    sampled_points = []
    for face_index in weighted_random_triangle_indices:
        [A,B,C] = vertices[faces[face_index]]
        [r1,r2] = np.random.rand(2)
        u = 1 - np.sqrt(r1)
        v = np.sqrt(r1)*(1-r2)
        w = np.sqrt(r1)*r2
        point = u*A + v*B + w*C
        sampled_points.append(point)
    sampled_points = np.asarray(sampled_points)
    return sampled_points
    # ###############
