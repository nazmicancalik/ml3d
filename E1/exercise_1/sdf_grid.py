"""Creating an SDF grid"""
import numpy as np


def sdf_grid(sdf_function, resolution):
    """
    Create an occupancy grid at the specified resolution given the implicit representation.
    :param sdf_function: A function that takes in a point (x, y, z) and returns the sdf at the given point.
    Points may be provides as vectors, i.e. x, y, z can be scalars or 1D numpy arrays, such that (x[0], y[0], z[0])
    is the first point, (x[1], y[1], z[1]) is the second point, and so on
    :param resolution: Resolution of the occupancy grid
    :return: An occupancy grid of specified resolution (i.e. an array of dim (resolution, resolution, resolution) with value 0 outside the shape and 1 inside.
    """

    # ###############
    coordinates = np.linspace(-0.5,0.5,num=resolution+1)
    
    sdf_grid = np.zeros((resolution,resolution,resolution))
    
    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                # The middle of the vortex
                x_ = (coordinates[x] + coordinates[x+1])/2.0
                y_ = (coordinates[y] + coordinates[y+1])/2.0
                z_ = (coordinates[z] + coordinates[z+1])/2.0
                sdf_value = sdf_function(x_,y_,z_)
                sdf_grid[x,y,z] = sdf_value
    
    return sdf_grid
    # ###############
