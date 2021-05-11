"""SDF to Occupancy Grid"""
import numpy as np


def occupancy_grid(sdf_function, resolution):
    """
    Create an occupancy grid at the specified resolution given the implicit representation.
    :param sdf_function: A function that takes in a point (x, y, z) and returns the sdf at the given point.
    Points may be provides as vectors, i.e. x, y, z can be scalars or 1D numpy arrays, such that (x[0], y[0], z[0])
    is the first point, (x[1], y[1], z[1]) is the second point, and so on
    :param resolution: Resolution of the occupancy grid
    :return: An occupancy grid of specified resolution (i.e. an array of dim (resolution, resolution, resolution) with value 0 outside the shape and 1 inside.
    """ 

    # ###############
    '''
    coordinates = np.linspace(-0.5,0.5,num=resolution+1)
    xx,y_,z_ = np.meshgrid(coordinates,coordinates,coordinates,indexing='ij')
    occupancy_grid = np.zeros((resolution,resolution,resolution))
    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                # The middle of the vortex
                x_ = (coordinates[x] + coordinates[x+1])/2.0
                y_ = (coordinates[y] + coordinates[y+1])/2.0
                z_ = (coordinates[z] + coordinates[z+1])/2.0
                
                sdf_value = sdf_function(x_,y_,z_)
                occupancy_value = 0
                
                if sdf_value <= 0:
                    occupancy_value = 1
                occupancy_grid[x,y,z] = occupancy_value
    
    return occupancy_grid
    '''
    coordinates = np.linspace(-0.5,0.5,num=resolution)
    xx,yy,zz = np.meshgrid(coordinates,coordinates,coordinates,indexing='ij')
    occupancy_grid = np.zeros((resolution,resolution,resolution))
    
    sdf_values = sdf_function(xx.flatten(),yy.flatten(),zz.flatten())
    occupancy_grid = sdf_values.reshape(resolution,resolution,resolution)
    
    occupancy_grid[occupancy_grid == 0] = -1
    occupancy_grid[occupancy_grid > 0] = 0
    occupancy_grid[occupancy_grid < 0] = 1

    return occupancy_grid
    # ###############
