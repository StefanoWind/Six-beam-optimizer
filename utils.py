# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt

def cosd(x):
    return np.cos(x/180*np.pi)

def sind(x):
    return np.sin(x/180*np.pi)
    
def tand(x):
    return np.tan(x/180*np.pi)

def arctand(x):
    return np.arctan(x)*180/np.pi

def vec2str(vec,separator=' ',format='%f'):
    s=''
    for v in vec:
        s=s+format % v+separator
    return s[:-len(separator)]

def axis_equal():
    from mpl_toolkits.mplot3d import Axes3D
    ax=plt.gca()
    is_3d = isinstance(ax, Axes3D)
    if is_3d:
        xlim=ax.get_xlim()
        ylim=ax.get_ylim()
        zlim=ax.get_zlim()
        ax.set_box_aspect((np.diff(xlim)[0],np.diff(ylim)[0],np.diff(zlim)[0]))
    else:
        xlim=ax.get_xlim()
        ylim=ax.get_ylim()
        ax.set_box_aspect(np.diff(ylim)/np.diff(xlim))
        
def draw_cube(ax,vertices):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    # Define the faces of the cube
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[4], vertices[7], vertices[3]],
        [vertices[1], vertices[5], vertices[6], vertices[2]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]]
    ]

    # Create a Poly3DCollection and add it to the plot
    cube = Poly3DCollection(faces, facecolors='k', edgecolors=None, alpha=0.1)
    ax.add_collection3d(cube)
    