from manapy.ddm import readmesh
from manapy.ddm import Domain

import numpy as np

from scipy import sparse
from scipy.sparse.linalg import spsolve

import os

from triangle import dist, orthocenter


# ... get the mesh directory
'''try:
    MESH_DIR = os.environ['MESH_DIR']
 
except:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    BASE_DIR = os.path.join(BASE_DIR , '..')
    MESH_DIR = os.path.join(BASE_DIR, 'mesh')
'''
filename = "carre.msh"

        
dim = 2
readmesh(filename, dim=dim, periodic=[0,0,0])
    
#Create the informations about cells, faces and nodes
domain = Domain(dim=dim)
    
cells   = domain.cells
nbcells = domain.nbcells
faces   = domain.faces
nodes   = domain.nodes


def Assembly(cellid:'int[:,:]', centerf:'float[:]', 
                mesuref:'float[:]', namef:'int[:]', innerfaces:'int[:]', 
                boundaryfaces:'int[:]'):
    
    sizeM  = 4*len(innerfaces)+len(boundaryfaces)
    center = np.zeros( (nbcells, 2), dtype = np.float64 )
    b      = np.zeros( nbcells, dtype = np.float64 )  
    row    = np.zeros(    sizeM, dtype = np.int32   )
    col    = np.zeros(    sizeM, dtype = np.int32   )
    data   = np.zeros(    sizeM, dtype = np.float64   )
    
    for i in range(nbcells):
        center[i, :] = orthocenter([ nodes.vertex[ cells.nodeid[i][k] ] for k in range(3) ])
        
    cmpt = 0
    for face in innerfaces:
        K = cellid[face][0] 
        L = cellid[face][1]
        mesure = mesuref[face]
        
        row[cmpt] = K
        col[cmpt] = K
        data[cmpt] = - (mesure/ dist(center[K], center[L]))
        cmpt = cmpt + 1
        
        row[cmpt] = L
        col[cmpt] = L
        data[cmpt] = -(mesure/dist(center[K], center[L])) 
        cmpt = cmpt + 1
        
        row[cmpt] = K
        col[cmpt] = L
        data[cmpt] = (mesure/dist(center[K], center[L])) 
        cmpt = cmpt + 1
        
        row[cmpt] = L
        col[cmpt] = K
        data[cmpt] = (mesure/dist(center[K], center[L])) 
        cmpt = cmpt + 1
        
    for face in boundaryfaces: 
        K = cellid[face][0] 
        mesure = mesuref[face]
        
        row[cmpt] = K
        col[cmpt] = K
        data[cmpt] = -(mesure/dist(center[K], centerf[face])) 
        cmpt = cmpt + 1
        
        if namef[face] == 1:
            b[K] = -20*(mesure/dist(center[K], centerf[face])) 
            
    mat = sparse.csr_matrix((data, (row, col)))
    
    return mat, row, col, data, b

 
M, row, col, data, b = Assembly(faces.cellid, faces.center, faces.mesure, 
                                    faces.name, domain.innerfaces, domain.boundaryfaces)

u = spsolve(M, b)
domain.save_on_cell(miter=0, value = u)
