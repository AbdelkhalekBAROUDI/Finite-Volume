from utilities.triangle import orthocenter
from scipy              import sparse

import numpy as np

def Assembly(cellid:'int[:,:]', centerf:'float[:]', 
                mesuref:'float[:]', namef:'int[:]', innerfaces:'int[:]', 
                boundaryfaces:'int[:]'):
    
    sizeM  = 4*len(innerfaces)+len(boundaryfaces)
    center = np.zeros( (nbcells, 2), dtype = np.float64 )
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
