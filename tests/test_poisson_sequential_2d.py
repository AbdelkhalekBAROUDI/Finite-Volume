from scipy.sparse.linalg        import spsolve
from manapy.ddm                 import readmesh
from manapy.ddm                 import Domain
from solvers.poisson.sequential import Assembly

filename = "../mesh/untitled.msh"
        
dim = 2
readmesh(filename, dim=dim, periodic=[0,0,0])
    
#Create the informations about cells, faces and nodes
domain = Domain(dim=dim)
    
cells   = domain.cells
nbcells = domain.nbcells
faces   = domain.faces
nodes   = domain.nodes
halos   = domain.halos

A, row, col, data, b3 = Assembly(faces.cellid, faces.center, halos.halosext, faces.halofid, cells.loctoglob,
                                 faces.mesure, faces.name, domain.innerfaces, domain.boundaryfaces, 
                                 domain.halofaces)

u = spsolve(A, b)

domain.save_on_cell(miter=0, value = u)
