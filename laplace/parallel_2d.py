from manapy.ddm import readmesh
from manapy.ddm import Domain

import mumps

import numpy as np

import os

from triangle import dist, orthocenter

from manapy.comms                import all_to_all
from manapy.comms.pyccel_comm    import define_halosend
from manapy.ast.pyccel_functions import convert_solution

from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# ... get the mesh directory
'''try:
    MESH_DIR = os.environ['MESH_DIR']
 
except:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    BASE_DIR = os.path.join(BASE_DIR , '..')
    MESH_DIR = os.path.join(BASE_DIR, 'mesh')
'''
filename = "../mesh/rectangle.msh"
#filename = os.path.join(MESH_DIR, filename)

        
dim = 2
readmesh(filename, dim=dim, periodic=[0,0,0])
    
#Create the informations about cells, faces and nodes
domain = Domain(dim=dim)
    
cells   = domain.cells
nbcells = domain.nbcells
faces   = domain.faces
nodes   = domain.nodes
halos   = domain.halos


def Assembly(cellid:'int[:,:]', centerf:'float[:]', haloext:'int[:,:]',halofid:'int[:]',
                loctoglob:'int[:]', mesuref:'float[:]', namef:'int[:]', innerfaces:'int[:]', 
                boundaryfaces:'int[:]', halofaces:'int[:]', globalsize):
    
    sizeM  = 4*len(innerfaces) + len(boundaryfaces) + 2 * len(halofaces)

    center = np.zeros( (nbcells, 2), dtype = np.float64 )
    b      = np.zeros(  globalsize, dtype = np.float64 )
    row    = np.zeros(    sizeM, dtype = np.int32   )
    col    = np.zeros(    sizeM, dtype = np.int32   )
    data   = np.zeros(    sizeM, dtype = np.float64   )
    
    for i in range(nbcells):
        center[i, :] = orthocenter([ nodes.vertex[ cells.nodeid[i][k] ] for k in range(3) ])
    
    
    #orthocenter = np.zeros(domain.nbcells)
    #center_ghost = np.zeros(domain.nbfaces)
    centerx = np.zeros(nbcells, dtype = np.float64)
    centery = np.zeros(nbcells, dtype = np.float64)
    
    centerx[:] = center[:, 0]
    centery[:] = center[:, 1]

    #array to send
    halotosendx  = np.zeros(len(domain.halos.halosint))
    halotosendy  = np.zeros(len(domain.halos.halosint))
    
    #array to receive
    center_halox  = np.zeros(domain.nbhalos)
    center_haloy  = np.zeros(domain.nbhalos)

 
    #prepare halotosend array
    define_halosend(centerx, halotosendx, domain.halos.indsend)
    define_halosend(centery, halotosendy, domain.halos.indsend)
    
    
    #send information to neighbor processors received on orthocenter_halo array
    all_to_all(halotosendx, domain.nbhalos, domain.halos.scount, domain.halos.rcount,
               center_halox, domain.halos.comm_ptr)
    all_to_all(halotosendy, domain.nbhalos, domain.halos.scount, domain.halos.rcount,
               center_haloy, domain.halos.comm_ptr)
    
    cmpt = 0
    for face in innerfaces:
        
        c_left = cellid[face][0]
        c_leftglob  = loctoglob[c_left]
        
        c_right = cellid[face][1]
        c_rightglob  = loctoglob[c_right]

        mesure = mesuref[face]
        
        row[cmpt] = c_leftglob
        col[cmpt] = c_leftglob
        data[cmpt] = - (mesure/ dist(center[c_left], center[c_right]))
        cmpt = cmpt + 1
        
        row[cmpt] = c_rightglob
        col[cmpt] = c_rightglob
        data[cmpt] = -(mesure/dist(center[c_left], center[c_right])) 
        cmpt = cmpt + 1
        
        row[cmpt] = c_leftglob
        col[cmpt] = c_rightglob
        data[cmpt] = (mesure/dist(center[c_left], center[c_right])) 
        cmpt = cmpt + 1
        
        row[cmpt] = c_rightglob
        col[cmpt] = c_leftglob
        data[cmpt] = (mesure/dist(center[c_left], center[c_right])) 
        cmpt = cmpt + 1
        
    for face in boundaryfaces: 
        c_left = cellid[face][0]
        c_leftglob  = loctoglob[c_left]
        
        #K = loctoglob[cellid[face][0]]
        mesure = mesuref[face]
        
        row[cmpt] = c_leftglob
        col[cmpt] = c_leftglob
        data[cmpt] = -(mesure/dist(center[c_left], centerf[face])) 
        cmpt = cmpt + 1
        
        if namef[face] == 1:
            b[c_leftglob] = -20*(mesure/dist(center[c_left], centerf[face])) 
    
    for face in halofaces:
        mesure = mesuref[face]
        
        c_left = cellid[face][0]
        c_leftglob  = loctoglob[c_left]
        
        #parameters[0] = param4[i]; parameters[1] = param2[i]
        
        c_rightglob = haloext[halofid[face]][0]
        c_right     = halofid[face]
        
        row[cmpt] = c_leftglob
        col[cmpt] = c_leftglob
        
        p =  - mesure / dist(center[c_left], [center_halox[c_right], center_haloy[c_right]] )
        data[cmpt] = p
        cmpt = cmpt + 1

        row[cmpt] = c_leftglob
        col[cmpt] = c_rightglob
        data[cmpt] = - p
        cmpt = cmpt + 1
    
    return row, col, data, b

 
globalsize = COMM.allreduce(nbcells, op=MPI.SUM)
  
row, col, data, b = Assembly(faces.cellid, faces.center, halos.halosext, faces.halofid, cells.loctoglob,
                                 faces.mesure, faces.name, domain.innerfaces, domain.boundaryfaces, 
                                 domain.halofaces, globalsize)

rhs0_glob = COMM.reduce(b, op=MPI.SUM, root=0)

#if RANK == 0:
#    print(rhs0_glob)

ctx = mumps.DMumpsContext(comm=COMM)
ctx.set_shape(nbcells)
ctx.set_silent()

ctx.set_distributed_assembled_rows_cols(row+1, col+1)
ctx.set_distributed_assembled_values(data)
ctx.set_icntl(18,3)
    
if COMM.Get_rank() == 0:
    ctx.id.n = globalsize
    
       
#Analyse 
ctx.run(job=1)
#Factorization Phase
ctx.run(job=2)

#u3 = np.zeros(nbcells)
#Allocation size of rhs
if RANK == 0:
    u = rhs0_glob.copy()
    ctx.set_rhs(u)
            
#Solution Phase
ctx.run(job=3)

#Destroy
ctx.destroy()
#if RANK == 0:
#    print(u3)
#for i in range(len(row)):
#    print(data[i], row[i], col[i])

sendcounts1 = np.array(COMM.gather(nbcells, root=0))
x1converted = np.zeros(globalsize)
u_loc = np.zeros(nbcells)

if RANK == 0:
    #Convert solution for scattering
    convert_solution(u, x1converted, domain.cells.tc, globalsize)
COMM.Scatterv([x1converted, sendcounts1, MPI.DOUBLE], u_loc, root = 0)

domain.save_on_cell(miter=2, value = u_loc )
