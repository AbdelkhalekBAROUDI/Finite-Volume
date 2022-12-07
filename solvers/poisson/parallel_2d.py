from utilities.triangle          import dist, orthocenter
from manapy.comms                import all_to_all
from manapy.comms.pyccel_comm    import define_halosend
from manapy.ast.pyccel_functions import convert_solution
from mpi4py                      import MPI


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


def Assembly_M3(cellid:'int[:,:]', centerf:'float[:]', haloext:'int[:,:]',halofid:'int[:]',
                loctoglob:'int[:]', mesuref:'float[:]', namef:'int[:]', innerfaces:'int[:]', 
                boundaryfaces:'int[:]', halofaces:'int[:]'):
    
    sizeM  = 4*len(innerfaces) + len(boundaryfaces) + 2 * len(halofaces)
    globalsize = COMM.allreduce(nbcells, op=MPI.SUM)

    center = np.zeros( (nbcells, 2), dtype = np.float64 )
    b      = np.zeros(  globalsize, dtype = np.float64 )
    row    = np.zeros(    sizeM, dtype = np.int32   )
    col    = np.zeros(    sizeM, dtype = np.int32   )
    data   = np.zeros(    sizeM, dtype = np.float64   )
    
    for i in range(nbcells):
        center[i, :] = orthocenter([ nodes.vertex[ cells.nodeid[i][k] ] for k in range(3) ])
    
    
    
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
