from numpy import sqrt, array, arccos, tan, pi

__all__ = ['dist', 'angles', 'orthocenter']

def dist(p1, p2):
    """
    Compute the euclidean distance between two points in 2d plane
    """
    return sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 ) 

def angles(T):
    """
    Compute the angles of a triangle using the Law of Cosines:
    
    T: (3, 2) array with coordinates of the 3 vetices.
    
    return : 1D array with the 3 angles
    """
    
    # Extract coordinates of the vertices
    A = T[0]
    B = T[1]
    C = T[2]
    
    # Compute length of the triangle's sides
    a = dist(B, C)
    b = dist(A, C)
    c = dist(A, B)
    
    # Compute the angles alpha, beta and gamma
    alpha = arccos( (b**2 + c**2 - a**2) / (2*b*c) ) #  angle(AB, AC)
    beta  = arccos( (a**2 + c**2 - b**2) / (2*a*c) ) #  angle(BA, BC)
    gamma = arccos( (a**2 + b**2 - c**2) / (2*a*b) ) #  angle(CA, CB)
    
    return array( [ alpha, beta, gamma ] )
    
def orthocenter(T):
    """
    Compute the coordinates of a triangle's orthocenter H(x, y) using
    trigonometric formulas 
    
    T: (3, 2) array with coordinates of the 3 vetices.
    
    return : 1D array with H coordinates 
    """
    
    angle = angles(T)
    
    # Extract coordinates of the vertices
    A = T[0]
    B = T[1]
    C = T[2]

    # Compute orthocenter coordinates
    x = ( A[0] * tan(angle[0]) + B[0] * tan(angle[1]) + C[0] * tan(angle[2]) ) \
        / ( tan(angle[0]) +  tan(angle[1]) + tan(angle[2]) )
    
    y = ( A[1] * tan(angle[0]) + B[1] * tan(angle[1]) + C[1] * tan(angle[2]) ) \
        / ( tan(angle[0]) +  tan(angle[1]) + tan(angle[2]) )
    
    return array( [x , y] ) 
