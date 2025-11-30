import numpy as np

def geodesic_distances(a,b,t,indices):
    
    #Compute pairwise distance between points in geodesic
    N = len(indices)
    M = np.zeros((N, N))
    for ii in range(N):

        Pi = a[indices[ii][0]]
        Qi = b[indices[ii][1]]
        Pi = np.array(Pi)
        Qi = np.array(Qi)
        for jj in range(ii+1, N):
            Pj = a[indices[jj][0]]
            Qj = b[indices[jj][1]]
            
            Pj = np.array(Pj)
            Qj = np.array(Qj)
            da = np.linalg.norm(Pi - Pj)
            db = np.linalg.norm(Qi - Qj)
            M[ii][jj] = (1-t)* da + t*db
        print(f"compute geodesic distance{ii}/{N}")
    return M + M.T 


def convex_point(a,b,t,indices):
    N = len(indices)
    v = []
    for ii in range(N):

        Pi = a[indices[ii][0]]
        Qi = b[indices[ii][1]]
        Pi = np.array(Pi)
        Qi = np.array(Qi)
        v.append( Pi*(1-t) + Qi*t)
    return v

def geodesic_distances_gw(a,b,t,indices):
    #Compute pairwise distance between points in geodesic
    N = len(indices)
    M = np.zeros((N, N))
    for ii in range(N):

        Pi = a[indices[ii][0]]
        Qi = b[indices[ii][1]]
        Pi = np.array(Pi)
        Qi = np.array(Qi)
        for jj in range(ii+1, N):
            Pj = a[indices[jj][0]]
            Qj = b[indices[jj][1]]
            
            Pj = np.array(Pj)
            Qj = np.array(Qj)
            da = np.linalg.norm(Pi - Pj)
            db = np.linalg.norm(Qi - Qj)
            M[ii][jj] = t * da + (1-t)*db
    return M + M.T 

#测地线上w距离
def geodesic_distances_coot(a,b,t,point_indices, cycle_indices):
    N = len(point_indices)
    M = np.zeros(a.shape)
    for ii in range(N):

        Pi = a[point_indices[ii][0]]
        Qi = b[point_indices[ii][1]]
        Pi = np.array(Pi)
        Qi = np.array(Qi)
        for jj in range(len(cycle_indices)-1):
            if cycle_indices[jj][1] <= len(cycle_indices)-2:
                M[ii][jj] = t * Pi[cycle_indices[jj][0]] + (1-t) * Qi[cycle_indices[jj][1]]
            else:
                M[ii][jj] = t * Pi[cycle_indices[jj][0]]
    return M

def geodesic_distances_iota(a,b,t,indices):
    #Compute pairwise distance between points in geodesic
    N = len(indices)-1
    M = np.zeros((N, 2))
    for ii in range(N):
        if indices[ii][1] == N:
            Pi = a[indices[ii][0]]
            Qi = [(Pi[0]+Pi[1])/2, (Pi[0]+Pi[1])/2]
        else:
            Pi = a[indices[ii][0]]
            Qi = b[indices[ii][1]]
        #Pi = np.array(Pi)
        #Qi = np.array(Qi)
        M[ii][0] = t*Pi[0] + (1-t)*Qi[0]
        M[ii][1] = t*Pi[1] + (1-t)*Qi[1]
    return M