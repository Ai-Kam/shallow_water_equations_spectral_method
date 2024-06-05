import math
import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt

M = 10
K=3*M+1 #theta
J=int(3*M/2)+20 #phi

a = 6400*1000 # m
g = 9.81 # m/s^2
omega = 7.2921e-5 # 1/s
H = 10*1000 # m
dt = 60 # s
max_steps = 1000

class coefficients():
    def __init__(self, N, M, dtype=np.complex128):
        self.N = N
        self.M = M
        self.data = np.zeros((N,M), dtype=dtype)

    def __getitem__(self, l, m):
        return self.data[l,m] if m >= 0 else self.data[self.M-l-1,m]
    
    def __setitem__(self, l, m, value):
        if m >= 0:
            self.data[l,m] = value
        else:
            self.data[self.M-l-1,m] = value

def diff(arr,axis):
    if axis == 0:
        arr = np.concatenate((arr[-1:],arr,arr[:1]), axis=0)
        return arr[1:]-arr[:-1]
    elif axis == 1:
        arr = np.concatenate((arr[:,-1:],arr,arr[:,:1]), axis=1)
        return arr[:,1:]-arr[:,:-1]

def iFFT(f_mp, M, theta_k):
    #print(f_mp.shape, theta_k.shape)
    return f_mp.T @ np.exp(1j*np.atleast_2d(np.arange(-M,M+1)).T*theta_k)

def FFT(f_jk, M, theta_k):
    #print(np.exp(-1j*np.atleast_2d(np.arange(-M,M+1)).T*theta_k).shape, f_jk.shape)
    return (np.exp(-1j*np.atleast_2d(np.arange(-M,M+1)).T*theta_k) @ f_jk.T / len(theta_k))


def F_dudt(theta_k, zero_points, v_pt):
    K = len(theta_k)
    J = len(zero_points)
    phi = np.arcsin(zero_points)
    f = 2*omega*np.atleast_2d(zero_points).T
    return f*v_pt

def F_dvdt(theta_k, zero_points, u_pt):
    K = len(theta_k)
    J = len(zero_points)
    phi = np.arcsin(zero_points)
    f = 2*omega*np.atleast_2d(zero_points).T
    return f*u_pt

def associated_legendre(gaussian_weights, f_mp, P_nm):
    #print(gaussian_weights.shape, f_mp.shape, P_nm.shape)
    #print(np.atleast_3d(np.atleast_2d(gaussian_weights) * f_mp).shape, P_nm.shape)
    return np.sum(np.atleast_3d(np.atleast_2d(gaussian_weights) * f_mp).transpose((2,0,1)) * P_nm / 2, axis=2).T

def associated_legendre_inverse(f_mn, P_nm):
    #print(f_mn.shape, P_nm.shape)
    tmp = (np.atleast_3d(f_mn) * P_nm.transpose((1,0,2)))
    #print(tmp.shape)
    ret = np.zeros((P_nm.shape[1], P_nm.shape[2]), dtype=np.complex128)
    for m in range(-M,M+1):
        #print(tmp[m+M,np.abs(m):,:].shape)
        ret[m+M] = np.sum(tmp[m+M,np.abs(m):,:], axis=0)
    return ret

def calc_legendre_zero_point(J):
    mat = np.zeros((J,J))
    for j in range(J-1):
        mat[j,j+1] = (j+1)/np.sqrt((2*j+1)*(2*j+3))
        mat[j+1,j] = (j+1)/np.sqrt((2*j+1)*(2*j+3))
    eigvals, eigvectors = np.linalg.eig(mat)
    gaussian_weights = 2*eigvectors[0]**2
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    gaussian_weights = gaussian_weights[order]
    return eigvals, gaussian_weights

def legendre(zero_points, M, J):
    memo = np.zeros((M+1,M+1,J), dtype=np.float64)
    memo[0,0] = 1
    if M >= 1:
        memo[1,0] = zero_points
        memo[1,1] = np.sqrt(1-zero_points**2)
    for n in range(2,M+1):
        memo[n,0] = ((2*n-1)*zero_points*memo[n-1,0] - (n-1)*memo[n-2,0])/n
        for m in range(1,n+1):
            memo[n,m] = (- (n-m+1)*zero_points*memo[n,m-1] + (n+m-1)*memo[n-1,m-1])/(np.sqrt(1-zero_points**2))
    reg_coef = np.zeros((M+1,M+1), dtype=np.float64)
    for n in range(M+1):
        for m in range(n+1):
            reg_coef[n,m] = np.sqrt((2*n+1)/2*math.factorial(n-m)/math.factorial(n+m))
    #print(reg_coef)
    memo *= reg_coef[:,:,np.newaxis]
    #print(memo)
    return np.concatenate((memo[:,-2::-1,:], memo), axis=1)



u_lm = coefficients(M,M)
v_lm = coefficients(M,M)
h_lm = coefficients(M,M)

def spec_test():
    zero_points, gaussian_weights = calc_legendre_zero_point(J)
    P_nm = legendre(zero_points, M, J)
    v_tilda_mn = np.zeros((2*M+1,M+1), dtype=np.complex128)
    v_tilda_mn[4,1] = 1
    v_tilda_mp = associated_legendre_inverse(v_tilda_mn, P_nm)
    print(v_tilda_mp)
    cs=plt.contourf(v_tilda_mp.real)
    plt.colorbar(cs)
    plt.show()   
    v_tilda_mn = associated_legendre(gaussian_weights, v_tilda_mp, P_nm)
    cs=plt.contourf(v_tilda_mn.real)
    plt.colorbar(cs)
    plt.show()

def FFT_test():
    u_pt = np.zeros((J,K), dtype=np.float64)
    #u_pt[9:12] = 1
    u_pt[3:16,3:16] = np.ones((13,13))*10
    print(np.sum(P_nm**2, axis=2))
    theta_k = np.arange(1,K+1) * 2*np.pi / K
    zero_points, gaussian_weights = calc_legendre_zero_point(J)
    pt = np.meshgrid( theta_k, np.arcsin(zero_points) )
    cs=plt.contourf(pt[0], pt[1], u_pt)
    plt.colorbar(cs)
    plt.show()
    for i in range(100):
        v_tilda_mp = FFT(F_dvdt(theta_k, zero_points, u_pt), M, theta_k)
        v_tilda_mn = associated_legendre(gaussian_weights, v_tilda_mp, P_nm)
        v_tilda_mp = associated_legendre_inverse(v_tilda_mn, P_nm)
        u_pt = iFFT(v_tilda_mp, M, theta_k)
    cs=plt.contourf(pt[0], pt[1], u_pt)
    plt.colorbar(cs)
    plt.show()

def P_nm_test():
    zero_points, gaussian_weights = calc_legendre_zero_point(J)
    P_nm = legendre(zero_points, M, J)
    for n in range(M+1):
        for m in range(n+1):
            plt.plot(zero_points, P_nm[n,m+M])
    plt.show()

def main():
    theta_k = np.arange(1,K+1) * 2*np.pi / K
    zero_points, gaussian_weights = calc_legendre_zero_point(J)
    pt = np.meshgrid( theta_k, np.arcsin(zero_points) )
    P_nm = legendre(zero_points, M, J)
    ns = np.arange(1,M+1)

    u_pt = np.zeros((J,K), dtype=np.float64)
    v_pt = np.zeros((J,K), dtype=np.float64)
    u_pt[J//2,:K//2] = 10
    #u_pt[3:6,3:6] = np.array([[0.1,0.1,0.1],[0.1,0.3,0.1],[0.1,0.1,0.1]])*100
    
    u_mn = FFT(u_pt, M, theta_k)
    u_mn = associated_legendre(gaussian_weights, u_mn, P_nm)
    v_mn = FFT(v_pt, M, theta_k)
    v_mn = associated_legendre(gaussian_weights, v_mn, P_nm)

    for i in range(100000):
        u_tilda_mp = FFT(F_dudt(theta_k, zero_points, v_pt), M, theta_k)
        u_tilda_mn = associated_legendre(gaussian_weights, u_tilda_mp, P_nm)

        v_tilda_mp = FFT(F_dvdt(theta_k, zero_points, u_pt), M, theta_k)
        v_tilda_mn = associated_legendre(gaussian_weights, v_tilda_mp, P_nm)

        for n in range(1,M+1):
            u_mn[:,n] += u_tilda_mn[:,n]/(n*(n+1))*dt
            v_mn[:,n] += v_tilda_mn[:,n]/(n*(n+1))*dt

        u_tilda_mp = associated_legendre_inverse(u_mn, P_nm)
        u_pt = iFFT(u_tilda_mp, M, theta_k)

        v_tilda_mp = associated_legendre_inverse(v_mn, P_nm)
        v_pt = iFFT(v_tilda_mp, M, theta_k)

        if i % 10000 == 0:
            cs = plt.contourf(pt[0], pt[1], v_pt.real)
            cs2 = plt.contour(pt[0], pt[1], u_pt.real, colors='black')
            plt.clabel(cs2)
            plt.colorbar(cs)
            plt.show()
    


if __name__ == '__main__':
    print(J,K)


    #P_nm_test()
    #spec_test()
    #FFT_test()
    main()
    exit()

    cs=plt.contourf(pt[0], pt[1], u_pt)
    plt.colorbar(cs)
    plt.show()

    cs=plt.contourf(pt[0], pt[1], dvdt_pt)
    plt.colorbar(cs)
    plt.show()

    dvdt_pt = np.zeros((J,K), dtype=np.complex128)
    for n in range(1,M+1):
        for m in range(-n,n+1):
            dvdt_pt += v_tilda_mn[m+M,n]/(n*(n+1))*sph_harm(m, n, pt[0], pt[1] )
    
    cs = plt.contourf(pt[0], pt[1], dvdt_pt.real)
    #plt.contour(pt[0], pt[1], dvdt_pt.imag)
    plt.colorbar(cs)
    plt.show()
    """
    for step in range(max_steps):
    dt_u_lm = coefficients(M)
    dt_v_lm = coefficients(M)
    dt_h_lm = coefficients(M)
    for l in range(M):
        for m in range(-l,l+1):
            dt_u_lm[l,m] = -f*u_lm[l,m] - g/a*1j*m*h_lm[l,m]
            dt_v_lm[l,m] = -f*v_lm[l,m] + g/a*1j*m*h_lm[l,m]"""