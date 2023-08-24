import numpy as np

def Sk_params(J,L,Ksym):
    hbar=197.3269804
    m=939.56542052 
    h2over2m=(hbar*hbar)/(2*m)
    two_thirds=2/3
    five_thirds=5/3
    five_ninths=5/9

    n0=0.1562
    E0=-15.926198132940234
    K0=239.6642289121857
    t1=301.8208
    t2=-273.2827
    x1=-0.3622
    x2=-0.4105
    a3=1/3
    a4=1
    
    n23 = n0**two_thirds
    n53 = n0**five_thirds
    A3 = n0**a3
    A4 = n0**a4
    
    ck = 0.6*((1.5*np.pi*np.pi)**two_thirds)
    CKE = h2over2m*ck
    DKE = five_ninths*CKE
    C12 = ck*0.125*0.5*(3.0*t1 + 4.0*t2*x2 + 5.0*t2)
    D12 = five_ninths*ck*0.125*(-3.0*t1*x1 + 5.0*t2*x2 + 4.0*t2)
    
    E0p = (E0 - CKE*n23 - C12*n53)/n0
    Jp  = (J  - DKE*n23 - D12*n53)/n0
    L0p = (  - 2.0*CKE*n23 - 5.0*C12*n53)/n0
    Lp  = (L - 2.0*DKE*n23 - 5.0*D12*n53)/n0
    K0p   = (K0   + 2.0*CKE*n23 - 10.0*C12*n53)/n0
    Ksymp = (Ksym + 2.0*DKE*n23 - 10.0*D12*n53)/n0
    
    C0 = 9.0*E0p*(a3+1.0)*(a4+1.0) - 3.0*L0p*(a3+a4+1.0) + K0p
    C0 = C0/(9.0*a3*a4)
    C3 = 9.0*E0p*(a4+1.0) - 3.0*L0p*(a4+1.0) + K0p
    C3 = C3/(9.0*A3*(a3*a3-a3*a4))
    C4 = 9.0*E0p*(a3+1.0) - 3.0*L0p*(a3+1.0) + K0p
    C4 = -C4/(9.0*A4*(a3*a4-a4*a4))
    D0 = 9.0*Jp*(a3+1.0)*(a4+1.0) - 3.0*Lp*(a3+a4+1.0) + Ksymp
    D0 = D0/(9.0*a3*a4)
    D3 = 9.0*Jp*(a4+1.0) - 3.0*Lp*(a4+1.0) + Ksymp
    D3 = D3/(9.0*A3*(a3*a3-a3*a4))  
    D4 = 9.0*Jp*(a3+1.0) - 3.0*Lp*(a3+1.0) + Ksymp
    D4 = -D4/(9.0*A4*(a3*a4-a4*a4))
    
    t0 = (8.0/3.0)*C0
    t3 = 16.0*C3
    t4 = 16.0*C4
    x0 = -0.5*((3.0*D0/C0) + 1.0)
    x3 = -0.5*((3.0*D3/C3) + 1.0)
    x4 = -0.5*((3.0*D4/C4) + 1.0)
    
    # print(t0,t1,t2,t3,t4)
    # print(x0,x1,x2,x3,x4)
    # #results from Will's code for my default injected parameters
    return t0,t1,t2,t3,t4,x0,x1,x2,x3,x4,a3,a4

#Note that Qsym is not independant in this version of the skyrme model
def Qsym(nsat,t0,t1,t2,t3,t4,x0,x1,x2,x3,x4,a3,a4):
    n=nsat  #definately could get nsat from the other parameters, but this will do for now
    hbar=197.3269804
    m=939.56542052 
    h2over2m=(hbar*hbar)/(2*m)
    kremains = 3/5 * (3*np.pi*np.pi)**(2/3)
    p0 = h2over2m * kremains * (1-2**(-2/3)) * n**(-7/3) * 8/27
    p1 = 0
    p2 = -t3/48 * n**(a3-2) * (2*x3+1) * (a3+1) * a3 * (a3-1)
    p3 = -t4/48 * n**(a4-2) * (2*x4+1) * (a4+1) * a4 * (a4-1)
    p4 = 1/8 * kremains * (1-2**(-2/3)) * (t1*(2+x1)+t2*(2+x2)) * n**(-4/3) * -10/27
    p5 = -1/8 * kremains * (1-2**(-5/3)) * (t1*(2*x1+1)-t2*(2*x2+1)) * n**(-4/3) * -10/27
    return 27*(nsat**3)*(p0+p1+p2+p3+p4+p5)

#Note that Qsym is not independant in this version of the skyrme model
def Zsym(nsat,t0,t1,t2,t3,t4,x0,x1,x2,x3,x4,a3,a4):
    n=nsat  #definately could get nsat from the other parameters, but this will do for now
    hbar=197.3269804
    m=939.56542052 
    h2over2m=(hbar*hbar)/(2*m)
    kremains = 3/5 * (3*np.pi*np.pi)**(2/3)
    p0 = h2over2m * kremains * (1-2**(-2/3)) * n**(-10/3) * -56/81
    p1 = 0
    p2 = -t3/48 * n**(a3-3) * (2*x3+1) * (a3+1) * a3 * (a3-1) * (a3-2)
    p3 = -t4/48 * n**(a4-3) * (2*x4+1) * (a4+1) * a4 * (a4-1) * (a4-2)
    p4 = 1/8 * kremains * (1-2**(-2/3)) * (t1*(2+x1)+t2*(2+x2)) * n**(-7/3) * 40/81
    p5 = -1/8 * kremains * (1-2**(-5/3)) * (t1*(2*x1+1)-t2*(2*x2+1)) * n**(-7/3) * 40/81
    return 81*(nsat**4)*(p0+p1+p2+p3+p4+p5)


t0,t1,t2,t3,t4,x0,x1,x2,x3,x4,a3,a4=Sk_params(34,50,-100) #J,L,Ksym
print("Qsym:   ",Qsym(0.15625851,t0,t1,t2,t3,t4,x0,x1,x2,x3,x4,a3,a4))
print("Zsym:   ",Zsym(0.15625851,t0,t1,t2,t3,t4,x0,x1,x2,x3,x4,a3,a4))


# E0:      -15.926198132940234
# nast:      0.1562
# K0:      239.6642289121857
# Q0:      -362.46919971374865
# Z0:      1465.6538274573425