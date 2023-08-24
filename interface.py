import numpy as np
import generate_Will_EOS as EOS_polycore

class Eos_nseospy_format():
    def __init__(self, rho,pres,bary,smod,gam1):
        rhodrip = 4.3e+11 #hardcoded in Will's code
        neos=len(bary)
        rho=np.array(rho)
        pres=np.array(pres)
        bary=np.array(bary)
        
        #jerome's code can only take arrays that are <=1000 entries long, so resample the EOS to be that length
        if (neos>1000):
            for i in (range(neos)):
                #seperate out the outer crust, since interpolating it gets very dodgy
                if(rho[i]<rhodrip):
                    ocstart=i
                    break
            ocbary=bary[ocstart:neos-1]
            ocrho=rho[ocstart:neos-1]
            ocpres=pres[ocstart:neos-1]
            newbary=np.logspace(np.log10(np.min(bary[0:ocstart-1])),np.log10(np.max(bary)),1000-len(ocbary))
            newrho=np.interp(newbary,np.flip(bary),np.flip(rho))
            newpres=np.interp(newbary,np.flip(bary),np.flip(pres))
            bary=np.array(np.flip(newbary).tolist()+ocbary.tolist())
            rho=np.array(np.flip(newrho).tolist()+ocrho.tolist())
            pres=np.array(np.flip(newpres).tolist()+ocpres.tolist())
        neos=len(bary)

        #Will's code currently doesn't give sound speed, so this is a numerical derivative for it
        cs2=[]
        rho=np.flip(rho)
        pres=np.flip(pres)
        for i in range(len(rho)):
            if (i==0):
                step=1/100*(rho[i+1]-rho[i])
                cs2.append((np.interp(rho[i]+step,rho,pres)-np.interp(rho[i],rho,pres))/step)
            elif (i==len(rho)-1):
                step=1/100*(rho[i]-rho[i-1])
                cs2.append((np.interp(rho[i],rho,pres)-np.interp(rho[i]-step,rho,pres))/step)
            else:
                step1=1/100*(rho[i+1]-rho[i])
                step2=1/100*(rho[i]-rho[i-1])
                cs2.append((np.interp(rho[i]+step1,rho,pres)-np.interp(rho[i]-step2,rho,pres))/(step1+step2))
        rho=np.flip(rho)
        pres=np.flip(pres)
        cs2=np.flip(np.array(cs2)/((2.99792458e+10)**2))
        for i in range(len(cs2)):
          if (cs2[i]>1 and cs2[i]<1.01):cs2[i]=1 #small numerical errors can cause cs2>1, so handle that here
            
        #save EOS details in the same format as jerome's EOS building code
        self.form     = "tov"
        self.aeos     = np.column_stack((np.flip(np.log10(bary/1e+39)),
                                 np.flip(np.log10(rho)),
                                 np.flip(np.log10(pres)),
                                 np.flip(np.log10(cs2)),
                                 np.zeros(neos),
                                 np.zeros(neos),
                                 np.zeros(neos),
                                 np.flip(cs2),
                                 np.zeros(neos),
                                 np.zeros(neos)))
        self.lognb    = self.aeos[:,0]
        self.logrho   = self.aeos[:,1]
        self.logpre   = self.aeos[:,2]
        self.logcs2   = self.aeos[:,3]
        self.logh     = self.aeos[:,4]
        self.logye    = self.aeos[:,5]
        self.logym    = self.aeos[:,6]
        self.cs2      = self.aeos[:,7]
        self.ye       = self.aeos[:,8]
        self.ym       = self.aeos[:,9]
        self.neos     = neos
        self.nbmax    = np.max(bary)/1e+39
        self.rhomin   = np.min(rho)
        self.rhomax   = np.max(rho)
        #use shear modulus to locate crust-core transition
        for i in range(len(smod)-1):
            if(smod[i]==0 and smod[i+1]!=0):
                rhocc=rho[i+1]
        self.rhocc    = rhocc
        self.rhodrip  = rhodrip
        self.flags    = np.array([0,1,0,0],dtype=np.int32)
        self.n_unit   = "fm^{-3}"
        self.rho_unit = "g cm^{-3}"
        self.e2a_unit = "MeV"
        self.mu_unit  = "MeV"
        self.pre_unit = "dyn cm^{-2}"
        self.cs2_unit = "c^2"
        self.h_unit   = "MeV"


def generate_polycore_EOS(J,L,Ksym,n1,n2):
    success,dens,pres,bary,smod,ss,sp,gam1,gs=EOS_polycore.generate_eos(J,L,Ksym,n1,n2)
    if (not success.decode("utf-8").replace(" ","")=='Y'):  #check whether EOS generated successfully
        return False
    for i in range(len(dens)):
        if (dens[i]==0.0):
            lines=i
            break

    #trim lists returned from fortran
    dens=dens[0:lines]
    pres=pres[0:lines]
    bary=bary[0:lines]
    smod=smod[0:lines]
    ss=ss[0:lines]
    sp=sp[0:lines]
    gam1=gam1[0:lines]
    gs=gs[0:lines]

    #Get correct units for the shear modulus. Note that its value is in different units in the outer crust
    for i in range(len(dens)):
        if (dens[i]<4.3e+11): #4.3e+11 = outer/inner crust boundary
            smod[i]=smod[i]*1.6022e+33
        else:
            smod[i]=smod[i]*pres[i]

    #if there is instability, smooth over it (rarely occurs, most common when polytropes are attached)
    for i in range(len(dens)-1):
        if (pres[i+1]>pres[i] or dens[i+1]>dens[i]):
            pres[i]=(pres[i]+pres[i+1])/2.0
            dens[i]=(dens[i]+dens[i+1])/2.0   
            for j in range(len(pres)):
                if(j>i+1):
                    pres[j-1]=pres[j]
                    dens[j-1]=dens[j]
            del pres[len(pres)-1]
            del dens[len(dens)-1]
  
    #check max pressure is not tiny, and that mu=0 at the max pressure (i.e., that there is a core)
    if(pres[0]<1e+33 or smod[0]>0):
        return False

    EOS=Eos_nseospy_format(dens,pres,bary,smod,gam1)

    return EOS
