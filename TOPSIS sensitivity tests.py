import sys
from abc import ABC

### math
import numpy as np
from scipy.stats import rankdata # rank with equal values
from scipy.stats import wilcoxon # Wilcoxon's test

### data
import pandas as pd

### Visualization
import matplotlib.pyplot as plt
from tabulate import tabulate

###! (-> my lib.) Indices of intersection of two arrays
def inters_index(ar1,ar2):
    inds=[]
    for i, x in enumerate(ar1):
        for j, y in enumerate(ar2):
            if x == y:
                inds+=[[i,j]]
    return inds


## Given a numpy array m, and list of [if column is on max. scale],
# normalizes each column so that min=0, max=1
def normalize(m,q=None):
    if q is None:
        l=[1 for i in m[0]]
    else:
        l=[1 if i else -1 for i in q]
    m=np.array(m)*np.array(l)
    mx=m.max(axis=0)
    mn=m.min(axis=0)
    return (m-mn)/(mx-mn)

### Object of MADM problem contains the decision matrix

class MAProblem:
    # .n - nr. of alternatives (int)
    # .k - nr. of objectives (int)
    # .M - decision matrix n x k (numpy ndarray)
    # .qMax - list (k) [is it maximization criterion (Boolean)]
    # ._matmax (np.ndarray or None) the maximization variant of decision matrix 
    # ._methods - list of MAMethod object set to solve the problem 
    def __init__(self,dec_matr,
                 ifmax=True, # all criteria are maximized
                 ifmatmax=False # need to produce max. variant of decision matrix 
                 ):
        ## initializing attributes
        self.M = np.array(dec_matr)
        self.n, self.k = self.M.shape
        if type(ifmax)==bool:
            self.qMax=np.array([ifmax for i in range(self.k)])
        else:
            self.qMax=np.array(ifmax) #! add exception if len(ifmax)!=self.k
        ## Check if max. variant of decision matrix is needed
        self._matmax=None
        if ifmatmax:
            self.matmax()
        self._methods=[]
            
    ## Returns the max. version of decision matrix (_matmax);
    #  creates one if create==True; returns existing if exists
    def matmax(self,create=True):
        if self._matmax is None:
            if create:
                if all(self.qMax):
                    # all maximize => link to M for saving space
                    self._matmax=self.M
                else:
                    xmax=np.array([1 if q else -1 for q in self.qMax])
                    self._matmax=self.M * xmax
                return self._matmax
            else:
                xmax=np.array([1 if q else -1 for q in self.qMax])
                return self.M * xmax
        else:
            return self._matmax
        
    ## Optionally, connect the problem with a method set to solve the problem         
    def addmethod(self,method):
        method.setproblem(self)
        self._methods.append(method)

### Abstract class of MADM method
# ._problem (MAProblem or None) - reference to the problem method solves
# ._k (int) - number of criteria
class MAMethod(ABC):
    def __init__(self,*args,**kwargs):
        self._problem=None
        self._k=None
        pass
    ## After the method object was connected to a problem,
    # do initial calculations
    def _initproblem(self, *args, **kwargs):
        self._k=self._problem.M.shape[1]
        pass
    ## Connect the method object with a specific MAProblem
    #  and do necessary calculations
    def setproblem(self,problem,**kwargs):
        self._problem = problem
        self._initproblem(**kwargs)
    ## Produces ranking
    def rank(self,**kwargs):
        pass

### Abstract class of MADM method based on scoring alternatives
# ._scores - list of scores assigned to each alternative (np.array of numbers)
class MAMethod_scoring(MAMethod):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._scores=None
        pass
    def _initproblem(self,*args, **kwargs):
        super()._initproblem()
        self._scores=None
    ## Produces scores of the alternatives -> ._scores
    def _make_scores(self, *args, **kwargs):
        pass
    ## Produces and returns ranking:
    # type=list of int; i-th element = rank for alternative i;
    # ranks start from 1; equivalent objects have same rank = minimal among them
    def rank(self,*args,**kwargs):
        if self._scores is None:
            self._make_scores()
        return rankdata(self._scores, method='min')

### TOPSIS method with the possibility to modify scoring function coordinat-wise
# It works with the max.-representation of the decision matrix 
# ._PIS, ._NIS - corresponding vectors of matmax (np.array)
# ._weights - np.array of weights
# _f_P, _f_N - list of functions (float -> float): for each criterion,
#    transform (PIS[i]-x[i]), (x[i]-NIS[i]) when calculating score of attr.vec. x
# _f_score - function (dist. to PIS, dist. to NIS)->score
class TOPSISmethod(MAMethod_scoring):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._PIS=None
        self._NIS=None
        self._weights=None
        self._f_P=kwargs.get("f_P",None)
        self._f_N=kwargs.get("f_N",None)
        self._weights=kwargs.get("weights",None)
        self._set_score_f(kwargs.get("f_score",None))
        self._f_score=None
        pass
## After problem is set for the method: calculate PIS and NIS;
#  set transformation functions if requested, or if they are not set yet;
#  set the scoring function if requested, or if it is not set yet
    def _initproblem(self,
                    f_P=None,f_N=None, # function arrays for _f_P,_f_N
                    f_score=None, # function for calculating score
                    weights=None
                    ):
        super()._initproblem()
        self._PIS=self._problem.matmax().max(axis=0)
        self._NIS=self._problem.matmax().min(axis=0)
        if (self._f_P is None) or (f_P is not None):
            self._set_Pfunc(f_P)
        if (self._f_N is None) or (f_N is not None):
            self._set_Nfunc(f_N)
        if (self._f_score is None) or (f_score is not None):
            self._set_score_f(f_score)
        if (self._weights is None) or (weights is not None):
            self._set_weights(weights)
        
## Setting two coordinate-wise functions, the score function and weights
    def _set_Pfunc(self,f_P=None):
        if f_P is None:
            if self._problem is not None:
                self._f_P=[(lambda x:x) for i in range(self._problem.k)]
        else:
            self._f_P=f_P
    def _set_Nfunc(self,f_N=None):
        if f_N is None:
            if self._problem is not None:
                self._f_N=[(lambda x:x) for i in range(self._problem.k)]
        else:
            self._f_N=f_N
    def _set_score_f(self,f_score=None):
        if f_score is None:
            self._f_score=( lambda d_P,d_N : d_N / (d_N + d_P) )
        else:
            self._f_score=f_score
    def _set_weights(self,weights=None):
        if weights is None:
            if self._problem is not None:
                self._weights=np.array([1 for i in range(self._problem.k)])
        else:
            self._weights=np.array(weights)


## Creates the list of scores for all corresponding alternatives       
    def _make_scores(self):
        self._scores=np.array([
                self._f_score(
                    # PIS-x part    
                    np.linalg.norm(
                        self._weights*[
                                fi(xi)
                            for fi,xi in zip(self._f_P,self._PIS-x)]
                            ),
                    # NIS-x part
                    np.linalg.norm(
                        self._weights*[
                                fi(xi)
                            for fi,xi in zip(self._f_N,x-self._NIS)]
                            ),                                            
                        )                
                for x in self._problem.matmax()])

### Calculating rank reversals
## Average of absolute rank difference
def rrev_avg(r1,r2):
    return (sum(np.array(r1-r2)**2)/len(r1))**0.5
    

#####################################
if __name__ == "__main__":    
    datadir="C:\\Mytemp\\Dropbox\\Science\\Papers\\MSDM2019\\Paper\\data\\"
    dfiles=["Normal_","Crisis_"]

    # Values of perturbation intensities as fractions 
    # of std. dev.
    pert_frac=[0.01,0.05,0.1]
    
    ## Setting TOPSIS methods
    # coefficients for s-shape function
    alpha=0.88
    beta=0.88
    lmb=2.25
    # function for modified difference to PIS
    def fpis(x):
        global beta,lmb
        return lmb*x**beta
    # function for modified difference to NIS
    def fnis(x):
        global alpha
        return x**alpha
    # initialising methods
    meths=[TOPSISmethod(),
           TOPSISmethod(f_P=[fpis for i in range(4)],f_N=[fnis for i in range(4)])]
    meths1=[TOPSISmethod(),
           TOPSISmethod(f_P=[fpis for i in range(4)],f_N=[fnis for i in range(4)])]
    
    
    # Results: (methods) x (data files) x (perturbation intensities) x (mean, deviation)
    results=np.zeros((len(meths),len(dfiles),len(pert_frac),2))
    # modif. / non-modif. ratios, and Wilcoxon's test:
    # (data files) x (perturbation intensities)
    ratios=np.zeros((len(dfiles),len(pert_frac)))
    wlx_test=np.zeros((len(dfiles),len(pert_frac)))
    
    dlen=[0,0] # nr. of decision variants in both problems
    # Nr of experimets
    nex=10000
    for idata in range(len(dfiles)):
        # reading data
        d=np.genfromtxt(
                datadir+dfiles[idata]+".txt", delimiter="\t")
        dlen[idata]=len(d)
        print(len(d))
        # calculating original rankings
        pr=MAProblem(normalize(d,[True,False,True,True]),ifmatmax=True)
        orig_rank=[] # (methods) x (alternatives)
        for i,m in enumerate(meths):
            m.setproblem(pr)
            orig_rank+=[m.rank()]
        # std. deviation of each criterion
        stdev=d.std(axis=0)
        # set of experiments for each perturbation intensity
        for pri,pert_f in enumerate(pert_frac):
            # nr. of rank reversals in each test: (nr of tests) x (nr of methods)
            restest=np.zeros((nex,len(meths)))
            # repeat tests, each done with all methods
            for iex in range(nex):
                dpert=normalize(
                        d+np.random.randn(*d.shape)*pert_f*stdev,
                        [True,False,True,True])
                pr1=MAProblem(dpert,ifmatmax=True)
                for i,m1 in enumerate(meths1):
                    m1.setproblem(pr1)
                    restest[iex,i]=rrev_avg(m1.rank(),orig_rank[i])
            # calculate average data
            means=restest.mean(axis=0)
            stdevs=restest.std(axis=0)
            for i in range(len(meths)):
                results[i,idata,pri,0]=means[i]
                results[i,idata,pri,1]=stdevs[i]
            # calculate means ratio
            ratios[idata,pri]=means[1]/means[0]
            # calculate p-value of Friedman test
            wlx_test[idata,pri]=wilcoxon(*restest.transpose())[1]
    
    print(results)
    print(ratios)
    print(wlx_test)
    np.save(datadir+"results",results)
    np.save(datadir+"ratios",ratios)
    np.save(datadir+"wlx_test",wlx_test)
    
    # For more analysis without repeating experiments,
    # comments the above code and uncomment data reading
    # results=np.load(datadir+"results2-2-3.npy")
    datanames=["Normal period","Crisis period"]
    for i in [0,1]:
        dmain=pd.DataFrame(
                results[:,i,:,0].transpose(),
                index=pert_frac,
                columns=["Original","Modified"])
        derr=pd.DataFrame(
                results[:,i,:,1].transpose(),
                index=pert_frac,
                columns=["Original","Modified"])
        fig, ax = plt.subplots(figsize=[20,12])
        plt.rcParams["font.size"]=32
        ax.set_title(datanames[i])
        ax=dmain.plot.bar(yerr=derr, ax=ax, capsize=4, 
                          color=["white","grey"],
                          linewidth=4,
                          edgecolor="black")
        ax.grid(which="both", axis="y")
        print(ax.get_position())
        plt.show()
    for i in [0,1]:
            d=np.genfromtxt(
                datadir+dfiles[i]+".txt", delimiter="\t")
            print(tabulate(
                    np.transpose([d.mean(axis=0),d.std(axis=0)]),
                    floatfmt=".2f"))
