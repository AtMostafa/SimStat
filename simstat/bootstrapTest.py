from . import bootstrap
import numpy as np

def bootstrapTest(data,n=10000):
    """
    Wrapper for 'bootstrap'
    input: data: 1D ndarray
    output: whether 'data's bootstrap distribution differs from zero
    p--> p-value
    """
    b_data=bootstrap.bootstrap(data[~np.isnan(data)],n)
    CI=np.nanpercentile(b_data,[5,95])
    
    p=1
    if np.prod(CI) >0:
        N=len(b_data[b_data<0]) if CI[0]>0 else len(b_data[b_data>0])
        p=N/n
    return p


if __name__ == '__main__':
    data=np.random.normal(loc=.1, scale=3, size=150)
    
    p=bootstrapTest(data,n=10000)
    print(p)
