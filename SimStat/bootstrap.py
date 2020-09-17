from sklearn.utils import check_random_state
import numpy as np
import logging

def bootstrap(data, n_bootstraps=10000, user_statistic=lambda x:np.mean(x,axis=1), kwargs=None, pass_indices=False, random_state=1):
    """Compute bootstraped statistics of a dataset.
    
    inputs
    ----------
    data : array_like
        An n-dimensional data array of size n_samples by n_attributes
    n_bootstraps : integer
        the number of bootstrap samples to compute.  Note that internally,
        two arrays of size (n_bootstraps, n_samples) will be allocated.
        For very large numbers of bootstraps, this can cause memory issues.
    user_statistic : function
        The statistic to be computed.  This should take an array of data
        of size (n_bootstraps, n_samples) and return the row-wise statistics
        of the data. default: lambda x:np.mean(x,axis=1)
    kwargs : dictionary (optional)
        A dictionary of keyword arguments to be passed to the
        user_statistic function.
    pass_indices : boolean (optional)
        if True, then the indices of the points rather than the points
        themselves are passed to `user_statistic`
    random_state: RandomState or an int seed (0 by default)

    Returns
    -------
    distribution : ndarray
        the bootstrapped distribution of statistics (length = n_bootstraps)

    code from [https://github.com/astroML/astroML/blob/master/astroML/resample.py]
    """
    # we don't set kwargs={} by default in the argument list, because using
    # a mutable type as a default argument can lead to strange results
    if kwargs is None:
        kwargs = {}

    rng = check_random_state(random_state)
    data = np.asarray(data)
    if data.ndim != 1:
        n_samples = data.shape[0]
        logging.warning("bootstrap data are n-dimensional: assuming ordered n_samples by n_attributes")
    else:
        n_samples = data.size

    # Generate random indices with repetition
    ind = rng.randint(n_samples, size=(n_bootstraps, n_samples))
    data = data[ind].reshape(-1, data[ind].shape[-1])
    # Call the function
    if pass_indices:
        stat_bootstrap = user_statistic(ind, **kwargs)
    else:
        stat_bootstrap = user_statistic(data, **kwargs)
        
    return stat_bootstrap

def bootstrapTest(data,n=10000):
    """
    Wrapper for 'bootstrap'
    input: data: 1D ndarray
    output: whether 'data's bootstrap distribution differs from zero
    p--> p-value
    """
    b_data=bootstrap(data[~np.isnan(data)],n)
    CI=np.nanpercentile(b_data,[5,95])
    
    if np.prod(CI) >0:
        N=len(b_data[b_data<0]) if CI[0]>0 else len(b_data[b_data>0])
        p=N/n
    return p


if __name__ == '__main__':
    data=np.random.normal(loc=1, scale=.5, size=150)
    
    p=bootstrapTest(data,n=10000)
    print(p)
