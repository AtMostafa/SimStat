import simstat
import numpy as np

if __name__ == '__main__':
    data=np.random.normal(loc=1.5, scale=3, size=50)
    
    p=simstat.bootstrapTest(data,n=10000)
    print(p)
