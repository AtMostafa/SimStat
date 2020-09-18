import numpy as np

__all__=['permTest'] #everything that will be imported by import *, like in __init__

class permTest:
    """
    A general-purpose permutation test, based on "Fujisawa 2008 Nature Neuroscience"
    To test whether there is a difference between group one and two.
    
    group1, group2: Represent the data.
                    They could be either 1D (several realizations)
                    or 2D (several realisaions through out the time/space/... course)
    
    EX: x.shape==(15,500) means 15 trials/samples over 500 time bins

    nIterations: Number of iterations used to shuffle. max(iterN)=(len(x)+len(y))!/len(x)!len(y)!

    initGlobConfInterval: Initial value for the global confidence band.

    smoothSigma: the standard deviation of the gaussian kernel used for smoothing when there are multiple data points, default value: 0.05

    Outputs:
        pVal: P-values
        highBand, lowBand: AKA boundary. Represents global bands.
        significantDiff: A binary array, indicating whether there is a difference.
    
    based on code by M. A. Sharbaf
    """  
    def __init__(self, group1, group2, nIterations=10000, initGlobConfInterval=5, smoothSigma=0.05, randomSeed=1):
        self.group1, self.group2 = self.setGroupData(group1), self.setGroupData(group2)
        self.nIterations, self.smoothSigma = nIterations, smoothSigma
        self.initGlobConfInterval = initGlobConfInterval
        self.randomSeed = randomSeed
        
        self.checkGroups()

        # origGroupDiff is also known as D0 in the paper.
        self.origGroupDiff = self.computeGroupDiff(group1, group2)

        # Generate surrogate groups, compute difference of mean for each group, and put in a matrix.
        self.diffSurGroups = self.setDiffSurrGroups()

        # Set statistics
        self.pVal = self.setPVal()
        self.highBand, self.lowBand = self.setBands()
        self.pairwiseHighBand = self.setPairwiseHighBand()
        self.pairwiseLowBand = self.setPairwiseLowBand()
        self.significantDiff = self.setSignificantGroup()

    def setGroupData(self, groupData):
        if not isinstance(groupData, dict):
            return groupData

        realizations = list(groupData.values())
        subgroups = list(groupData.keys())
                    
        dataMat = np.zeros((len(subgroups), len(realizations[0])))
        for index, realization in enumerate(realizations):
            if len(realization) != len(realizations[0]):
                raise Exception("The length of all realizations in the group dictionary must be the same")
            
            dataMat[index] = realization

        return dataMat

    def checkGroups(self):
        # input check
        if not isinstance(self.group1, np.ndarray) or not isinstance(self.group2, np.ndarray):
            raise ValueError("In permutation test, \"group1\" and \"group2\" should be numpy arrays.")

        if self.group1.ndim > 2 or self.group2.ndim > 2:
            raise ValueError('In permutation test, the groups must be either vectors or 2D matrices.')

        elif self.group1.ndim == 1 or self.group2.ndim == 1:
            self.group1 = np.reshape(self.group1, (len(self.group1), 1))
            self.group2 = np.reshape(self.group2, (len(self.group2), 1))

    def computeGroupDiff(self, group1, group2):
        meanDiff = np.nanmean(group1, axis=0) - np.nanmean(group2, axis=0)
        
        if len(self.group1[0]) == 1 and len(self.group2[0]) == 1:
            return meanDiff
        
        return smooth(meanDiff, sigma=self.smoothSigma)

    def  setDiffSurrGroups(self):
        # Fix seed 
        np.random.seed(seed=self.randomSeed)
        # shuffling the data
        self.concatenatedData = np.concatenate((self.group1,  self.group2), axis=0)
        
        diffSurrGroups = np.zeros((self.nIterations, self.group1.shape[1]))
        for iteration in range(self.nIterations):
            # Generate surrogate groups
            # Shuffle every column.
            np.random.shuffle(self.concatenatedData)  

            # Return surrogate groups of same size.            
            surrGroup1, surrGroup2 = self.concatenatedData[:self.group1.shape[0], :], self.concatenatedData[self.group1.shape[0]:, :]
            
            # Compute the difference between mean of surrogate groups
            surrGroupDiff = self.computeGroupDiff(surrGroup1, surrGroup2)
            
            # Store individual differences in a matrix.
            diffSurrGroups[iteration, :] = surrGroupDiff

        return diffSurrGroups
 
    def setPVal(self):
        positivePVals = np.sum(1*(self.diffSurGroups > self.origGroupDiff), axis=0) / self.nIterations
        negativePVals = np.sum(1*(self.diffSurGroups < self.origGroupDiff), axis=0) / self.nIterations
        return np.array([np.min([1, 2*pPos, 2*pNeg]) for pPos, pNeg in zip(positivePVals, negativePVals)])

    def setBands(self):
        if not isinstance(self.origGroupDiff, np.ndarray):  # single point comparison
            return None, None
        
        alpha = 100 # Global alpha value
        highGlobCI = self.initGlobConfInterval  # global confidance interval
        lowGlobCI = self.initGlobConfInterval  # global confidance interval
        while alpha >= 5:
            # highBand = np.percentile(a=self.diffSurGroups, q=100-highGlobCI, axis=0)
            # lowBand = np.percentile(a=self.diffSurGroups, q=lowGlobCI, axis=0)

            highBand = np.percentile(a=self.diffSurGroups, q=100-highGlobCI)
            lowBand = np.percentile(a=self.diffSurGroups, q=lowGlobCI)

            breaksPositive = np.sum(
                [np.sum(self.diffSurGroups[i, :] > highBand) > 1 for i in range(self.nIterations)]) 
            
            breaksNegative = np.sum(
                [np.sum(self.diffSurGroups[i, :] < lowBand) > 1 for i in range(self.nIterations)])
            
            alpha = ((breaksPositive + breaksNegative) / self.nIterations) * 100
            highGlobCI = 0.95 * highGlobCI
            lowGlobCI = 0.95 * lowGlobCI
        return highBand, lowBand           

    def setSignificantGroup(self):
        if not isinstance(self.origGroupDiff, np.ndarray):  # single point comparison
            return self.pVal <= 0.05

        # finding significant bins
        globalSig = np.logical_or(self.origGroupDiff > self.highBand, self.origGroupDiff < self.lowBand)
        pairwiseSig = np.logical_or(self.origGroupDiff > self.setPairwiseHighBand(), self.origGroupDiff < self.setPairwiseLowBand())
        
        significantGroup = globalSig.copy()
        lastIndex = 0
        for currentIndex in range(len(pairwiseSig)):
            if (globalSig[currentIndex] == True):
                lastIndex = self.setNeighborsToTrue(significantGroup, pairwiseSig, currentIndex, lastIndex)

        return significantGroup
    
    def setPairwiseHighBand(self, localBandValue=0.5):        
        return np.percentile(a=self.diffSurGroups, q=100 - localBandValue, axis=0)

    def setPairwiseLowBand(self, localBandValue=0.5):        
        return np.percentile(a=self.diffSurGroups, q=localBandValue, axis=0)

    def setNeighborsToTrue(self, significantGroup, pairwiseSig, currentIndex, previousIndex):
        """
            While the neighbors of a global point pass the local band, set the global band to true.
            Returns the last index which was set to True.
        """ 
        if (currentIndex < previousIndex):
            return previousIndex
        
        for index in range(currentIndex, previousIndex, -1):
            if (pairwiseSig[index] == True):
                significantGroup[index] = True
            else:
                break

        previousIndex = currentIndex
        for index in range(currentIndex + 1, len(significantGroup)):
            previousIndex = index
            if (pairwiseSig[index] == True):
                significantGroup[index] = True
            else:
                break
        
        return previousIndex

if __name__ == '__main__':
    group1=np.random.normal(loc=1, scale=.1, size=150)
    group2=np.random.normal(loc=1.2, scale=.1, size=150)
    
    pT=permTest(group1, group2, nIterations=10000, initGlobConfInterval=5, smoothSigma=0.05, randomSeed=1)
    print(pT.pVal)