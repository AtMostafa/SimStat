class TwoTailPermTest:
    """
    Permutation test as to whether there is significant difference between group one and two.
    
    group1, group2: Represent the data. they could be either one dimentional (several realizations)
        or 2-D (several realizaions through out the time/space/... course)
        EX: x.shape==(15,500) means 15 trials/samples over 500 time bins

    nIterations: Number of iterations used to shuffle. max(iterN)=(len(x)+len(y))!/len(x)!len(y)!

    initGlobConfInterval:
        Initial value for the global confidence band.

    smoothSigma: the standard deviation of the gaussian kernel used for smoothing when there are multiple data points,
        based on the Fujisawa 2008 paper, default value: 0.05

    Outputs:
        pVal: P-values
        highBand, lowBand: AKA boundary. Represents global bands.
        significantDiff: An array of True or False, indicating whether there is a difference.
    
    """  
    def __init__(self, group1, group2, nIterations=1000, initGlobConfInterval=5, smoothSigma=0.05, randomSeed=1):
        self.__group1, self.__group2 = self.__setGroupData(group1), self.__setGroupData(group2)
        self.__nIterations, self.__smoothSigma = nIterations, smoothSigma
        self.__initGlobConfInterval = initGlobConfInterval
        self.__randomSeed = randomSeed
        
        self.__checkGroups()

        # origGroupDiff is also known as D0 in the definition of permutation test.
        self.__origGroupDiff = self.__computeGroupDiff(group1, group2)

        # Generate surrogate groups, compute difference of mean for each group, and put in a matrix.
        self.__diffSurGroups = self.__setDiffSurrGroups()

        # Set statistics
        self.pVal = self.__setPVal()
        self.highBand, self.lowBand = self.__setBands()
        self.pairwiseHighBand = self.__setPairwiseHighBand()
        self.pairwiseLowBand = self.__setPairwiseLowBand()
        self.significantDiff = self.__setSignificantGroup()

    def __setGroupData(self, groupData):
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

    def __checkGroups(self):
        # input check
        if not isinstance(self.__group1, np.ndarray) or not isinstance(self.__group2, np.ndarray):
            raise ValueError("In permutation test, \"group1\" and \"group2\" should be numpy arrays.")

        if self.__group1.ndim > 2 or self.__group2.ndim > 2:
            raise ValueError('In permutation test, the groups must be either vectors or matrices.')

        elif self.__group1.ndim == 1 or self.__group2.ndim == 1:
            self.__group1 = np.reshape(self.__group1, (len(self.__group1), 1))
            self.__group2 = np.reshape(self.__group2, (len(self.__group2), 1))

    def __computeGroupDiff(self, group1, group2):
        meanDiff = np.nanmean(group1, axis=0) - np.nanmean(group2, axis=0)
        
        if len(self.__group1[0]) == 1 and len(self.__group2[0]) == 1:
            return meanDiff
        
        return smooth(meanDiff, sigma=self.__smoothSigma)

    def __setDiffSurrGroups(self):
        # Fix seed 
        np.random.seed(seed=self.__randomSeed)
        # shuffling the data
        self.__concatenatedData = np.concatenate((self.__group1,  self.__group2), axis=0)
        
        diffSurrGroups = np.zeros((self.__nIterations, self.__group1.shape[1]))
        for iteration in range(self.__nIterations):
            # Generate surrogate groups
            # Shuffle every column.
            np.random.shuffle(self.__concatenatedData)  

            # Return surrogate groups of same size.            
            surrGroup1, surrGroup2 = self.__concatenatedData[:self.__group1.shape[0], :], self.__concatenatedData[self.__group1.shape[0]:, :]
            
            # Compute the difference between mean of surrogate groups
            surrGroupDiff = self.__computeGroupDiff(surrGroup1, surrGroup2)
            
            # Store individual differences in a matrix.
            diffSurrGroups[iteration, :] = surrGroupDiff

        return diffSurrGroups
 
    def __setPVal(self):
        positivePVals = np.sum(1*(self.__diffSurGroups > self.__origGroupDiff), axis=0) / self.__nIterations
        negativePVals = np.sum(1*(self.__diffSurGroups < self.__origGroupDiff), axis=0) / self.__nIterations
        return np.array([np.min([1, 2*pPos, 2*pNeg]) for pPos, pNeg in zip(positivePVals, negativePVals)])

    def __setBands(self):
        if not isinstance(self.__origGroupDiff, np.ndarray):  # single point comparison
            return None, None
        
        alpha = 100 # Global alpha value
        highGlobCI = self.__initGlobConfInterval  # global confidance interval
        lowGlobCI = self.__initGlobConfInterval  # global confidance interval
        while alpha >= 5:
            # highBand = np.percentile(a=self.__diffSurGroups, q=100-highGlobCI, axis=0)
            # lowBand = np.percentile(a=self.__diffSurGroups, q=lowGlobCI, axis=0)

            highBand = np.percentile(a=self.__diffSurGroups, q=100-highGlobCI)
            lowBand = np.percentile(a=self.__diffSurGroups, q=lowGlobCI)

            breaksPositive = np.sum(
                [np.sum(self.__diffSurGroups[i, :] > highBand) > 1 for i in range(self.__nIterations)]) 
            
            breaksNegative = np.sum(
                [np.sum(self.__diffSurGroups[i, :] < lowBand) > 1 for i in range(self.__nIterations)])
            
            alpha = ((breaksPositive + breaksNegative) / self.__nIterations) * 100
            highGlobCI = 0.95 * highGlobCI
            lowGlobCI = 0.95 * lowGlobCI
        return highBand, lowBand           

    def __setSignificantGroup(self):
        if not isinstance(self.__origGroupDiff, np.ndarray):  # single point comparison
            return self.pVal <= 0.05

        # finding significant bins
        globalSig = np.logical_or(self.__origGroupDiff > self.highBand, self.__origGroupDiff < self.lowBand)
        pairwiseSig = np.logical_or(self.__origGroupDiff > self.__setPairwiseHighBand(), self.__origGroupDiff < self.__setPairwiseLowBand())
        
        significantGroup = globalSig.copy()
        lastIndex = 0
        for currentIndex in range(len(pairwiseSig)):
            if (globalSig[currentIndex] == True):
                lastIndex = self.__setNeighborsToTrue(significantGroup, pairwiseSig, currentIndex, lastIndex)

        return significantGroup
    
    def __setPairwiseHighBand(self, localBandValue=0.5):        
        return np.percentile(a=self.__diffSurGroups, q=100 - localBandValue, axis=0)

    def __setPairwiseLowBand(self, localBandValue=0.5):        
        return np.percentile(a=self.__diffSurGroups, q=localBandValue, axis=0)

    def __setNeighborsToTrue(self, significantGroup, pairwiseSig, currentIndex, previousIndex):
        """
            While the neighbors of a global point pass the local band (consecutively), set the global band to true.
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
    
    def plotSignificant(self,ax: plt.Axes.axes,y: float,x=None,**kwargs):
        if x is None:
            x=np.arange(0,len(self.significantDiff))+1
        for x0,x1,p in zip(x[:-1],x[1:],self.significantDiff):
            if p:
                ax.plot([x0,x1],[y,y],zorder=-2,**kwargs)
                
    @staticmethod
    def plotSigPair(ax: plt.Axes.axes,y: float,x=None, s: str ='*',**kwargs):
        if x is None:
            x=(0,len(self.significantDiff))
        if 'color' not in kwargs:
            kwargs['color']='k'
        
        dy=.03*(ax.get_ylim()[1]-ax.get_ylim()[0])
        ax.plot(x,[y,y],**kwargs)
        ax.plot([x[0],x[0]],[y-dy,y],[x[1],x[1]],[y-dy,y],**kwargs)
        ax.text(np.mean(x),y,s=s,
                ha='center',va='center',color=kwargs['color'],
                size='xx-small',fontstyle='italic',backgroundcolor='w')