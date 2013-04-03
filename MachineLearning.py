import numpy as np
from MLFunctions import *

class MachineLearning(object):
    '''
        Machine Learning Class for Capturing Basic Infrastructure
    '''

    def __init__(self,trainingfeatures,traininglabels):
        self._training_data = TrainingData(trainingfeatures,traininglabels)

class TrainingData(object):
    '''
        Training Data Class to Maintain the Training Data Portion for Machine Learning Applications
    '''

    def __init__(self,trainingfeatures,traininglabels):
        '''
            Initializer for TrainingData Class
            Input:
                trainingfeatures: NxM numpy array
                traininglabels:   List of length N where the elements are a meaningful identifier (i.e. "control")
        '''
        self._features,self._means,self._stds = normalizearray(trainingfeatures)
        self._labels = traininglabels
        self._get_training_dict()
        self._split_data()

    def _get_training_dict(self):
        n = 0
        self._int_dict = {}
        self._label_dict = {}
        self._int_labels = []
        for el in self._labels:
            val = self._label_dict.get(el)
            if val is None:
                val = n
                self._label_dict[el]=n
                self._int_dict[n]=el
                n+=1
            self._int_labels.append(val)

    def get_class(self,n):
        return self._int_dict.get(n)

    def _split_data(self):
        self.test_features,self.test_labels,self.training_features,self.test_features = splitTrainingTesting(self._features,self._int_labels)
