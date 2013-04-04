#!/usr/bin/env python

import sys
from MachineLearning import *

def main(argv):
	f = open(argv[1])
	labels = []
	features = []
	print '-- Loading Data From File --'
	for line in f:
		line = line.strip().split('\t')
		labels.append(line[0])
		features.append(line[1:])
	print '-- Converting Data Type --'
	features = np.array(features,dtype=np.float64)
	print '-- Creating Machine Learning Module --'
	ml = MachineLearning(features,labels)

if __name__=='__main__':
	main(sys.argv)
