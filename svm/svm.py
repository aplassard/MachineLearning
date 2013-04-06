#!/usr/bin/env python

from sklearn import svm

def build_svm(training_data,kernel=['linear','rbf','poly','sigmoid'],C=[0.01,0.1,1,10,100],G=[0.0001,0.001,0.01,0.1,1],D=[1,3,5,7,9]):
    percent = 0
    model = None
    params = None
    testModel = None

    for k in kernel:
        for c in C:
            for g in G:
                print 'Testing Model With C =',c,', Gamma =',g,', Kernel =',k
                testmodel = svm.SVC(C=c,kernel=k,gamma=g)
                testmodel.fit(training_data.training_features,training_data.training_labels)
                p = testmodel.score(training_data.test_features,training_data.test_labels)
                print 'The success rate was',p*100,'%'
                if p > percent:
                    model = testmodel
                    percent = p
                    params = 'C = '+str(c)+' Gamma = '+str(g)+' Kernel = '+str(k)

    print
    print 'Params Chosen:',params,'with percent:',percent
    return model
