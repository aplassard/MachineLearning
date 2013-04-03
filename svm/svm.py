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
                for d in D:
                    print 'Testing Model With C =',c,', Gamma =',g,', Kernel =',k,', and degree = ',d
                    testmodel = svm.SVC(C=c,kernel=k,gamma=g,degree=d)
                    testmodel.fit(training_data.training_features,training_data.training_labels)
                    p = testmodel.score(training_data.test_features,training_data.test_labels)
                    if p > percent:
                        model = testmodel
                        percent = p
                        params = 'C = '+str(c)+' Gamma = '+str(g)+' Kernel = '+str(k)+' degree = '+str(d)

    print
    print 'Params Chosen:',params,'with percent:',percent
    return model
