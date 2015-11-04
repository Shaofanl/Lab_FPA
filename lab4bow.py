import cPickle
import numpy as np
import os
from sklearn.cluster import KMeans
import sys

def bagOfWord(filename):
    x, y, tag = cPickle.load(open(filename, 'rb'))
#    x = np.array(x)
#    print map(lambda x: x.shape, x)
    #print map(lambda x: x.shape, x)
    #print y
    K = 400

    all_x = np.concatenate(x)
    all_x = all_x.reshape(all_x.shape[0], -1)
    # normalize on descriptors
    all_x = np.array(all_x, dtype=float)/all_x.sum(1).reshape(all_x.shape[0], 1)
    print all_x.sum(1)
    print('Original shape:', all_x.shape)
    print('Clustering into %d categories ...'%K)

    # bag of words
    est = KMeans(n_clusters=K)
    labels = est.fit_predict(all_x)

    new_x = np.zeros((len(x), K))
    acc = 0
    for ind, video in enumerate(x):
        length = video.shape[0]
        new_x[ind] = np.bincount(labels[acc:acc+length], minlength=K)
        acc += length

    x = new_x
    y = np.array(y)
    tag = np.array(tag)

    # sort
    index = tag.argsort()
    x = x[index]
    y = y[index]
    tag = tag[index]

    print('New shape:', x.shape)
    print("="*60)
    return x, y, tag


if __name__ == '__main__':
    bowFilename = 'lab4/bow.pkl'
    globalFeaturesFilename = 'lab4/globalFeaturesRaw.pkl'
    localFeaturesFilename = 'lab4/localFeaturesRaw.pkl'
    globalFeaturesBoW = 'lab4/globalFeaturesBoW.pkl'
    localFeaturesBoW = 'lab4/localFeaturesBoW.pkl'


    for filename in [localFeaturesBoW, globalFeaturesBoW]:
        if '-f' in sys.argv or\
            not os.path.exists(filename) or\
            raw_input('%s existed, overwrite?[y/N]' % filename)=='y':

            x_bow, y, tag = bagOfWord(filename.replace('BoW','Raw'))
            cPickle.dump((x_bow, y, tag), open(filename, 'w'))


    if '-f' in sys.argv or\
        not os.path.exists(bowFilename) or raw_input('%s existed, overwrite?[y/N]' % bowFilename)=='y':
        for checkname in [globalFeaturesBoW, localFeaturesBoW]:
            if not os.path.exists(checkname):
                raise IOError("No such file '%s'."%checkname)

        global_x_bow, global_y, global_tag = cPickle.load(open(globalFeaturesBoW, 'r'))
        local_x_bow, local_y, local_tag = cPickle.load(open(localFeaturesBoW, 'r'))
    
        if (local_y != global_y).any():
            raise Exception("Local y and global y are not matching.")
        if (local_tag != global_tag).any():
            raise Exception("Local tag and global tag are not matching.")

        #n = local_x_bow.shape[0]
        #local_x_bow = local_x_bow.reshape(n, 1, -1)
        #global_x_bow = global_x_bow.reshape(n, 1, -1) 

        x_bow = np.concatenate([global_x_bow, local_x_bow], 1)
        print "Final Bag of Word shape:", x_bow.shape

        cPickle.dump((x_bow, local_y, local_tag), open(bowFilename, 'w'))


