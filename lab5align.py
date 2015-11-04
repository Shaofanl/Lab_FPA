import cPickle
import numpy as np
import os
import sys

if __name__ == '__main__':
    globalFeaturesFilename = 'lab4/globalFeaturesRaw.pkl'
    localFeaturesFilename = 'lab4/localFeaturesRaw.pkl'
    filename = alignedFeaturesFilename = 'lab4/alignedFeaturesRaw.pkl'

    if '-f' in sys.argv or\
        not os.path.exists(filename) or\
        raw_input('%s existed, overwrite?[y/N]' % filename)=='y':

        def tagsort(x, y, tag):
            index = np.argsort(map(lambda x: '%02d%02d' % tuple(map(int, x.split('_')[:2])), tag))
            x = np.array(x)[index]
            y = np.array(y)[index]           
            tag = np.array(tag)[index]
            return x, y, tag
        
        local_x, local_y, local_tag = \
            tagsort( *cPickle.load(open(localFeaturesFilename, 'rb')) )
        global_x, global_y, global_tag = \
            tagsort( *cPickle.load(open(globalFeaturesFilename, 'rb')) )

##      print local_tag == global_tag
##      print local_y == global_y
##      print global_y

        tag = np.array(map(lambda x: int(x.split('_')[0]), local_tag))
        local_x = np.array(map(lambda x: x.reshape(x.shape[0], -1), local_x) )
        global_x = np.array(map(lambda x: x.reshape(x.shape[0], -1), global_x) )
        y = np.array(local_y, dtype=int)-1

#       print local_x.shape
#       print map(lambda x: x.shape, local_x)
#       print global_x.shape
#       print map(lambda x: x.shape, global_x)

        cPickle.dump((local_x, global_x, y, tag), open(filename, 'wb'))
        
        


