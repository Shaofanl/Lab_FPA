import cv2
import os
import cPickle
import numpy as np
import sys

if __name__ == '__main__':
    globalFeaturesFilename = 'lab4/globalFeaturesRaw.pkl'
    localFeaturesFilename = 'lab4/localFeaturesRaw.pkl'

    for filename in [localFeaturesBoW, globalFeaturesBoW]:
        if '-f' in sys.argv or\
            not os.path.exists(filename) or\
            raw_input('%s existed, overwrite?[y/N]' % filename)=='y':

        pass

