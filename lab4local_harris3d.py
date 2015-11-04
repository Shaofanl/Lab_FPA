# use harris3d

#import cv2
import os
import numpy as np
import cPickle
import sys

MAX_DES = 500

if __name__ == '__main__':
    harris3d_dir = './tools/stip-2.0-linux'
    POI = os.path.join(harris3d_dir, 'mine/POI.txt')

    localFeaturesFilename = 'lab4/localFeaturesRaw.pkl'

    if '-f' in sys.argv or \
            not os.path.exists(localFeaturesFilename)  or\
            raw_input('%s existed, overwrite?[y/N]'%localFeaturesFilename)=='y':
        if not os.path.exists(POI):
            raise Exception("Cannot find Point Of Interest file!")

        # './bin/stipdet -i ./mine/video-list.txt -vpath /home/share/shaofan/jpl_interaction_segmented_iyuv/ -o ./data/POI.txt -det harris3d -vis no'
        x = []
        y = []
        tag = []
        tmp = []

        with open(POI, 'r') as f:
            new_flag = False
            for line in f:
                if new_flag:
                    print 'handling %d ...\r' % len(tag),
                    sys.stdout.flush()
                    tag.append(line.strip().split()[1])
                    y.append(tag[-1].split('_')[1])

                    if tmp:
                        tmp = np.array(tmp)
                        l = len(tmp)
                        if l > MAX_DES:
                            index = np.random.randint(l, size=MAX_DES)
                            tmp = tmp[index, :]
                        x.append(tmp)
                        tmp = []

                    new_flag = False
                    continue
                if line.strip() == '' or line.startswith('# point-type'):
                    new_flag = True
                    continue
                data = map(float, line.strip().split())
                # hof [9: 9+72]
                # hof [9+72: ]
                tmp.append(data[9: ])

        if tmp:
            tmp = np.array(tmp)
            l = len(tmp)
            if l > MAX_DES:
                index = np.random.randint(l, size=MAX_DES)
                tmp = tmp[index, :]
            x.append(tmp)

        print 
        print len(x)
        print map(len, x)

        cPickle.dump((x, y, tag), open(localFeaturesFilename, 'w'))

