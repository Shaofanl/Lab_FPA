# use hog 3d

import cv2
import os
import numpy as np
import cPickle

hog3d = './extractFeatures_x86_64_v1.3.0'

def getLocalFeatures(filename):
    c = cv2.VideoCapture(filename)
    nfrm = c.get(cv2.CAP_PROP_FRAME_COUNT)

#    xy_stride = 40;
    xy_nstride = 3;
    t_nstride = int(nfrm/10);#3;
    xy_ncell = 4;
    t_ncell = 3;
    spatial_support_region = 32;
    t_support_region_factor = 1.0; # t_support_region = t_stride * t_support_region_factor;
    scale_overlap = 1; # no overlap between cells
    xy_max_scale = 1; # single scale used
    t_max_scale = 1;
    t_stride = nfrm / t_nstride;
    t_support_region = t_stride * t_support_region_factor;

    cmd = [hog3d, 
            '--xy-nstride', xy_nstride, 
            '--t-nstride', t_nstride, 
            '--xy-ncells', xy_ncell, 
            '--t-ncells', t_ncell, 
            '--sigma-support', spatial_support_region, 
            '--tau-support', t_support_region, 
            '--scale-overlap', scale_overlap, 
            '--xy-max-scale', xy_max_scale, 
            '--t-max-scale', t_max_scale, 
            filename ]
    cmd = ' '.join(map(str, cmd) )
    res = np.array(map(lambda x: x.strip().split(' '), os.popen(cmd).readlines()),  dtype=float)
    if not res.tolist():
        res = np.zeros((0, 0)) 
    print '\tfeatrues shape: ', res[:, 8:].shape
    return res[:, 8:]

if __name__ == '__main__':
    #dataset_dir = './dataset'
    dataset_dir = '/home/share/shaofan/jpl_interaction_segmented_iyuv/'
    localFeaturesFilename = 'lab4/localFeaturesRaw.pkl'

    if not os.path.exists(localFeaturesFilename) or raw_input('%s existed, overwrite?[y/N]'%localFeaturesFilename)=='y':
        x = []
        y = []
        tag = []
        count = len(os.listdir(dataset_dir))
        for index, filename in enumerate(os.listdir(dataset_dir)):
            print 'Calc the features of %d/%d' % (index, count)
            features = getLocalFeatures(os.path.join(dataset_dir, filename))

            x.append(features)
            y.append(filename.split('.')[0].split('_')[1])
            tag.append(filename.split('.')[0])

        cPickle.dump((x, y, tag), open(localFeaturesFilename, 'w'))
