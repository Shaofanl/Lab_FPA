import cv2
import os
import cPickle
import numpy as np
import sys

def getGlobalMotionDescriptor(filename, plot=False,
    S = 8, DIRECTION_SOLUTION = 8, TIME_INTERVAL = 0.5, **kwargs):
    '''
        get global motion descriptor with Dense Optical Flow
        cv2.calcOpticalFlowFarneback(prev, next, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags[, flow]) -> flow 
    '''

    # video loader
    cap = cv2.VideoCapture(filename)

    # get the first frame
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_RGB2GRAY)

    # init
    if plot:
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
    CELL_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/S+1)
    CELL_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/S+1)
    FRAME_INTERVAL = int(cap.get(cv2.CAP_PROP_FPS)*TIME_INTERVAL)
    FRAME_COUNT = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    counter = 0
    des_list = [np.zeros((S, S, DIRECTION_SOLUTION))]

    for frame_index in xrange(1, int(FRAME_COUNT)):
        ret, frame2 = cap.read()
        if frame_index % FRAME_INTERVAL == 0: 
            des_list.append(np.zeros((S, S, DIRECTION_SOLUTION)) )

        print 'Press Esc to stop: %.3f%%\r' % (frame_index/FRAME_COUNT*100.),
        sys.stdout.flush()

        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        # cv2.calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags) -> flow 
        flow = cv2.calcOpticalFlowFarneback(prvs, next, 
                None,#flow
                0.5,    # pyrscale
                1, # levels: #pyramid layers
                2, #  winsize
                20, #iterations
                5, #poly_n 
                1.1, # poly_sigma 
                0) # flags

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

        # place angle into 8 predefined types
        ang = np.array(ang/(2*np.pi/DIRECTION_SOLUTION)-0.000001, dtype=int)
        #des = np.zeros((S, S, DIRECTION_SOLUTION))
        for (x, y), value in np.ndenumerate(ang):
            des_list[-1][x/CELL_HEIGHT, y/CELL_WIDTH, value] += 1
        #des_list.append(des)

        # ploting
        if plot:
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            cv2.imshow('frame2',rgb)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == ord('s'):
                cv2.imwrite('opticalfb.png',frame2)
                cv2.imwrite('opticalhsv.png',rgb)

        prvs = next

    cap.release()
    cv2.destroyAllWindows()

    print '[%s] Done.' % (filename)

    return np.array(des_list)


if __name__ == '__main__':
    #dataset_dir = './dataset'
    dataset_dir = '/home/share/shaofan/jpl_interaction_segmented_iyuv'
    globalFeaturesFilename = 'lab4/globalFeaturesRaw.pkl'

    if '-f' in sys.argv or\
            not os.path.exists(globalFeaturesFilename) or\
            raw_input('%s existed, overwrite?[y/N]' % globalFeaturesFilename)=='y':
        x = []
        y = []
        tag = []
        count = len(os.listdir(dataset_dir))
        for index, filename in enumerate(os.listdir(dataset_dir)):
            print 'Calc the features of %d/%d' % (index, count)
            features = getGlobalMotionDescriptor(os.path.join(dataset_dir, filename), 
                    plot=False, TIME_INTERVAL = 0.25)
            x.append(features)
            y.append(filename.split('.')[0].split('_')[1])
            tag.append(filename.split('.')[0])

        cPickle.dump((x, y, tag), open(globalFeaturesFilename, 'w'))


