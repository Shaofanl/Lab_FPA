import cPickle
import numpy as np
import sys
import os

from sklearn import svm
#from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

def train_test_split(x, y, tag):
    X_train, X_test, y_train, y_test = [], [], [], []

    train = set(np.random.choice(range(1, 13), size=(6,), replace=False))
    test = set(range(1, 13)) - train

    for ind, ele in enumerate(tag):
        if int(ele.split('_')[0]) in train:
        #if np.random.randint(0, 2) == 1:
            X_train.append(x[ind])
            y_train.append(y[ind])
        else:
            X_test.append(x[ind])
            y_test.append(y[ind])

    X_train =  np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    #print 'train:', train, 'test:', test,
    #print 'set size: ', map(lambda x: x.shape, [X_train, X_test, y_train, y_test]),
    return X_train, X_test, y_train, y_test 

    
from kernels import chi_square_kernel, histogram_intersection_kernel, zero_kernel, multichannel_wrapper

if __name__ == '__main__':
    bowFilename = 'lab4/bow.pkl'

    if not os.path.exists(bowFilename):
        raise IOError("No such file '%s'."%bowFilename)

    x, y, tag = cPickle.load(open(bowFilename, 'rb'))

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=int)
    y = y-1
    

    y = label_binarize(y, classes=range(7))
    n_classes = 7
    X_train, X_test, y_train, y_test = train_test_split(x, y, tag)

    print("Training SVM")
    TIMES = 100
    l = []
    for i in range(TIMES):
        print '\rFitting %d/%d ' % (i, TIMES),
        sys.stdout.flush()

        # resampling
        classifier = OneVsRestClassifier(svm.SVC(kernel=multichannel_wrapper(2, chi_square_kernel), probability=True))
        X_train, X_test, y_train, y_test = train_test_split(x, y, tag)
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)

        l.append(float((y_test.argmax(1) == y_score.argmax(1)).sum())/y_score.shape[0]*100)
    print map(lambda x: '%.3f%%' % x, l), '=', np.mean(l)

    y_score = classifier.decision_function(x)
    print 'Test all = %.3f%%' % (float((y.argmax(1) == y_score.argmax(1)).sum())/y_score.shape[0]*100 )

    if True:
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import average_precision_score

        # Compute Precision-Recall and plot curve
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y[:, i], y_score[:, i])
            average_precision[i] = average_precision_score(y[:, i], y_score[:, i])

        # Compute micro-average ROC curve and ROC area
        precision["micro"], recall["micro"], _ = precision_recall_curve(y.ravel(), y_score.ravel())
        average_precision["micro"] = average_precision_score(y, y_score, average="micro")

        # Plot Precision-Recall curve
        plt.clf()
        plt.plot(recall[0], precision[0], label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
        plt.legend(loc="lower left")
        plt.show()

        # Plot Precision-Recall curve for each class
        plt.clf()
        plt.plot(recall["micro"], precision["micro"],
                label='micro-avr S={0:0.2f}' #'micro-average Precision-recall curve (area = {0:0.2f})'
                       ''.format(average_precision["micro"]))
        for i in range(n_classes):
            plt.plot(recall[i], precision[i],
                    label='S[{0}]={1:0.2f}' #'Precision-recall curve of class {0} (area = {1:0.2f})'
                           ''.format(i, average_precision[i]))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to multi-class')
        plt.legend(loc="upper right")
        plt.show()
