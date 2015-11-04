import cPickle
import numpy as np
from sklearn.cluster import KMeans
import sys
import os

from sklearn import svm
#from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier


def bagOfWord(x, K):
    all_x = np.concatenate(x.tolist())
    all_x = all_x.reshape(all_x.shape[0], -1)
    # normalization
    all_x = np.array(all_x, dtype=float)/all_x.sum(1).reshape(all_x.shape[0], 1)
    print('Original shape:', all_x.shape)
    print('Clustering into %d categories ...'%K)

    est = KMeans(n_clusters=K)
    est.fit(all_x)

    return est

def calcHistogram(codebook, x, K):
    all_x = np.concatenate(x)
    all_x = all_x.reshape(all_x.shape[0], -1)
    # normalize
    all_x = np.array(all_x, dtype=float)/all_x.sum(1).reshape(all_x.shape[0], 1)
    labels = codebook.predict(all_x) 

    new_x = np.zeros((len(x), K))
    acc = 0
    for ind, video in enumerate(x):
        length = video.shape[0]
        new_x[ind] = np.bincount(labels[acc:acc+length], minlength=K)
        acc += length

    print 'New shape:', new_x.shape
    return new_x

def train_test_split(local_x, global_x, y, tag):
    train = set(np.random.choice(range(1, 13), size=(6,), replace=False))
    test = set(range(1, 13)) - train
    train_index = [x for x in range(len(local_x)) if tag[x] in train] 
    test_index = [x for x in range(len(local_x)) if tag[x] in test] 
    print train
    print test
    print train_index
    print test_index

    K = 300
    local_codebook = bagOfWord(local_x[list(train_index)], K)
    local_x = calcHistogram(local_codebook, local_x, K)

    global_codebook = bagOfWord(global_x[list(train_index)], K)
    global_x = calcHistogram(global_codebook, global_x, K)

    x = np.concatenate([local_x, global_x], 1)
    print 'Final x.shape: ', x.shape

    X_train = x[train_index] 
    X_test = x[test_index] 
    y_train = y[train_index] 
    y_test = y[test_index] 
    return X_train, X_test, y_train, y_test 

    
from kernels import chi_square_kernel, histogram_intersection_kernel, zero_kernel, multichannel_wrapper

if __name__ == '__main__':
    # bowFilename = 'lab4/bow.pkl'
    alignedFeaturesFilename = 'lab4/alignedFeaturesRaw.pkl'

    local_x, global_x, y, tag = cPickle.load(open(alignedFeaturesFilename, 'rb'))

    y = label_binarize(y, classes=range(7))
    n_classes = 7

    print("Training SVM")
    TIMES = 100
    l = []
    for i in range(TIMES):
        print '\rFitting %d/%d ' % (i, TIMES),
        sys.stdout.flush()

        # resampling
        classifier = OneVsRestClassifier(svm.SVC(kernel=multichannel_wrapper(2, chi_square_kernel), probability=True))
        X_train, X_test, y_train, y_test = train_test_split(local_x, global_x, y, tag)
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
