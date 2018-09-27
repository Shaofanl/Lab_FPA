# First Person Activity

Recongnize activities/actions given a first person view.

## Workflow
- Global feature: Dense Optical Flow -> Catagorize in region by angle -> s\*s\*8
- Local feature: Harris3D -> HOG||HOF
- Create Bag-of-Words
- SVM(kernel=chi-square)

## Logs
### [PCA] -2015-11-04
- Add PCA to local features to reduced/unify the feature dimension.
- PCA on training or whole dataset
- The feature dimension is small enough (162)



### [Cluster only on TrainSet] -2015-11-04
- [lab5align.py]: Align local and global features.
- [lab5svm.py]:  Cluster and calculate the histogram after split the train/test set
