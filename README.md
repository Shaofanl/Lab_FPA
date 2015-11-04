First Person Activity
---------------------

### Workflow

- Global feature: Dense Optical Flow -> Catagorize in region by angle -> s\*s\*8
- Local feature: Harris3D -> HOG||HOF
- Create Bag-of-Words
- SVM(kernel=chi-square)
