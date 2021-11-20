# Cobra ROSboost for imbalance class problem

## Dataset 
### pima-indian-diabetes:
file: pima-indians-diabetes.csv \
--------------points\
majority class 500 \
minority class 268 

### data: 
file data.csv\
--------------points\
majority class 950 \
minority class 50 

## Directory Structure

```bash
│   CoBra.pdf
│   README.md
│
├───Data
│       haberman.csv
│       pima-indians-diabetes.csv
|       data.csv 
│
├───Notebooks
│       Cobra.ipynb //implementation of COBRA
│       SmoteAdaBoostedCC.ipynb //implementation of AdaBoost, SMOTE and ROSBoost using SMOTE for oversampling
|       CreateDataset.ipynb //creates imbalanced dataset. data.csv containd data created using this code with 95% points in majority class. 
│
└───Scripts
        Boosting.py //implementation of AdaBoost, SMOTE and ROSBoost using SMOTE for oversampling
        CobraClassifier.py //implemenatation of COBRA
```

## Results 
n_samples: number of points generated while oversampling\
k_neighbours: number of neighbours to be considered in SMOTE

### pima-indian-diabetes:

### Data:
