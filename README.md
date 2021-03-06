# Cobra ROSboost for imbalance class problem

## Dataset 
### pima-indian-diabetes:
file: pima-indians-diabetes.csv \
--------------points\
majority class 500 \
minority class 268 

### data: 
file: data.csv\
--------------points\
majority class 950 \
minority class 50 

## Directory Structure

```bash
│   CoBra.pdf
│   README.md
|   SMOTEBoost results.pdf
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
SMOTEBoost results.pdf contains the results obtained. Results were obtained after applying ROSBoost using SMOTE on a cobra classifier made from 3 desision trees.\
\
\
\
\
DISCLAIMER: This work is for learning purposes only. The work can not be used for publications or commercial products etc. without the mentor’s consent.

