# Basics of ML

## Linear Regression

### Step 1: Read Data
'''
Read input data from csv using pd.read_csv
'''
### Step 2: Clean Data
1) Replace null values with median.
2) Remove outiers using IQR (Interquantile Range) method. Calculate p25 and p275, IQR = p75 - p25 and then create upper and lower bounds using IQR.

### Step 3: 
1) Feature creation: Use pd.get_dummies for categorical variables
