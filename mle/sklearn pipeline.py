import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
from category_encoders import OneHotEncoder
from sklearn.model_selection import cross_val_predict
from warnings import filterwarnings
from sklearn.impute import SimpleImputer
filterwarnings('ignore')
import os
print(os.listdir("../input"))


train = pd.read_csv("../input/train.csv")
print("train shape", train.shape)
test = pd.read_csv("../input/test.csv")
print("test shape", test.shape)

target_column = "target"
id_column = "id"
categorical_cols = [c for c in test.columns if test[c].dtype in [np.object]]
numerical_cols = [c for c in test.columns if test[c].dtype in [np.float, np.int] and c not in [target_column, id_column]]
print("Number of features", len(categorical_cols)+len(numerical_cols))

# lightGBM



classifier = make_pipeline(
    ColumnTransformer([('num', StandardScaler(), numerical_cols), ('cat', OneHotEncoder(), categorical_cols),]), \
        LGBMClassifier(n_jobs=-1))

oof_pred = cross_val_predict(classifier, train, train[target_column], cv=5, method="predict_proba")



def datapipeline(path):

    df = pd.read_csv(path)
    # return a bool dataframe for NaN values
    bool_df_isnull = df.isnull()
    # return a bool dataframe for false NaN values
    bool_df_notnull = df.notnull()
    # creating bool series True for NaN values
    bool_series = pd.isnull(df["Gender"])


    # fill nan:
    # In order to fill null values in a datasets, we use fillna(), replace() and interpolate() function these function replace NaN values with some value of their own. 
    # fill nan with a singal value
    df.fillna(0)
    # Filling null value with the previous ones
    df.fillna(method ='pad')
    # Filling null value with the next ones 
    df.fillna(method ='bfill')
    # filling a null values using fillna()
    df["Gender"].fillna("No Gender", inplace = True)
    # will replace  Nan value in dataframe with value -99 
    df.replace(to_replace = np.nan, value = -99)
    # to interpolate the missing values
    df.interpolate(method ='linear', limit_direction ='forward')
    # drop nan values
    # using dropna() function 
    df.dropna()
    # Dropping rows if all values in that row are missing. 

    # TODO drop a row if specified value is missing
    df.dropna(subset=['name', 'toy'])

    df.dropna(how = 'all')
    # dropping columns if it has at least one nan value
    df.dropna(axis = 1)
    new_df = df.dropna(axis = 0, how ='any')

    # You can count missing values in each column by default, and in each row with axis=1
    df.isnull().sum()
    df.isnull().sum(axis = 1)
    df.count()
    df.count(axis=1)

    # check if there is at least one nan value
    df.isnull().values.sum() != 0
    # check if there is at least one not nan value
    df.isnull().values.sum() == df.size

    # For series
    s = df['state']
    print(s)
    # 0     NY
    # 1    NaN
    # 2     CA
    # Name: state, dtype: object

    print(s.isnull())
    # 0    False
    # 1     True
    # 2    False
    # Name: state, dtype: bool

    print(s.notnull())
    # 0     True
    # 1    False
    # 2     True
    # Name: state, dtype: bool

    print(s.isnull().any())
    # True

    print(s.isnull().all())
    # False

    print(s.isnull().sum())
    # 1

    print(s.count())
    # 2

    # count conditions matching
    df_bool = (df == 'CA')
    df_bool.sum()
    df_bool.sum(axis = 1)
    df_bool.sum().sum()


    # check data imbalance
    df.Class.value_counts() # class -> label column
    len(df.loc[df.Class==1]) / len(df.loc[df.Class == 0])


    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
    
    dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    accuracy_score(y_test, dummy_pred)


    # Modeling the data as is
    # Train model
    lr = LogisticRegression(solver='liblinear').fit(X_train, y_train)
    
    # Predict on training set
    lr_pred = lr.predict(X_test)
    # Checking accuracy
    accuracy_score(y_test, lr_pred)
    predictions = pd.DataFrame(lr_pred)
    predictions[0].value_counts()



    # train model
    rfc = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)

    # predict on test set
    rfc_pred = rfc.predict(X_test)

    accuracy_score(y_test, rfc_pred)
    0.999592708069998
    # f1 score
    f1_score(y_test, rfc_pred)
    pd.DataFrame(confusion_matrix(y_test, rfc_pred))
    recall_score(y_test, rfc_pred)


    # deal with imbalance dataset (resampling)
    not_fraud = X[X.Class==0]
    fraud = X[X.Class==1]

    # upsample minority
    fraud_upsampled = resample(fraud,
                            replace=True, # sample with replacement
                            n_samples=len(not_fraud), # match number in majority class
                            random_state=27) # reproducible results

    # combine majority and upsampled minority
    upsampled = pd.concat([not_fraud, fraud_upsampled])

    # check new class counts
    upsampled.Class.value_counts()
    y_train = upsampled.Class
    X_train = upsampled.drop('Class', axis=1)

    upsampled = LogisticRegression(solver='liblinear').fit(X_train, y_train)

    upsampled_pred = upsampled.predict(X_test)

    # downsample majority
    # still using our separated classes fraud and not_fraud from above

    # downsample majority
    not_fraud_downsampled = resample(not_fraud,
                                    replace = False, # sample without replacement
                                    n_samples = len(fraud), # match minority n
                                    random_state = 27) # reproducible results

    # combine minority and downsampled majority
    downsampled = pd.concat([not_fraud_downsampled, fraud])

    # checking counts
    downsampled.Class.value_counts()

    """Method 1: Rename Specific Columns df. rename(columns = {'old_col1':'new_col1', 'old_col2':'new_col2'}, inplace = True)
    Method 2: Rename All Columns df. columns = ['new_col1', 'new_col2', 'new_col3', 'new_col4']
    Method 3: Replace Specific Characters in Columns df."""
    #Drop the rows where all elements are missing.

    df.dropna(how='all')

    # Keep only the rows with at least 2 non-NA values.

    df.dropna(thresh=2) 
    # drop columns: df.drop(['B', 'C'], axis=1), df.drop(columns=['B', 'C'])

    # drop duplicate
    df.drop_duplicates()
    df.drop_duplicates(subset=[''])
    #To remove duplicates and keep last occurrences, use keep.

    df.drop_duplicates(subset=['brand', 'style'], keep='last')