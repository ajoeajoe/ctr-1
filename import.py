from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


### Utility functions ###
def make_categorical(df):
    """
    :param df: pandas.DataFrame of categorical variables
    :return: pandas.DataFrame with categorical variables encoded as factors
    """
    return df.apply(LabelEncoder().fit_transform, axis=0, args=None)


### Import data ###
print "Loading data..."
test_df = pd.read_csv("/Users/Bill/Downloads/test.csv").drop(["hour"], axis=1) # ignore hour
train_df = pd.read_csv("/Users/Bill/Downloads/train.csv", nrows=10000).drop(["id", "hour"], axis=1) #ignore id, hour

train_x = train_df.drop(["click"], axis=1)
train_y = train_df["click"] # only 160 'clicks', must account for skewmes
test_x = test_df.drop(["id"], axis=1)
train_categorical_x = make_categorical(train_x)
test_categorical_x = make_categorical(test_x)
print "Data loaded...."

### Training ###
print "Training..."
pipeline = Pipeline([
    ("encode", OneHotEncoder(sparse=False, n_values=100000)),
#    ("feature_select", SelectKBest(f_regression, k=10)),
#   ("classify", SVC(kernel="linear", probability=True))
    ("classify", RandomForestClassifier(n_jobs=-1, verbose=True))
])
pipeline.fit(train_categorical_x, train_y)
print "Training complete...."

### Prepare results ###
print "Predicting...."
p = pipeline.predict_proba(test_categorical_x)
result = pd.concat([test_df["id"], pd.DataFrame(p[:, 1])], axis = 1)
result.to_csv("/Users/Bill/Desktop/results.csv", header=["id", "click"], index=False)
print "All done."





