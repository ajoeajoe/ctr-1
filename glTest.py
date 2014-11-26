__author__ = 'Bill'
import graphlab as gl
import math

## Load training data
train = gl.load_sframe("/Users/Bill/Dropbox/cs151/ctr/full_train_data")
train.remove_column("hour")
train.remove_column("id")
train["click"] = train["click"].astype(int)

subset = train[1:100]

print train.groupby("click", {'count': gl.aggregate.COUNT()})
clicked = train.filter_by([1], "click")
no_click = train.filter_by([0], "click")
no_skew = clicked.append(no_click.sample(0.2))
print no_skew.groupby("click", {'count': gl.aggregate.COUNT()})

## Train model
svm_model = gl.svm_classifier.create(subset, target="click")
full_boosted_model = gl.load_model("/Users/Bill/Dropbox/cs151/ctr/full_boosted_model")
full_logistic_model = gl.load_model("/Users/Bill/Dropbox/cs151/ctr/full_logistic_model")


## Testing
test = gl.load_sframe("/Users/Bill/Dropbox/cs151/ctr/full_test_data")
results = gl.SFrame()
results["id"] = test["id"]
test.remove_column("hour")
test.remove_column("id")
ttest = test[1:100]


## Predict
svm_pred = svm_model.predict(ttest, output_type="margin").apply(lambda x: 1.0 / (1.0 + math.exp(-x)))
boosted_pred = full_boosted_model.predict(ttest, output_type="probability")
logistic_pred = full_logistic_model.predict(ttest, output_type="probability")


## Write results
results["click"] = boosted_pred * 0.6 + logistic_pred * 0.2 + svm_pred * 0.2
print results
#results.save("/Users/Bill/Desktop/joinedResults.csv", format="csv")


