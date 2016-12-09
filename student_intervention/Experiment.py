import numpy as np
import pandas as pd
from Helper_functions import preprocess_features, train_predict, multi_predict_result
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn import linear_model

# Read student data
student_data = pd.read_csv("student-data.csv")
#print "Student data read successfully!"

feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1] 

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

X_all = preprocess_features(X_all)
#print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))

num_train = 300

num_test = X_all.shape[0] - num_train

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, train_size =num_train, random_state=42)

X_train_100 = X_train[0:100]
y_train_100 = y_train[0:100]

X_train_200 = X_train[0:200]
y_train_200 = y_train[0:200]

X_train_300 = X_train[0:300]
y_train_300 = y_train[0:300]

gaussian_clf = GaussianNB()
decision_tree_clf = DecisionTreeClassifier(random_state=0)
adaboost_clf = AdaBoostClassifier(n_estimators=100)
random_forest_clf = RandomForestClassifier(n_estimators=10)
gradient_boosting_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
KNearest_Neighbor_clf = KNeighborsClassifier(n_neighbors=3)
Gradient_Descent_clf = SGDClassifier(loss="hinge", penalty="l2")
Svm_clf = svm.SVC()
Logistic_regression_clf = linear_model.LogisticRegression(C=1e5)
bagging_clf = BaggingClassifier(base_estimator=gaussian_clf, max_samples=0.5, max_features=0.5)

clf = bagging_clf
multi_predict_result(clf, X_train_100, y_train_100, X_test, y_test, 10)
print ""
multi_predict_result(clf, X_train_200, y_train_200, X_test, y_test, 10)
print ""
multi_predict_result(clf, X_train_300, y_train_300, X_test, y_test, 10)
