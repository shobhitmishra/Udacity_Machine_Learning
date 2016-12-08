import pandas as pd
from time import time
from sklearn.metrics import f1_score
def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    #print "Trained model in {:.4f} seconds".format(end - start)
    return (end - start)

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    #print "Made predictions in {:.4f} seconds.".format(end - start)
    prediction_time = (end - start)
    return prediction_time, f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    #print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
    
    # Train the classifier
    train_time = train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    #print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    #print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))
    train_set_prediction_time, train_set_F1_score = predict_labels(clf, X_train, y_train)
    test_set_prediction_time, test_set_F1_score = predict_labels(clf, X_test, y_test)
    return train_time, train_set_prediction_time, train_set_F1_score, test_set_prediction_time, test_set_F1_score

def multi_predict_result(clf, X_train, y_train, X_test, y_test, num_of_run):
    print "Set size is ", len(X_train)
    train_time_total , train_set_prediction_time_total , train_set_F1_score_total = [0.0 for _ in xrange(3)]
    test_set_prediction_time_total , test_set_F1_score_total = [0.0 for _ in xrange(2)]
    for _ in xrange(num_of_run):
        t1, t2, t3, t4, t5 = train_predict(clf, X_train, y_train, X_test, y_test)
        train_time_total += t1
        train_set_prediction_time_total += t2
        train_set_F1_score_total += t3
        test_set_prediction_time_total += t4
        test_set_F1_score_total += t5
    print "Trained model in {:.4f} seconds".format(train_time_total/num_of_run)    
    #print "Train_set predictions in {:.4f} seconds.".format(train_set_prediction_time_total/num_of_run)
    print "Test_set predictions in {:.4f} seconds.".format(test_set_prediction_time_total/num_of_run)
    print "F1 score for training set: {:.4f}.".format(train_set_F1_score_total/num_of_run)    
    print "F1 score for test set: {:.4f}.".format(test_set_F1_score_total/num_of_run)
