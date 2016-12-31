#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### Choosing all the features except email_id and other
features_list = ['poi',
				'salary', 
				'to_messages', 
				'deferral_payments', 
				'total_payments',
 				'exercised_stock_options',
 				'bonus',
				'restricted_stock', 
				'shared_receipt_with_poi', 
				'restricted_stock_deferred', 
				'total_stock_value',
			    'expenses', 
				'loan_advances', 
				'from_messages',
 				'from_this_person_to_poi', 
			    'director_fees',
			    'deferred_income', 
				'long_term_incentive',
				'from_poi_to_this_person'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0 )
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("LOCKHART EUGENE E",0)

### Task 3: Create new feature(s)
# Helper function to create a fraction
def computeFraction(poi_messages, all_messages):
	if poi_messages == "NaN" or all_messages == "NaN":
		return 0
 	fraction = float(poi_messages)/float(all_messages)
 	return fraction

 # Create two new features: fraction_from_poi & fraction_to_poi
for key, value in data_dict.iteritems():
 	
 	from_poi_to_this_person = value["from_poi_to_this_person"]
 	to_messages = value["to_messages"]
 	fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages )
 	data_dict[key]["fraction_from_poi"] = fraction_from_poi

	from_this_person_to_poi = value["from_this_person_to_poi"]
	from_messages = value["from_messages"]
	fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages )
	data_dict[key]["fraction_to_poi"] = fraction_to_poi

### Appending the new features fraction_to_poi and fraction_from_poi 
#   to the features list
features_list.append('fraction_to_poi')
features_list.append('fraction_from_poi')

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Select 5 best features using SelectKBest
from sklearn.feature_selection import SelectKBest, f_classif
kbest = SelectKBest(k=5)
selected_features = kbest.fit_transform(features, labels)
score = kbest.scores_

# Print features and scores
feature_scores = zip(features_list[1:], score)
sorted_dict = sorted(feature_scores, key=lambda feature: feature[1], reverse=True)
print "Features and scores"
for item in sorted_dict:
	print item[0], item[1]

# Scaling features
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
scaler = preprocessing.MinMaxScaler()

### Task 4: Try a variety of classifiers

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb_clf = Pipeline(steps = [("scaling", scaler), ("kbest", kbest),
	("GaussianNB", GaussianNB())])

# Decision tree
from sklearn.tree import DecisionTreeClassifier
dt_clf = Pipeline(steps = [("scaling", scaler), ("kbest", kbest),
	("DecisionTree", DecisionTreeClassifier())])

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_clf = Pipeline(steps =[("scaling", scaler), ("kbest", kbest),
	("LogisticRegression", LogisticRegression())])

# Cross validation
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
train_test_split(features, labels, test_size=0.3, random_state=42)

# Algorithm performance metric
from sklearn import metrics
from sklearn.metrics import accuracy_score

classifier = [nb_clf, dt_clf, lr_clf]
for c in classifier:
	clf = c
	clf.fit(features_train, labels_train)
	predictions = clf.predict(features_test)
	print "\n Classifier: ", clf
	print "Accuracy:", metrics.accuracy_score(labels_test, predictions)
	print "Precision score:", metrics.precision_score(labels_test, predictions)
	print "Recall score:", metrics.recall_score(labels_test, predictions)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Parameter tuning for decision tree
parameters = {'Dtree__criterion':('gini', 'entropy'),
			  'Dtree__min_samples_split':list(xrange(3,5)),
			  'Dtree__min_samples_leaf':list(xrange(3,10))}

from sklearn.grid_search import GridSearchCV
clf_dt = Pipeline(steps=[("scaling", scaler),("kbest", kbest), ("Dtree", DecisionTreeClassifier(random_state=0))])
clf_tuned = GridSearchCV(estimator=clf_dt, param_grid=parameters,scoring='f1')
clf_tuned.fit(features_train, labels_train)
predictions = clf_tuned.predict(features_test)

print '\n Decision Tree Performance after tuning
print "Accuracy:", metrics.accuracy_score(labels_test, predictions)
print "Precision score:", metrics.precision_score(labels_test, predictions)
print "Recall score:", metrics.recall_score(labels_test, predictions)
print clf_tuned.best_params_

# Example starting point. Try investigating other evaluation techniques!

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

# Choose Naive Bayes as the final algorithm
clf = nb_clf
dump_classifier_and_data(clf, my_dataset, features_list)