### Identify Fraud from Enron mail

### Project Overview
In 2000, Enron was one of the largest companies in the United States. By 2002, it  had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. This dataset is being used for this project to build a predictive model that will determine if a person is a “Person of interest”.

### Goal of this project
The goal of this project is to build a model that predicts if a person is a “Person of Interest” or not based on the financial and email data available from Enron scandal.

This is a supervised classification machine learning problem,  

* Classification problem, as the output we are trying to predict is the 2 discrete classes - POI or not POI
* Supervised,  as we also have the correct labels for the classes ( POI or non-POI) in the dataset . 

Hence machine learning can be used in this project to accomplish this goal.

### Dataset
The dataset contains 146 data points. These are the email and financial data of 146 people, most of whom are from the Senior management in Enron. It has 21 features including 14 financial features, 6 email features and 1 labeled feature (POI). Of the 146 records, 18 are labeled as POI and the remaining 128 records are labeled as non-POI.

### Sample data

{'METTS MARK': {  
'salary': 365788,  
'to_messages': 807,     
'deferral_payments': 'NaN',   
'total_payments': 1061827,  
 'exercised_stock_options': 'NaN',  
 'bonus': 600000,  
 'restricted_stock': 585062,   
'shared_receipt_with_poi': 702,   
'restricted_stock_deferred': 'NaN',   
'total_stock_value': 585062,  
 'expenses': 94299,   
'loan_advances': 'NaN',   
'from_messages': 29,  
 'other': 1740,  
 'from_this_person_to_poi': 1,   
'poi': False,  
 'director_fees': 'NaN',  
 'deferred_income': 'NaN',   
'long_term_incentive': 'NaN',  
 'email_address': 'mark.metts@enron.com',   
'from_poi_to_this_person': 38}  

### Missing data
As we can see in the above sample data point, there are some features that have missing values or NaN. I noticed a lot of missing values for features like ‘Loan advances’, ‘director fees’, ‘Restricted stock deferred’ etc. These NaN values will be replaced with 0.

### Outliers and Invalid data
On exploring the data, I found the following three outliers:

* TOTAL - From the scatter plot of “salary” & ‘bonus’, I found one outlier - TOTAL whose value is way off than other data points. This TOTAL is in fact  a spreadsheet quirk which has totalled all the data points and needs to be removed.
* THE TRAVEL AGENCY IN THE PARK - Another invalid record I found in the dataset after exploring the employee names was for a travel agency called ‘THE TRAVEL AGENCY IN THE PARK’ which had most of the values as NaN. This looks like a data entry error and hence removing this data point as well.
* LOCKHART EUGENE E - This record has all the feature values as NaN. Hence removing this data.

After removing these 3 outliers, the dataset has 143 records now.

### Feature selection and scaling
The dataset has 20 input features. To begin with, I have decided to use all the features except ‘email_address’ and ‘other’

### Adding 2 features:
In addition to the above features, I am creating 2 aggregated features:

* fraction_to_poi : Fraction of emails the person sent to poi
* fraction_from_poi : Fraction of emails the person received from poi

Based on the intuition that, POIs send emails to other POIs at a higher rate than for the general population, I created the two new features which are the fraction of emails a person sent to POI or received from POI.

I then used sklearn SelectKBest  to select the best features out of the 20 features and used these features in my algorithm.

The SelectKBest returned the following scores for all the features. 

#### Features and scores  
exercised_stock_options 		24.8150797332  
total_stock_value 			24.1828986786  
bonus 					20.7922520472  
salary 					18.2896840434  
fraction_to_poi 			16.409712548  
deferred_income 			11.4584765793  
long_term_incentive 		9.92218601319  
restricted_stock 			9.21281062198  
total_payments 			8.77277773009  
shared_receipt_with_poi 		8.58942073168  
Loan_advances			 7.18405565829  
expenses 				6.09417331064  
from_poi_to_this_person 		5.24344971337  
fraction_from_poi 			3.12809174816  
from_this_person_to_poi 		2.38261210823  
director_fees 			2.12632780201  
to_messages 			1.64634112944  
deferral_payments 			0.224611274736  
From_messages		            0.169700947622  
restricted_stock_deferred 		0.0654996529099  

SelectKBest computes an ANOVA F-statistic or score for each feature based on how different the feature is distributed across levels of our outcome variable. If we see a feature with a large score, then that means that the feature is distributed very differently across outcome labels and it is likely that it will be useful in a classification model. 

If we look at the above table of the sorted scores, we can gauge a cutoff value for k.  I picked k=5,  the top 5 features, since their scores were higher than the rest of the features there (from the fact that the score of the 5th feature was 16.41 and dropped off quite far to 11.46 for the feature right below it). 


### Top 5 Features I selected

exercised_stock_options 		24.8150797332  
total_stock_value 			24.1828986786  
bonus 					20.7922520472  
salary 					18.2896840434  
fraction_to_poi 			16.409712548  


### Feature scaling
I then scaled the above 5 features using sklearn MinMaxScaler()

### Algorithm choices
I tried the below classifiers

* Naive Bayes
* Decision tree
* Logistic Regression

### Performance of Algorithms

* Naive Bayes  
  Accuracy: 0.883720930233  
  Precision score: 0.5  
  Recall score: 0.6  

* Decision Tree  
  Accuracy: 0.813953488372  
  Precision score: 0.2  
  Recall score: 0.2  

* Logistic Regression  
  Accuracy: 0.883720930233  
  Precision score: 0.0  
  Recall score: 0.0  

Of the above 3 algorithms, Naive Bayes has the highest precision and recall score and hence I choose to use this algorithm for the project.

### Parameter tuning
Tuning the algorithm parameters is important to get the best performance. Sklearn GridSearchCV is a way of systematically working through multiple combinations of parameter tunes, cross-validating as it goes to determine which tune gives the best performance. 

Using sklearn GridSearchCV, I found the best parameters for Decision Tree as below. The precision has improved, but the recall is still the same after parameter tuning.  

Accuracy: 0.860465116279  
Precision score: 0.333333333333  
Recall score: 0.2  

{'Dtree__min_samples_leaf': 6,  
 'Dtree__criterion': 'gini',   
'Dtree__min_samples_split': 3}


### Validation
Validation is an important step in machine learning as it helps us estimate how well the algorithm performs with new data,  that is, beyond the training data it has seen already. It also serves as a check on overfitting.

One of the biggest mistakes one can make with validation is to use the same data for both training and testing. Overfitting can occur when the algorithm is fit too closely to the training data, such that it performs really well on the training data, but poorly on any other new unseen data. This is why it is important to always set aside data for testing, since testing on the same data used for training can lead to a misleading assessment of an algorithm’s performance.  

I did validation using cross validation, where I split the training data and use 70% of that data for training and remaining 30% of the data for testing. I then measured the evaluation metrics for all the three algorithms. Since there is an imbalance in the dataset between the number of POIs and non POIs, accuracy may not be an appropriate evaluation metric. Hence , I used precision as recall as evaluation metrics 

Precision is the ratio of  (true positives)  and (true positives + false positives). It tells us how well the model distinguishes between true POIs and false alarms. A precision score of 0.5 tells us that that if the model predicts 100 people as POIs, then the chances  are 50 people who are truly POI and the rest 50 people are non-POIs

Recall is the ratio of (true positives) and (true positives + false negatives). It tells us how well the model can detect a true POI. A recall score of 0.6 tells us that the model can find 60% of all true POIs in the prediction.

For this project, I believe having a higher recall is more important than having a higher precision. 

To identify as many POIs for further investigation seems to me as more important than trying to avoid false alarms and possibly letting a guilty individual escape scrutiny. Hence I chose the algorithm that has the higher recall value.

Cross validation using the provided StratifiedShuffleSplit method  returned the following evaluation  metrics for the Naive Bayes algorithm.   
 Precision: 0.42562       
 Recall: 0.35050        

