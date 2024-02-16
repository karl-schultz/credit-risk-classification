# Module 20 Challenge: Report

## Overview of the Analysis

The purpose of this analytical project is to create a machine learning model on credit data from customers of a large banking institution. The financial data has eight variables. They are: 'loan_size', 'interest_rate', 'borrower_income', 'debt_to_income', 'num_of_accounts', 'derogatory_marks', 'total_debt', and'loan_status'. We create a model that seeks to predict one variable, our dependent variable 'loan_status', as a function of all the other variables defined as independent variables. The 'loan-status' variable is a categorical variable with either 0 or 1 as the value. 0 means that the loan is in good standing; 1 means that it is in default or has been determined to be high-risk for some other factor. Using the 'value_counts' pandas method on this variable, we find that there are many more examples of 'good credit' than 'bad credit': 75,036 examples of good credit vs only 2,500 examples of bad credit. This will have an effect on our model. Because we are creating a model making a binary determination of whether a customer has good or bad credit, we are using a logistical regression rather than a linear regression. A linear regression would be more suitable for determining a continuous dependent variable. To create our model we first define our y-variable as 'loan_status' and separate all the other variables into a separate dataframe, X. Then we split our sample into training and testing data. We 'train' our model using the logistical regression package from the scikitlearn on our training data. Then we test it on our separated testing data. Using f1-scores generated from our classification report, we find that the model is overall very good at predicting examples good credit, but only 88% effective at determining examples of bad credit. While this isn't superficially so bad, for an institution to use a model that is 12% inaccurate in finding bad creditors could expose our business to dangerous systemic risk. To improve our model, we run it again by 'oversampling' our data. Essentially, we randomly duplicate the values of 'bad' credit in the y data sample so the model has more training data to learn on. This results in a modest improvement of our model which is now 91% effective in predicting high-risk creditors. 

## Results

* Machine Learning Model 1 (No Oversampling):
  * Balanced Accuracy Score: .95
  * Precision Scores: {0:1, 1:.85} 
  * Recall Scores: {0:1, 1:.95} 
  * f1 Scores: {0:1, 1:.88} 

* Machine Learning Model 2 (Oversampled):
  * Balanced Accuracy Score: .99
  * Precision Scores: {0:1, 1:.84} 
  * Recall Scores: {0:.99, 1: 1} 
  * f1 Scores: {0:1, 1:.91} 

## Summary
While these models provide decent working prototypes of a credit-risk machine learning model, we cannot recommend either of them in their current form to go 'live' for business purposes. The f1 score for 1's or examples of bad credit is only 88 percent accurate overall. This doesn't sound bad, but when you consider that banking institutions lend credit to thousands if not millions of customers, a 12 percent inaccuracy rate could expose businesses using this model to substantial sytemic risk. Even when we modify our model by oversampling examples of bad credit, this only improves our f-1 accuracy in finding risky creditors to 91%. This is better, but still not good enough to mitigate substantial risk. In our opinion, if we are looking to create a working model we need far more than the 2500 raw examples of bad credit to train our predictive regression. At a minimum the model we already have would require substantial 'tuning' and that would still likely be insufficient.
