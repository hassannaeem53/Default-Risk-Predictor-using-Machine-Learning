# Credit-Risk-Modeling-Bandora
In this project we will be doing credit risk modelling of peer to peer lending Bondora systems.Data for the study has been retrieved from a publicly available data set of a leading European P2P lending platform (Bondora)..In P2P lending, loans are typically uncollateralized and lenders seek higher returns as a compensation for the financial risk they take. In addition, they need to make decisions under information asymmetry that works in favor of the borrowers. In order to make rational decisions, lenders want to minimize the risk of default of each lending decision, and realize the return that compensates for the risk. In this project we will preprocess the raw dataset and will create new preprocessed csv that can be used for building credit risk models.

## Table on Contents
- <a href='#understanding-the-dataset'>Understanding the Dataset</a>
- <a href='#preprocessing'>Preprocessing</a>
- <a href='#eda'>Exploratory Data Analysis</a>
- <a href='#feature-engineering'>Feature Engineering</a>
- <a href='#feature-scaling'>Feature Scaling</a>
- <a href='#spiliting-data-into-training-and-testing-sets'>Data Splitting</a>
- <a href='#model-building'>Model Building</a>
- <a href='#deployment'>Deployment</a>


## Understanding the Dataset
The retrieved data is a pool of both defaulted and non-defaulted loans from the time period between 1st March 2009 and 27th January 2020. The data comprises of demographic and financial information of borrowers, and loan transactions

- The news dataset contains the customer data from the time period between **2009** to **2020**. 

- The dataset contains **112** features and **134529** records. 

- The Target label of the dataset is whether the client is **defaulted** (labeled as **1**) or **Not Defaulted**
(labeled as **0**) in his period.


## Preprocessing 
### **Percentage of Missing Values**
Removing all the features which have more than 40% missing values.

### **Removal of Redundant Features**
- Removal features such as “Loan ID”, “Loan number”, “UserName”, and "DateOfBirth"(because age is already present), since they are assigned to each loan (or borrower) mainly for data storage and identification purposes and are meaningless for default prediction.

- As IncomeTotal is present, Removal of "IncomeFromPrincipalEmployer", "IncomeFromPension", "IncomeFromFamilyAllowance", "IncomeFromSocialWelfare","IncomeFromLeavePay", "IncomeFromChildSupport", and "IncomeOther" Features

- Removal unnecessary date features as we don't work on a time series study.

- Removal of other features that have no effects on our analysis.

### **Creating of Target Variable ie Loan Status**
Target Variable is created using **Status** feature of the provided dataset. 
The reason for not making status as target variable is that it has three unique values current, Late and repaid. There is no default feature but there is a feature default date which tells us when the borrower has defaulted means on which date the borrower defaulted. So, combining Status and Default date features for creating target variable.The reason we cannot simply treat Late as default because it also has some records in which actual status is Late but the user has never defaulted i.e., default date is null. So we will first filter out all the current status records because they are not matured yet they are current loans.

### **DataType Mismatch Correction**
After analysing dataset's numeric column distribution there are many columns which are present as numeric but they are actually categorical as per data description such as Verification Type, Language Code, Gender, Use of Loan, Education, Marital Status,EmployementStatus, OccupationArea etc.
So converting these features to categorical features.



## EDA

Dataset consist of different categorical and numerical distribution features.
Exploratory Data Analysis Process for this Projects involves plotting of the following.

### Categorical Features

- Percentage of Defaulted loans in the dataset.
- Residency of the Defaulters
- Gender distribution of defaulters
- Check if the most Defaluters are New Credit Customers or not 
- Language Code Distribution of the Defaulters
- Education wise distribution of defaulters
- Employment Status distribution of the Defaulters
- Marital Status distribution of the Defaulters
- Distribution of Loan purpose the Defaulters
- Distribution of Employment Duration of the Defaulters
- Distribution of Occupation Area of the Defaulters
- Distribution of Home Ownership Type of the Defaulters
- Distribution of Ratings of the Defaulters
- Distribution of Credit Score EsMicroL of the Defaulters 
### Numerical Features

- Age distribution of the Defaulters
   - Mean Age = Median Age which is about 40 years
   - The data is almost Symmetric.
- Monthly Payment distribution of the Defaulters
   - Mean value is 130, it's smaller and mean that the monthly payments of the Defaulters are smaller.
   - There are a large number of Outliers above the upper limit.
- AppliedAmount & Amount distribution of the Defaulters
- Previous Repayments Before Loan distribution of the Defaulters
   - Mean value of the repaid money by the defaulters before the loan is 861.138387
   - Data very skewed to the right and this means that the defaulters repaid small amount of money with a mean value about 861
   - There are alot of Ouliers.

### Correlation Plot of Numerical Variables
After Identifying outliers features, Heatmap is used to check the corelation of the features with the Target variable.
- There are featues that are positively correlated to each other.
- There are features that are negatively correlated to each other.
- Mostly, features are not correalted to each others.

![image](https://user-images.githubusercontent.com/80200407/208665863-9dfea992-0830-4c4d-ad89-b9d60f575a9e.png)


## Feature Engineering
### Handling outliers
- Data point that falls outside of 1.5 times of an interquantile range above the 3rd quartile and below the 1st quartile.
- Data point that falls outside of 3 standard deviations. We can use a Z-score, and if the Z-score falls outside of 2 standard deviation.
In EDA process outlier were already identified. There are different methods of dealing with outliers.
- Using Scatter plots
- Using Box plots
- Using Z-score
- Using the interquartile range (IQR)

```
# Loop for replacing outliers above upper bound with the upper bound value:
for column in df.select_dtypes([float, int]).columns :
   
    col_IQR = df[column].quantile(.75) - df[column].quantile(.25)
    col_Max =  df[column].quantile(.75) + (1.5*col_IQR)
    df[column][df[column] > col_Max] =  col_Max
```

```
# Loop for replacing outliers under lower bound with the lower bound value:
for column in df.select_dtypes([float, int]).columns :
    col_IQR = df[column].quantile(.75) - df[column].quantile(.25)
    col_Min =  df[column].quantile(.25) - (1.5*col_IQR)
    df[column][df[column] < col_Min] =  col_Min
```

## Feature Scaling

We used StandardScalar to scale our data:
StandardScaler is used to resize the distribution of values
- so that the mean of the observed values is 0 and the standard deviation is 1.
- The values will lie be between -1 and 1.

```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
rescaledValidationX = scaler.transform(X_valid)
```

**fit_transform()** is used on the training data so that we can scale the training data and also learn the scaling parameters of that data. Here, the model built by us will learn the mean and variance of the features of the training set. These learned parameters are then used to scale our test data.

**transform()** uses the same mean and variance as it is calculated from our training data to transform our test data. Thus, the parameters learned by our model using the training data will help us to transform our test data. As we do not want to be biased with our model, but we want our test data to be a completely new and a surprise set for our model.

## Feature Extraction and Dimensionality-reduction using (PCA)
Principal component analysis, or PCA, is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information.

The idea of PCA is simple — reduce the number of variables of a data set, while preserving as much information as possible.

Due to the nature of Dataset, it was observed that performing PCA negatively affected the accuracy of our models. So that is why we opt to leave this dimentionality reduction method.

## Spiliting Data into training and testing sets

80% of the dataset is consider as the training data while the remaining is used as testing data for our machine learning models.
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size = .8)
```


## Model Building

#### Metrics considered for Model Evaluation
**Accuracy , Precision , Recall and F1 Score**
- Accuracy: What proportion of actual positives and negatives is correctly classified?
- Precision: What proportion of predicted positives are truly positive ?
- Recall: What proportion of actual positives is correctly classified ?
- F1 Score : Harmonic mean of Precision and Recall

#### Logistic Regression
- Logistic Regression helps find how probabilities are changed with actions.
- The function is defined as P(y) = 1 / 1+e^-(A+Bx) 
- Logistic regression involves finding the **best fit S-curve** where A is the intercept and B is the regression coefficient. The output of logistic regression is a probability score.

#### Random Forest Classifier
- The random forest is a classification algorithm consisting of **many decision trees.** It uses bagging and features randomness when building each individual tree to try to create an uncorrelated forest of trees whose prediction by committee is more accurate than that of any individual tree.
- **Bagging and Boosting**: In this method of merging the same type of predictions. Boosting is a method of merging different types of predictions. Bagging decreases variance, not bias, and solves over-fitting issues in a model. Boosting decreases bias, not variance.
- **Feature Randomness**:  In a normal decision tree, when it is time to split a node, we consider every possible feature and pick the one that produces the most separation between the observations in the left node vs. those in the right node. In contrast, each tree in a random forest can pick only from a random subset of features. This forces even more variation amongst the trees in the model and ultimately results in lower correlation across trees and more diversification.

#### Linear Discriminant Analysis
- Linear Discriminant Analysis, or LDA, uses the information from both(selection and target) features to create a new axis and projects the data on to the new axis in such a way as to **minimizes the variance and maximizes the distance between the means of the two classes.**
- Both LDA and PCA are linear transformation techniques: LDA is supervised whereas PCA is unsupervised – PCA ignores class labels. LDA chooses axes to maximize the distance between points in different categories.
- PCA performs better in cases where the number of samples per class is less. Whereas LDA works better with large dataset having multiple classes; class separability is an important factor while reducing dimensionality.
- Linear Discriminant Analysis fails when the covariances of the X variables are a function of the value of Y.


### Choosing the features
After choosing LDA model based on confusion matrix here where **choose the features** taking in consideration the deployment phase.

Following Features were choosen for the deployment phase as they were observed to be highly corelated to the target varibale in the above steps.
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| 1. Bids portfolio Manager | 11. Refinance Liabilities | 21. Country |
| 2. Bids Api | 12. Debt To Income | 22. Use Of Loan |
| 3. Bids Manual | 13. Free Cash | 23. Education |
| 4. New Credit Customer | 14. Restructured | 24. Marital Status |
| 5. Age | 15. Principle Payment Made | 25. Employment Status |
| 6. Applied Amount | 16. Interest And Penalty Payments Made | 26. Employment Duration Current Employere |
| 7. Interest | 17. Previous Early Repayments Before Loan. | 27. Occurpation Area |
| 8. Monthly Payment | 18. Verification Type | 28. Home Ownership Type |
| 9. Income Total | 19. Language Code | 29. Rating |
| 10. Existing Liabilities | 20. Gender | 30. Credit Score Es MicroL |










## Deployment
you can access our app by following this link [stock-price-application-streamlit](https://stock-price-2.herokuapp.com/) or by click [stock-price-application-flask](https://stock-price-flask.herokuapp.com/)

### Application GUI
![image](https://user-images.githubusercontent.com/80200407/209480606-53f3603a-7735-40b4-92fa-f29baf7d9950.png)

### Streamlit
- It is a tool that lets you creating applications for your machine learning model by using simple python code.
- We write a python code for our app using Streamlit; the app asks the user to enter the following data (**news data**, **Open**, **Close**).
- The output of our app will be 0 or 1 ; 0 indicates that stock price will decrease while 1 means increasing of stock price.
- The app runs on local host.
- To deploy it on the internt we have to deploy it to Heroku.

### Heroku
We deploy our Streamlit app to [ Heroku.com](https://www.heroku.com/). In this way, we can share our app on the internet with others. 
We prepared the needed files to deploy our app sucessfully:
- Procfile: contains run statements for app file and setup.sh.
- setup.sh: contains setup information.
- requirements.txt: contains the libraries must be downloaded by Heroku to run app file (stock_price_App_V1.py)  successfully 
- stock_price_App_V1.py: contains the python code of a Streamlit web app.
- stock_price_xg.pkl : contains our XGBClassifier model that built by modeling part.
- X_train2.npy: contains the train data of modeling part that will be used to apply PCA trnsformation to the input data of the app.

### Flask 
We also create our app   by using flask , then deployed it to Heroku . The files of this part are located into (Flask_deployment) folder. You can access the app by following this link : [stock-price-application-flask](https://stock-price-flask.herokuapp.com/)
