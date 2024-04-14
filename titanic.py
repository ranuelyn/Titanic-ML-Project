# -*- coding: utf-8 -*-
"""
Titanic EDA.

@author: yusuf
"""
#%% Importing the Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
plt.style.use("dark_background")

import seaborn as sns

from collections import Counter

import warnings
warnings.filterwarnings('ignore')


#%% Load and Check Data
    
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
test_PassengerId = test_df['PassengerId']

train_df.columns
train_df.head()
train_df.describe()

#oldMoran = train_df[train_df['Name'] == 'Moran, Mr. James']



#%% Variable Description

"""
Variables:
PassengerId - unique id number to each passenger
Survived    - passenger survive(1) or died(0)
Pclass      - passenger class
Name        - name
Sex         - gender of passenger
Age         - age of passenger
SibSp       - number of siblings/spouses
Parch       - number of parents/children
Ticket      - ticket number
Fare        - amount of money spent on ticket
Cabin       - cabin category
Embarked    - port where passenger embarked (C = Cherbourg, Q = Queenstown, S = Southampton)


Types of Variables:
float64(2)  - Fare ve Age
int64(5)    - Pclass, sibsp, parch, passengerId and survived
object(5)   - Cabin, embarked, ticket, name and sex

"""

train_df.info()


#%% Univariate Variable Analysis

# Categorical Variable : Survived, Sex, Pclass, Embarked, Cabin, Name, Ticket, Sibsp and Parch

def bar_plot(variable):
    """
        input   : variable ex: 'Sex'
        output  : bar plot & value count
    """
    # get feature
    var = train_df[variable]
    # count number of categorical variable (value / sample)
    varValue = var.value_counts()
    
    # visualize
    plt.figure(figsize=(9, 3))
    plt.bar(varValue.index, varValue)
    print("varvalue", varValue.index)
    print("**************")
    print(varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
 
    plt.title(variable)
    plt.show()
    print('{}: \n{}'.format(variable, varValue))
    
category1 = ["Survived","Sex","Pclass","Embarked","SibSp", "Parch"]
for a in category1:
    bar_plot(a)

category2 = ['Cabin', 'Name', 'Ticket']
for b in category2:
    print("{} \n".format(train_df[b].value_counts()))


#%% Numerical Variable : Age, PassengerId, Fare

def plot_hist(variable):
    plt.figure(figsize = (9, 3))
    plt.hist(train_df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.title('{} distribution with histogram.'.format(variable))
    plt.show()
    
numericVar = ['Fare', 'Age', 'PassengerId']

for c in numericVar:
    plot_hist(c)


#%% Basic Data Analysis

# Pclass - Survived
# Sex    - Survived
# SibSp  - Survived
# Parch  - Survived

# Pclass - Survived
train_df[['Pclass','Survived']].groupby(['Pclass'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)

# Sex    - Survived
train_df[['Sex','Survived']].groupby(['Sex'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)

# Sibsp vs Survived
train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by="Survived",ascending = False)

# Parch vs Survived
train_df[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived",ascending = False)


# Age vs Survived
dfAge = pd.cut(train_df['Age'], bins = [0, 10, 20, 40, 65, np.inf])
dfAge = pd.DataFrame({'Age' : dfAge.values})
dfAge['Survived'] = train_df.Survived
dfAge[["Age","Survived"]].groupby(["Age"], as_index = False).mean().sort_values(by="Age",ascending = False)

    
#%% Outlier Detection

"""
1st Quartile = Q1
------------------   -> 2nd Quartile = Q2
3rd Quartile = Q3

IQR = Q3 - Q1
IQR * 1.5 = Outlier Detecter (O.D.)
Q1 - O.D. || Q3 + O.D.
other elements on the list rather then this range is an outlier.
"""

def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        # IQR
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # store indeces
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers

train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]

# drop outliers
train_df = train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)


#%% Find and Fill the Missing Values

# We gonna find it for the test too. Just run it once.
train_df_len = len(train_df)
train_df = pd.concat([train_df, test_df], axis = 0).reset_index(drop = True)

# Find the Missing Values

train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()

# Embarked has 2 and fare has 1 missing values.
# Embarked.
train_df[train_df['Embarked'].isnull()]
train_df.boxplot(column = 'Fare', by = 'Embarked')
plt.show()
train_df['Embarked'] = train_df['Embarked'].fillna('C')

# Fare. 
train_df[train_df['Fare'].isnull()]
train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))

#%% Fill the missing Age feature values.
"""
train_df[train_df['Age'].isnull()]

sns.catplot(x = 'Sex', y = 'Age', data = train_df, kind = 'box')
plt.show()

# Sex is not informative for age prediction. Age distrubition seems to be same.

sns.catplot(x = 'Sex', y = 'Age', hue = 'Pclass', data = train_df, kind = 'box')
plt.show()

# 1st class passengers is older then others. 

sns.catplot(x = 'Parch', y = 'Age', data = train_df, kind = 'box')
sns.catplot(x = 'SibSp', y = 'Age', data = train_df, kind = 'box')
plt.show()
"""
train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]]


sns.heatmap(train_df[['Age', 'Sex', 'SibSp', 'Parch', 'Pclass']].corr(), annot = True)
plt.show()

# Age is not corelated with sex but it is correlated with parch, sibsp and parch.

indexNonAge = list(train_df['Age'][train_df['Age'].isnull()].index)
for i in indexNonAge:
    agePrediction = train_df['Age'][((train_df['SibSp'] == train_df.iloc[i]['SibSp']) & (train_df['Parch'] == train_df.iloc[i]['Parch']) & (train_df['Pclass'] == train_df.iloc[i]['Pclass']))].median() 
    ageMedian = train_df['Age'].median()
    if not np.isnan(agePrediction):
        train_df['Age'].iloc[i] = agePrediction
    else:
        train_df['Age'].iloc[i] = ageMedian

train_df[train_df["Age"].isnull()]

newMoran = train_df[train_df['Name'] == 'Moran, Mr. James']

# Moran, Mr. James now has a age of 25.




#%% Visualization

# Correlation Between -> Sibsp -- Parch -- Age -- Fare -- Survived

list1 = ["SibSp", "Parch", "Age", "Fare", "Survived"]
sns.heatmap(train_df[list1].corr(), annot = True, fmt = ".2f")
plt.show() # annot -> to see values on heatmap, fmt -> to see that long number after the comma.

# Fare feature seems to have correlation with survived feature (0.28).


#%% SibSp -- Survived
# factorplot -> cat plot, size -> height bcs of the update on sns.
g = sns.catplot(x = 'SibSp', y = 'Survived', data = train_df, kind = 'bar', height = 6)
g.set_ylabels('Survived Probability')
plt.show()

# if SibSp <= 2, survivor probability is higher.


#%% Parch -- Survived

g = sns.catplot(x = 'Parch', y = 'Survived', data = train_df, kind = 'bar', height = 6)
g.set_ylabels('Survived Probability')
plt.show()

# SibSp and Parch can be used for new feature extraction with threshold 3.
# Small families have more chance to survive.


#%% Pclass -- Survived

g = sns.catplot(x = 'Pclass', y = 'Survived', data = train_df, kind = 'bar', height = 6)
g.set_ylabels('Survived Probability')
plt.show()

# If the Pclass level is higher is more likely to be survived.


#%% Age -- Survived 

g = sns.FacetGrid(train_df, col = 'Survived')
g.map(sns.distplot, 'Age', bins = 25)
plt.show()

# Age <= 10 has a high survived rate.
# Older passengers (80) survived.
# Large number of 20 years old didn't survived.
# Most of the passengers are in the age range of 15-25.
# Use age distrubiton for missing values of ages.


#%% Pclass -- Survived -- Age

g = sns.FacetGrid(train_df, col = 'Survived', row = 'Pclass')
g.map(plt.hist, 'Age', bins = 25)
g.add_legend()
plt.show()

# Pclass is important feature fore model training.


#%% Embarked -- Sex -- Pclass -- Survived

g = sns.FacetGrid(train_df, row = 'Embarked')
g.map(sns.pointplot, 'Pclass', 'Survived', 'Sex')
g.add_legend()
plt.show()

# Female passengers have much better survival rate then males.
# Male passengers have better survival rate in Pclass 3 in Embarked C.
# Embarked and sex will be used in training.


#%% Embarked -- Sex -- Fare -- Survived

g = sns.FacetGrid(train_df, row = 'Embarked', col = 'Survived')
g.map(sns.barplot, 'Sex', 'Fare')
g.add_legend()
plt.show()

# Passengers who paid high fare have better survival rate.
# Fare can be used as categorical in training.
# Embarked C > S > Q in survival rate.



#%% Feature Engineering

# Name -- Title

train_df['Name'].head(10)

name = train_df['Name']
train_df['Title'] = [i.split('.')[0].split(',')[-1].strip() for i in name]

sns.countplot(x = 'Title', data = train_df)
plt.xticks(rotation = 60)
plt.show()

#%%
# convert to categorical

train_df['Title'] = train_df['Title'].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")
train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_df["Title"]]

#%%
# look at the survival rate of these new titles.

g = sns.catplot(x = 'Title', y = 'Survived', data = train_df, kind = 'bar')
g.set_xlabels(['Master', 'Mrs', 'Mr', 'Other'])
g.set_ylabels('Survival Probability')
plt.show()

#%%
# drop the name column

train_df.drop(labels = ['Name'], axis = 1, inplace = True)
train_df = pd.get_dummies(train_df,columns=["Title"])


#%% Family Size
train_df['FSize'] = train_df['SibSp'] + train_df['Parch'] + 1 # 1 for himself/herself

g = sns.barplot(x = 'FSize', y = 'Survived', data = train_df)
#g.set_ylabels('Survival Probability')
plt.show()
train_df["family_size"] = [1 if i < 5 else 0 for i in train_df["FSize"]]
sns.countplot(x = "family_size", data = train_df)
plt.show()
#train_df = train_df.dropna(subset = ['Survived'])

#%% 
g = sns.catplot(x = "family_size", y = "Survived", data = train_df, kind = "bar")
g.set_ylabels("Survival")
plt.show()

# Small familes have more chance to survive than large families.

train_df = pd.get_dummies(train_df, columns= ["family_size"])
train_df.head()

#%% Embarked

train_df["Embarked"].head()
sns.countplot(x = "Embarked", data = train_df)
plt.show()

train_df = pd.get_dummies(train_df, columns=["Embarked"])
train_df.head()

#%% Ticket

#train_df['Ticket'].head(20)
ticketList = []

for i in list(train_df.Ticket):
    if not i.isdigit():
        ticketList.append(i.replace('.','').replace('/','').strip().split(' ')[0])
    else:
        ticketList.append('X')
        
train_df['Ticket'] = ticketList

#train_df['Ticket'].head(20)

#%% categorize the ticket

train_df = pd.get_dummies(train_df, columns = ['Ticket'], prefix = 'T') # Name with using T rather than using Ticket_

#train_df.head(20)

#%% Pclass

sns.countplot(x = 'Pclass', data = train_df)
plt.show()

train_df['Pclass'] = train_df['Pclass'].astype('category')
train_df = pd.get_dummies(train_df, columns = ['Pclass'])
train_df.head()

#%% Sex

train_df['Sex'] = train_df['Sex'].astype('category')
train_df = pd.get_dummies(train_df, columns = ['Sex'])
#train_df.head(20)

#%% Drop Passenger ID and Cabin

train_df.drop(labels = ['PassengerId', 'Cabin'], axis = 1, inplace = True)

#%% Modeling Imports

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#%% Train-Test Split

test = train_df[train_df_len:]
test.drop(labels = ['Survived'], axis = 1, inplace = True)

train = train_df[:train_df_len]
XTrain = train.drop(labels = 'Survived', axis = 1)
yTrain = train['Survived']

XTrain, XTest, yTrain, yTest = train_test_split(XTrain, yTrain, test_size = 0.33, random_state = 43)

print("XTrain",len(XTrain))
print('XTest', len(XTest))
print("yTrain",len(yTrain))
print("yTest",len(yTest))
print("test",len(test))

#%% Simple Logistic Regression

logReg = LogisticRegression()
logReg.fit(XTrain, yTrain)
accLogTrain = round(logReg.score(XTrain, yTrain) * 100,2)
accLogTest = round(logReg.score(XTest, yTest) * 100,2)

print("Training Accuracy: % {}".format(accLogTrain)) 
print("Testing Accuracy: % {}".format(accLogTest))

#%% Hyperparameter Tuning -- Grid Search -- Cross Validation
"""
We will compare 5 ml classifier and evaluate mean accuracy of each of them by stratified cross validation.
Decision Tree
SVM
Random Forest
KNN
Logistic Regression
"""

random_state = 42
classifier = [DecisionTreeClassifier(random_state= random_state),
              SVC(random_state = random_state),
              RandomForestClassifier(random_state = random_state),
              LogisticRegression(max_iter = 3000, random_state = random_state),
              KNeighborsClassifier(),
              ]

dt_param_grid = {'min_samples_split' : range(10, 500, 20),
                 'max_depth' : range(1, 20, 2)}

svc_param_grid = {'kernel' : ['rbf'],
                  'gamma' : [0.001, 0.01, 0.1, 1],
                  'C' : [1, 10, 50, 100, 200, 300, 1000]}

rf_param_grid = {"max_features": [1,3,10],
                "min_samples_split":[2,3,10],
                "min_samples_leaf":[1,3,10],
                "bootstrap":[False],
                "n_estimators":[100,300],
                "criterion":["gini"]}

logreg_param_grid = {"C":np.logspace(-3,3,7),
                    "penalty": ["l1","l2"]}

knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),
                 "weights": ["uniform","distance"],
                 "metric":["euclidean","manhattan"]}

classifier_param = [dt_param_grid,
                   svc_param_grid,
                   rf_param_grid,
                   logreg_param_grid,
                   knn_param_grid]

#%% 

cvResults = []
bestEstimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid = classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = 'accuracy', n_jobs = -1, verbose = 1)
    clf.fit(XTrain, yTrain)
    cvResults.append(clf.best_score_)
    bestEstimators.append(clf.best_estimator_)
    print(cvResults[i])

#%%

cv_Results = pd.DataFrame({'Cross Validation Means' : cvResults, 'ML Models' : ["DecisionTreeClassifier", "SVM","RandomForestClassifier",
             "LogisticRegression",
             "KNeighborsClassifier"]})

g = sns.barplot(x = "Cross Validation Means", y = "ML Models", data = cv_Results)
g.set_xlabel("Mean Accuracy")
plt.yticks(rotation = 45)
g.set_title("Cross Validation Scores")
plt.show()

#%% Ensemble Modeling

votingC = VotingClassifier(estimators = [("dt",bestEstimators[0]),
                                        ("rfc",bestEstimators[2]),
                                        ("lr",bestEstimators[3])],
                                        voting = "soft", n_jobs = -1)
votingC = votingC.fit(XTrain, yTrain)
print(accuracy_score(votingC.predict(XTest),yTest))

#%% Prediction and Submission Part

test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)
results = pd.concat([test_PassengerId, test_survived],axis = 1)
results.to_csv("titanic.csv", index = False)



