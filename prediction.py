import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import the dataset
Data = pd.read_csv('C:/Users/HP/Desktop/Python/Calories/train.csv')

Data.head()

#data info
Data.info()

#dropping unnecessary column [id]
Data = Data.drop(columns = ['id'])

# installing scikit-learn + encoding sex
from sklearn.preprocessing import LabelEncoder

Data['Sex'] = LabelEncoder().fit_transform(Data['Sex'])

#check for duplicates
Data.duplicated().sum()

#dropping duplicates
Data = Data.drop_duplicates()

Data.duplicated().sum()

#data visualisation
plt.figure(figsize=(10, 6))
sns.countplot(data = Data, x = 'Sex')
plt.title('Count of Male and Female')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data = Data, x = 'Age', bins = 30, kde = False)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Density')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data = Data, x = 'Height', bins = 30, kde = False)
plt.title('Distribution of Height')
plt.xlabel('Height')
plt.ylabel('Density')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data = Data, x = 'Weight', bins = 30, kde = False)
plt.title('Distribution of Weight')
plt.xlabel('Weight')
plt.ylabel('Density')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data = Data, x = 'Duration', bins = 30, kde = False)
plt.title('Distribution of Duration')
plt.xlabel('Duration')
plt.ylabel('Duration')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data = Data, x = 'Heart_Rate', bins = 30, kde = False)
plt.title('Distribution of Heart Rate')
plt.xlabel('Heart Rate')
plt.ylabel('Density')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data = Data, x = 'Body_Temp', bins = 30, kde = False)
plt.title('Distribution of Body Temperature')
plt.xlabel('Body Temperature')
plt.ylabel('Density')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data = Data, x = 'Calories', bins = 30, kde = False)
plt.title('Distribution of Calories')
plt.xlabel('Calories')
plt.ylabel('Density')
plt.show()

#preparing for splitting
x = Data.drop(columns = ['Calories'])
y = Data['Calories']

#splitting in train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

#prediction + results
y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

print(f"""Train Score : {model.score(X_train,y_train) * 100:.2f}%""")
print(f"""Test Score : {model.score(X_test,y_test) * 100:.2f}%""")

#xgboost_1
import xgboost as xgb

model_xgb = xgb.XGBRegressor()
model_xgb.fit(X_train, y_train)

#prediction + results
y_pred_xgb = model_xgb.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_xgb))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_xgb))
print("R2 Score:", r2_score(y_test, y_pred_xgb))

print(f"""Train Score : {model_xgb.score(X_train,y_train) * 100:.2f}%""")
print(f"""Test Score : {model_xgb.score(X_test,y_test) * 100:.2f}%""")

#xgboost_2
dtrain = xgb.DMatrix(X_train, y_train, enable_categorical = True)
dtest = xgb.DMatrix(X_test, y_test, enable_categorical = True)

#setting parameters
params = {
    'objective': 'reg:squarederror', # minimizing error function
    'tree_method': 'hist' # using histogram-based algorithm
}

n = 1000 # number of boosting rounds

evals = [(dtrain, 'train'), (dtest, 'validation')] # measuring model performance at each boosting round

evals_result = {} # store boosting rounds results for model evaluation

model_xgb = xgb.train(
    params = params,
    dtrain = dtrain,
    num_boost_round = n,
    evals = evals,
    verbose_eval = 100, # print evaluation metric every x (10) rounds
    early_stopping_rounds = 50, # stop after x (50) rounds if the model does not improve
    evals_result = evals_result,
)

print("Best iteration (round):", model_xgb.best_iteration)
print("Best score:", model_xgb.best_score)

metric = 'rmse'
eval_name = list(evals_result.keys())[0] # get the name of the first evaluation set

results = evals_result[eval_name][metric] # get the results for the specified metric

plt.figure(figsize=(10, 6))

rounds = range(1, len(results) + 1)  # boosting rounds, starting at 1
plt.plot(rounds, results, label='RMSE on validation set', marker='o', markersize=0)

best_round = model_xgb.best_iteration + 1  # +1 because best_iteration is 0-based
best_score = model_xgb.best_score

plt.axvline(best_round, color='red', linestyle='--', label=f'Best round: {best_round} (RMSE: {best_score:.3f})')
plt.scatter(best_round, best_score, color='red', zorder=5)

plt.title('XGBoost Training Progress')
plt.xlabel('Boosting Rounds')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.show()

#prediction + results
preds = model_xgb.predict(dtest)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f'RMSE of the base model: {rmse:.3f}')

# xgboost_2 hyperparameter tuning with HYPEROPT (scikit-learn API -> XGBClassifier)
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe # import HYPEROPT library

space = {
    'max_depth': hp.quniform('max_depth', 3, 18, 1),
    'gamma': hp.uniform('gamma', 1, 9),
    'reg_alpha': hp.quniform('reg_alpha', 0, 180, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1), # usually >0
    'n_estimators': 180,
}

from xgboost import XGBRegressor

def objective(space):
    model = XGBRegressor(
        n_estimators=int(space['n_estimators']),
        max_depth=int(space['max_depth']),
        gamma=space['gamma'],
        reg_alpha=int(space['reg_alpha']),
        min_child_weight=int(space['min_child_weight']),
        colsample_bytree=float(space['colsample_bytree']),
        random_state=0
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    # Evaluate with RMSE (or other regression metric)
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print("RMSE:", rmse)
    return {'loss': rmse, 'status': STATUS_OK}

trials = Trials()

best_hyperparams = fmin(fn=objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials
                        )

print("The best hyperparameters are : ","\n")
print(best_hyperparams)

#setting parameters
params = {
    'objective': 'reg:squarederror', # minimizing error function
    'tree_method': 'hist', # using histogram-based algorithm
    'colsample_bytree': 0.62,
    'gamma': 2.06,
    'max_depth': 8,
    'min_child_weight': 3,
    'reg_alpha': 179,
    'reg_lambda': 0.93
}

n = 1000 # number of boosting rounds

evals = [(dtrain, 'train'), (dtest, 'validation')] # measuring model performance at each boosting round

evals_result = {} # store boosting rounds results for model evaluation

model_xgb = xgb.train(
    params = params,
    dtrain = dtrain,
    num_boost_round = n,
    evals = evals,
    verbose_eval = 100, # print evaluation metric every x (10) rounds
    early_stopping_rounds = 50, # stop after x (50) rounds if the model does not improve
    evals_result = evals_result,
)

print("Best iteration (round):", model_xgb.best_iteration)
print("Best score:", model_xgb.best_score)

metric = 'rmse'
eval_name = list(evals_result.keys())[0] # get the name of the first evaluation set

results = evals_result[eval_name][metric] # get the results for the specified metric

plt.figure(figsize=(10, 6))

rounds = range(1, len(results) + 1)  # boosting rounds, starting at 1
plt.plot(rounds, results, label='RMSE on validation set', marker='o', markersize=0)

best_round = model_xgb.best_iteration + 1  # +1 because best_iteration is 0-based
best_score = model_xgb.best_score

plt.axvline(best_round, color='red', linestyle='--', label=f'Best round: {best_round} (RMSE: {best_score:.3f})')
plt.scatter(best_round, best_score, color='red', zorder=5)

plt.title('XGBoost Training Progress')
plt.xlabel('Boosting Rounds')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.show()

preds = model_xgb.predict(dtest)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f'RMSE of the base model: {rmse:.3f}')
print(f'R2 of the base model: {r2_score(y_test, preds):.3f}')