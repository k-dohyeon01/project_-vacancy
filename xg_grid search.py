import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import product
from tqdm import tqdm

train_data = pd.read_csv('C:/Users/User/OneDrive/바탕 화면/도전학기/분석시작/R분석_train.csv', na_values=',')
test_data = pd.read_csv('C:/Users/User/OneDrive/바탕 화면/도전학기/분석시작/R분석_test.csv', na_values=',')

train_data['y'] = train_data['y2'] / train_data['y1']
test_data['y'] = test_data['y2'] / test_data['y1']

def evaluate_model(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    return mae, rmse, r2

feature_sets = {
    "방법 1: 상관관계 + VIF": ['x2_3','x4_4','x4_5','x1_3','x4_2','x5_6'],
    "방법 2: 상관관계 + 단계적 선택": ['x2_1','x2_3','x1_4'],
    "방법 3: 상관관계 + VIF": ['x1_2','x1_4','x4_2','x4_4','x4_5','x5_6'],
    "방법 4: 전체 변수": [
        'x4_1','x4_2','x4_5','x4_7','x3_2','x3_3','x3_4','x3_5','x3_6','x3_7','x5_5',
        'x4_3','x2_1','x3_1','x2_4','x2_5','x2_6','x4_4','x5_6','x3_8','x5_1','x2_2',
        'x2_3','x1_1','x1_2','x1_3','x1_4','x1_5','x5_2','x5_3','x5_4','x4_6'
    ]
}

param_grid = {
    'reg_lambda': [0.5, 0.7, 1, 1.5, 2, 5, 10],
    'reg_alpha': [0, 0.1, 0.5, 1, 2],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'eta': [0.01, 0.05, 0.1, 0.2, 0.3],
    'gamma': [0, 0.1, 0.2],
    'grow_policy': ['depthwise', 'lossguide']
}

param_combinations = list(product(
    param_grid['reg_lambda'],
    param_grid['reg_alpha'],
    param_grid['max_depth'],
    param_grid['eta'],
    param_grid['gamma'],
    param_grid['grow_policy']
))

final_results = []
results_for_plot = []

for method_name, features in tqdm(feature_sets.items(), desc="🔁 Feature Sets"):
    print(f"\nProcessing feature set: {method_name}")
    
    X_train = train_data[features]
    y_train = train_data['y']
    X_test = test_data[features]
    y_test = test_data['y']

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    best_score = float('inf')
    best_result = None
    best_params = None

    for reg_lambda, reg_alpha, max_depth, eta, gamma, grow_policy in tqdm(param_combinations, desc=f"🔍 Grid Search ({method_name})", leave=False):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'reg_lambda': reg_lambda,
            'reg_alpha': reg_alpha,
            'max_depth': max_depth,
            'eta': eta,
            'gamma': gamma,
            'grow_policy': grow_policy
        }

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=500,
            evals=[(dtrain, 'train'), (dtest, 'eval')],
            early_stopping_rounds=10,
            verbose_eval=False
        )

        pred = model.predict(dtest)
        mae, rmse, r2 = evaluate_model(y_test, pred)

        if rmse < best_score:
            best_score = rmse
            best_result = (mae, rmse, r2)
            best_params = (reg_lambda, reg_alpha, max_depth, eta, gamma, grow_policy)

    final_results.append((method_name, *best_result, *best_params))
    results_for_plot.append({
        'method': method_name,
        'reg_lambda': best_params[0],
        'reg_alpha': best_params[1],
        'max_depth': best_params[2],
        'eta': best_params[3],
        'gamma': best_params[4],
        'grow_policy': best_params[5],
        'mae': best_result[0],
        'rmse': best_result[1],
        'r2': best_result[2]
    })

for result in final_results:
    method_name, mae, rmse, r2, reg_lambda, reg_alpha, max_depth, eta, gamma, grow_policy = result
    print(f"{method_name}")
    print(f"Best Params -> reg_lambda: {reg_lambda}, reg_alpha: {reg_alpha}, max_depth: {max_depth}, eta: {eta}, gamma: {gamma}, grow_policy: {grow_policy}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}\n")
