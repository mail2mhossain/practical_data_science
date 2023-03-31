from clearml import Task, Logger
import argparse
import optuna
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yaml
from typing import Text, Dict
import sys
sys.path.append('.')

from src.utils.logs import get_logger

def tune_xgb_with_optuna(config_path: Text, task: Task) -> None:
    ml_logger = task.get_logger()
    task.connect_configuration(config_path)

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('XGB OPTUNA', log_level=config['base']['log_level'])

    random_state = config['base']['random_state']
    estimator_tuner = config['train']['estimator_tuner']
    logger.info('Estimator Tuner Name - ' + estimator_tuner)
    task.add_tags(estimator_tuner)
    target = config['featurize']['target_column']
    best_rmse = config['train']['best_rmse']
    best_rmse = round(best_rmse, 4)
    best_r2_score = config['train']['best_r2_score']
    best_r2_score = round(best_r2_score, 2)


    logger.info('Load train dataset')
    train_df = pd.read_csv(config['data_split']['trainset_path'])
    X_train = train_df.drop(target , axis=1)
    y_train = train_df[target]

    logger.info('Load test dataset')
    test_df = pd.read_csv(config['data_split']['testset_path'])
    X_test = test_df.drop(target , axis=1)
    y_test = test_df[target]

    logger.info('Load validation dataset')
    validation_df = pd.read_csv(config['data_split']['validationset_path'])
    X_validation = validation_df.drop(target , axis=1)
    y_validation = validation_df[target]

    param_grid = config['train']['estimators'][estimator_tuner]['param_grid']
    early_stopping_rounds = config['train']['estimators'][estimator_tuner]['early_stopping_rounds']
    n_trials = config['train']['estimators'][estimator_tuner]['n_trials']

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', param_grid['n_estimators'][0], param_grid['n_estimators'][1], param_grid['n_estimators'][2]),
            'max_depth': trial.suggest_int('max_depth', param_grid['max_depth'][0], param_grid['max_depth'][1]),
            'learning_rate': trial.suggest_float('learning_rate', param_grid['learning_rate'][0], param_grid['learning_rate'][1]),
            'subsample': trial.suggest_uniform('subsample',param_grid['subsample'][0], param_grid['subsample'][1]),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', param_grid['colsample_bytree'][0], param_grid['colsample_bytree'][1]),
            'gamma': trial.suggest_float('gamma', param_grid['gamma'][0], param_grid['gamma'][1])
        }

        # Train and evaluate the model
        model = XGBRegressor(objective='reg:squarederror', 
                            eval_metric='rmse',
                            booster='gbtree', 
                            early_stopping_rounds=early_stopping_rounds,
                            random_state=random_state,
                            n_jobs=-1, 
                            **params)
        model.fit(X_train, y_train,
                eval_set=[(X_validation, y_validation)])
        
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        return rmse

    logger.info('Create the study object and optimize the objective function')
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials)

    logger.info("Best trial:")
    xgb_trial = study.best_trial
    params = xgb_trial.params
    task.connect(params)

    model = XGBRegressor(objective='reg:squarederror', 
                        eval_metric='rmse',
                        booster='gbtree', 
                        early_stopping_rounds=early_stopping_rounds,
                        random_state=random_state,
                        n_jobs=-1,
                        **params)

    model.fit(X_train, y_train,          
            eval_set=[(X_validation, y_validation)])

    preds = model.predict(X_test)
    new_rmse = np.sqrt(mean_squared_error(y_test, preds))
    new_rmse = round(new_rmse, 4)
    new_r2_score = r2_score( y_test, preds)
    new_r2_score = round(new_r2_score, 2)

    logger.info("  Trial RMSE: {}".format(xgb_trial.value))
    logger.info("  Params: {}".format(params))
    logger.info("  RMSE: {}".format(new_rmse))
    logger.info("  R2 Score: {}".format(new_r2_score))

    logger.info('Report Single Scalars - RMSE & R2 Score')
    ml_logger.report_single_value(name="RMSE", value=new_rmse)
    ml_logger.report_single_value(name="R2 Score", value=new_r2_score)

    if best_rmse > new_rmse and best_r2_score < new_r2_score:
        config['train']['best_rmse'] = new_rmse
        config['train']['best_r2_score'] = new_r2_score

        config['train']['xgboost']['param_grid']['n_estimators'] = params['n_estimators']
        config['train']['xgboost']['param_grid']['max_depth'] = params['max_depth']
        config['train']['xgboost']['param_grid']['learning_rate'] = params['learning_rate']
        config['train']['xgboost']['param_grid']['subsample'] = params['subsample']
        config['train']['xgboost']['param_grid']['colsample_bytree'] = params['colsample_bytree']
        config['train']['xgboost']['param_grid']['gamma'] = params['gamma']

        config['train']['xgboost']['early_stopping_rounds'] = early_stopping_rounds

        config['train']['estimator_name'] = 'xgboost'

        with open(config_path, 'w') as file:
            yaml.dump(config, file)

        logger.info('Save model')
        model.save_model(config['train']['estimators'][estimator_tuner]['model_path'])

    ml_logger.flush()
    logger.info('Task Completed.')

if __name__ == '__main__':
    task = Task.init(project_name='evi_experiments', task_name='EVI XGB Experiment')
    
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    tune_xgb_with_optuna(config_path=args.config, task=task)