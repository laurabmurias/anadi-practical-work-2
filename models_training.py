import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone

def train_linear_regression(X, y, preprocessor, kf):
    pipeline_lr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    rmse_scores, mae_scores, r2_scores = [], [], []
    for train_index, test_index in kf.split(X):
        # Suporte tanto para pandas quanto numpy
        if hasattr(X, 'iloc'):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        else:
            X_train, X_test = X[train_index], X[test_index]
        if hasattr(y, 'iloc'):
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        else:
            y_train, y_test = y[train_index], y[test_index]
        model = clone(pipeline_lr)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    # Treinar o pipeline em todos os dados para retornar o modelo ajustado
    pipeline_lr.fit(X, y)
    return rmse_scores, mae_scores, r2_scores, pipeline_lr

def train_regression_tree(X, y, preprocessor, kf, grid_flag, param_grid):
    pipeline_tree = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(random_state=42))
    ])

    if grid_flag:
        grid_search = GridSearchCV(
            pipeline_tree,
            param_grid,
            cv=kf,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X, y)
        best_tree = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        # use grid search parameters but without doing the actual search
        def get_scalar(param):
            return param[0] if isinstance(param, list) else param

        best_tree = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', DecisionTreeRegressor(
                max_depth=get_scalar(param_grid['regressor__max_depth']),
                min_samples_split=get_scalar(param_grid['regressor__min_samples_split']),
                min_samples_leaf=get_scalar(param_grid['regressor__min_samples_leaf']),
                random_state=42
            ))
        ])
        best_params = {k: get_scalar(v) for k, v in param_grid.items()}

    rmse_scores, mae_scores, r2_scores = [], [], []
    for train_index, test_index in kf.split(X):
        if hasattr(X, 'iloc'):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        else:
            X_train, X_test = X[train_index], X[test_index]
        if hasattr(y, 'iloc'):
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        else:
            y_train, y_test = y[train_index], y[test_index]
        model = clone(best_tree)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))
    return rmse_scores, mae_scores, r2_scores, best_tree, best_params

def train_svm(X, y, preprocessor, kf, grid_flag, param_grid):
    pipeline_svm = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', SVR())
    ])
    if grid_flag:
        grid_search_svm = GridSearchCV(
            pipeline_svm,
            param_grid,
            cv=kf,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search_svm.fit(X, y)
        best_params = grid_search_svm.best_params_
    else:
        # use grid search parameters but without doing the actual search
        def get_scalar(param):
            return param[0] if isinstance(param, list) else param

        best_params = {
            'regressor__kernel': get_scalar(param_grid['regressor__kernel']),
            'regressor__C': get_scalar(param_grid['regressor__C']),
            'regressor__gamma': get_scalar(param_grid['regressor__gamma']),
            'regressor__degree': get_scalar(param_grid.get('regressor__degree', 3))
        }
    
    best_svm = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', SVR(
            kernel=best_params['regressor__kernel'],
            C=best_params['regressor__C'],
            gamma=best_params['regressor__gamma'],
            degree=best_params.get('regressor__degree', 3)
        ))
    ])

    mae_svm, rmse_svm, r2_svm = [], [], []
    for train_index, test_index in kf.split(X):
        if hasattr(X, 'iloc'):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        else:
            X_train, X_test = X[train_index], X[test_index]
        if hasattr(y, 'iloc'):
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        else:
            y_train, y_test = y[train_index], y[test_index]
        model = clone(best_svm)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae_svm.append(mean_absolute_error(y_test, y_pred))
        rmse_svm.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2_svm.append(r2_score(y_test, y_pred))
    return mae_svm, rmse_svm, r2_svm, best_svm, best_params

def train_mlp(X, y, preprocessor, kf, grid_flag, param_grid):
    pipeline_mlp = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', MLPRegressor(max_iter=2000, random_state=42))
    ])
    if grid_flag:
        grid_search_mlp = GridSearchCV(
            pipeline_mlp,
            param_grid,
            cv=kf,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search_mlp.fit(X, y)
        best_params = grid_search_mlp.best_params_
    else:
        # use grid search parameters but without doing the actual search
        def get_scalar(param, default=None):
            if param is None:
                return default
            return param[0] if isinstance(param, list) else param

        best_params = {
            'regressor__hidden_layer_sizes': get_scalar(param_grid['regressor__hidden_layer_sizes']),
            'regressor__activation': get_scalar(param_grid['regressor__activation']),
            'regressor__solver': get_scalar(param_grid['regressor__solver']),
        }
        if 'regressor__learning_rate_init' in param_grid:
            best_params['regressor__learning_rate_init'] = get_scalar(param_grid['regressor__learning_rate_init'])
        else:
            best_params['regressor__learning_rate_init'] = 0.001

    best_mlp = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', MLPRegressor(
            hidden_layer_sizes=best_params['regressor__hidden_layer_sizes'],
            activation=best_params['regressor__activation'],
            solver=best_params['regressor__solver'],
            learning_rate_init=best_params['regressor__learning_rate_init'],
            max_iter=2000,
            random_state=42
        ))
    ])

    mae_mlp, rmse_mlp, r2_mlp = [], [], []
    for train_idx, test_idx in kf.split(X):
        if hasattr(X, 'iloc'):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        else:
            X_train, X_test = X[train_idx], X[test_idx]
        if hasattr(y, 'iloc'):
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            y_train, y_test = y[train_idx], y[test_idx]
        model = clone(best_mlp)
        model.fit(X_train, y_train)
        y_pred_fold = model.predict(X_test)
        mae_mlp.append(mean_absolute_error(y_test, y_pred_fold))
        rmse_mlp.append(np.sqrt(mean_squared_error(y_test, y_pred_fold)))
        r2_mlp.append(r2_score(y_test, y_pred_fold))

    return mae_mlp, rmse_mlp, r2_mlp, best_mlp, best_params
