"""Frozen-hyperparameter XGB ensemble (thesis train-only choices; no retuning)."""
import numpy as np

from config import (N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE,
                    COLSAMPLE, MOM_FEATURES)

FEATURES_PI = MOM_FEATURES + ['pi']
FEATURES_NOPI = list(MOM_FEATURES)


def ensemble_scores(X_tr, y_tr, X_te, seeds, n_jobs=4):
    from xgboost import XGBRegressor
    preds = np.zeros(len(X_te))
    for s in seeds:
        m = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                         learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                         colsample_bytree=COLSAMPLE, tree_method='hist',
                         random_state=s, verbosity=0, n_jobs=n_jobs)
        m.fit(X_tr, y_tr)
        preds += m.predict(X_te)
    return preds / len(seeds)
