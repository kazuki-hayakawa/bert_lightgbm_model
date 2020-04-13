import os
import uuid
import pickle
import numpy as np
import lightgbm as lgb
import optuna
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MediaClassifier():
    """ livedoorニュースコーパスの多クラス分類をするモデル """

    def __init__(self, output_dir, use_gpu=False):
        JST = timezone(timedelta(hours=+9), 'JST')
        dt_now = datetime.now(JST)
        training_date = dt_now.strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, training_date)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.device = 'gpu' if use_gpu else 'cpu'

    def train(self, features, targets):
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=0)

        def objectives(trial):
            trial_uuid = str(uuid.uuid4())
            trial.set_user_attr("trial_uuid", trial_uuid)

            # パラメータとコールバックのセッティング
            params = {
                # liverdoorニュースコーパスの媒体数は9なので9つの多クラス分類
                'objective': 'multiclass',
                'num_class': 9,
                'metric': 'multi_logloss',
                'num_leaves': trial.suggest_int("num_leaves", 10, 500),
                'feature_fraction': trial.suggest_uniform("feature_fraction", 0.0, 1.0),
                'class_weight': 'balanced',
                'device': self.device,
                'verbose': -1
            }

            pruning_callback = optuna.integration.LightGBMPruningCallback(
                trial, "multi_logloss")

            # training
            lgb_model = lgb.train(params, lgb.Dataset(X_train, y_train), num_boost_round=100,
                                  valid_sets=lgb.Dataset(X_test, y_test), callbacks=[pruning_callback])

            y_pred_train = np.argmax(lgb_model.predict(X_train), axis=1)
            y_pred_test = np.argmax(lgb_model.predict(X_test), axis=1)
            accuracy_train = accuracy_score(y_train, y_pred_train)
            accuracy_test = accuracy_score(y_test, y_pred_test)

            trial.set_user_attr("accuracy_train", accuracy_train)
            trial.set_user_attr("accuracy_test", accuracy_test)

            # モデル保存
            output_file = os.path.join(self.output_dir, f"{trial_uuid}.pkl")
            with open(output_file, "wb") as fp:
                pickle.dump(lgb_model, fp)

            return 1.0 - accuracy_test

        study = optuna.create_study()
        study.optimize(objectives, n_trials=100)

        result_df = study.trials_dataframe()
        result_csv = os.path.join(self.output_dir, "result.csv")
        result_df.to_csv(result_csv, index=False)

        return study.best_trial.user_attrs
