"""
Georgia Stinchfield and Natali Khalife
Oct. 2023

Training trees for the process family design surrogates formulation.
"""

from lineartree import LinearTreeRegressor
from lineartree import LinearTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import pickle
import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb

from onnxmltools.convert.lightgbm.convert import convert
import onnxmltools as onnxmltools
from skl2onnx.common.data_types import FloatTensorType

from process_family.utils.trainer.base import BaseTrainer

class TreeTrainer(BaseTrainer):
    """
    Tools for training trees (gradient boosted decision tree, linear model decision tree) for the two tasks needed in process family design.
    """

    def __init__(self, params, label):
        super().__init__(params, label)
        
    # CREDIT: the following two functions get_onnx_model, write_onnx_to_file
    # were copied directly from bo_with_trees.ipynb from the omlt repo
    def get_onnx_model(self,lgb_model):
        # export onnx model
        float_tensor_type = FloatTensorType([None, lgb_model.num_feature()])
        initial_types = [('float_input', float_tensor_type)]
        onnx_model = convert(lgb_model, 
                             initial_types=initial_types, 
                             target_opset=8)
        return onnx_model
    
    def write_onnx_to_file(self, onnx_model, path, file_name):
        from pathlib import Path
        with open(Path(path) / file_name, "wb") as onnx_file:
            onnx_file.write(onnx_model.SerializeToString())
            print(f'Onnx model written to {onnx_file.name}')
    
    def train_gbdt(self, task, directory, plot_metrics=False, hp_tune=False):
        """
        trains a gradient boosted decision tree, using the lightgbm package from Microsoft
        Note: I have issues getting lightgbm to work properly on M1 chip

        Args:
            task : str
                this should be one of two options
                    - "regression" --> trains a gbdt to predict cost
                    - "classification" --> trains a gbdt to classify
            directory : str
                directory to the location where all information will be saved.
                in particular, the model itself and any performance metrics.
            plot_metrics : bool, options
                decides if metrics for model are plotted or not
                optional, default is False
        """
        self.task=task

        if task=="regression":
            x_train=self.X_train_regression
            y_train=self.y_train_regression
            x_test=self.X_test_regression
            y_test=self.y_test_regression
            x_val=self.X_val_regression
            y_val=self.y_val_regression
            metric="mse"
            obj="regression"

        if task=="classification" or task=="classification-regression":
            x_train=self.X_train_classification
            y_train=self.y_train_classification
            x_test=self.X_test_classification
            y_test=self.y_test_classification
            x_val=self.X_val_classification
            y_val=self.y_val_classification
            if task=="classification":
                metric='binary_logloss'
                obj = "binary"
            else:
                metric = "mse"
                obj="regression"

        base_estimator = lgb.LGBMRegressor() if \
            (task == "regression" or task == "classification-regression") \
                  else lgb.LGBMClassifier()

        # convert data for tree training
        training_data=lgb.Dataset( x_train,
                                   y_train )

        # set parameters
        if hp_tune: 
        
            # set parameters
            PARAMS_grid = {'objective' : [obj],
                            'metric' :[metric],
                            'boosting_type' : ['gbdt'],
                            'num_trees': [10,50,100],
                            'max_depth': [10,50,100],
                            'min_data_in_leaf': [1,5,15,20],
                            'random_state':[1,10,50]}
            
            # Create the GridSearchCV object
            grid = GridSearchCV(estimator=base_estimator, param_grid=PARAMS_grid, cv=3)
 
            # train the model
            grid_result = grid.fit(x_train, y_train)
            gbdt_model = grid_result.best_estimator_
            train_accuracy = gbdt_model.score(x_train, y_train)

            # Evaluate the best model on the test data
            test_accuracy = gbdt_model.score(x_test, y_test)
                
        else:
             # set parameters
            PARAMS = {'objective': 'regression',
                    'metric': metric,
                    'boosting_type': 'gbdt',
                    'num_trees': 100,
                    'max_depth': 10,
                    'min_data_in_leaf': 1,
                    'random_state': 42,
                    'verbose':0,
                    'force_col_wise':True}
        
            # train the model
            gbdt_model=base_estimator.fit(params=PARAMS,
                                           train_set=training_data)
        
        # predict on testing and validation set
        if (task=="classification" or task=="classification-regression") \
            and plot_metrics==True:
            test_pred=gbdt_model.predict(x_test)
            self.plot_classification_prediction(predictions=test_pred,
                                                labels=y_test,
                                                dataset="test-set-gbdt",
                                                directory=directory)
            val_pred=gbdt_model.predict(x_val)
            self.plot_classification_prediction(predictions=val_pred,
                                                labels=y_val,
                                                dataset="val-set-gbdt",
                                                directory=directory)

        # pickle!
        with open(f"{directory}/{task}-gbdt", "wb") as f:
            pickle.dump(gbdt_model, f)
        
        return gbdt_model


    def train_lmdt(self,task,directory,hp_tune=None):
        """
        trains a linear model decision tree, using the linear-tree package

        Args:
            task : str
                this should be one of two options
                    - "regression" --> trains a gbdt to predict cost
                    - "classification" --> trains a gbdt to classify
            directory : str
                directory to the location where all information will be saved.
                in particular, the model itself and any performance metrics.
        """

        # build model
        if task=="regression":
            x_train=self.X_train_regression
            y_train=self.y_train_regression
            x_test=self.X_test_regression
            y_test=self.y_test_regression
            base_estimator = LinearRegression()
            cre= 'mse'
            lmdt=LinearTreeRegressor(base_estimator=base_estimator)

        if task=="classification":
            x_train=self.X_train_classification
            y_train=self.y_train_classification
            x_test=self.X_test_classification
            y_test=self.y_test_classification
            base_estimator=LogisticRegression()
            lmdt=LinearTreeClassifier(base_estimator=base_estimator,
                                      max_depth=20,
                                      min_samples_leaf=5,
                                      max_bins=100)
            cre= 'crossentropy'

        if task=="classification-regression":
            x_train=self.X_train_classification
            y_train=self.y_train_classification
            x_test=self.X_test_classification
            y_test=self.y_test_classification
            base_estimator =LinearRegression()
            lmdt=LinearTreeRegressor(base_estimator=base_estimator)
            cre= 'crossentropy'

        if hp_tune:
            param_grid = {
                            'criterion': [cre],
                            'max_bins': [10,11,12,13],
                            'min_samples_leaf': [1,2,3,4,5],
                            'max_depth':[11,12,13,14,15,16,17,18],
                        }
 
            # Create the GridSearchCV object
            grid = GridSearchCV(estimator=lmdt, 
                                param_grid=param_grid, 
                                cv=3, 
                                verbose=1)
 
             # train
            grid_result = grid.fit(x_train, y_train)
            best_lmdt_model = grid_result.best_estimator_
            train_accuracy = best_lmdt_model.score(x_train, y_train)
            test_accuracy = best_lmdt_model.score(x_test, y_test)

            # visualize
            print(f"Best Hyperparameters = {best_lmdt_model}")
            print(f"LMDT training accuracy = {train_accuracy}")
            print(f"LMDT testing accuracy = {test_accuracy}")

            # train final, best model for saving
            lmdt_model = best_lmdt_model.fit(x_train, y_train)
        
        else:
            lmdt_model = lmdt.fit(x_train, y_train)    
            train_accuracy = lmdt_model.score(x_train, y_train)
            test_accuracy = lmdt_model.score(x_test, y_test)

            print(f"LMDT training accuracy = {train_accuracy}")
            print(f"LMDT testing accuracy = {test_accuracy}")

        # pickle model
        with open(f"{directory}/{task}-lmdt", "wb") as f:
            pickle.dump(lmdt_model.summary(), f)
        
        return lmdt_model