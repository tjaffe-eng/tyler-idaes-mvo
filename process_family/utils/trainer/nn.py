from typing import Union

import keras_tuner
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import CSVLogger, EarlyStopping
from keras.layers import Dense
from keras.models import Sequential

from process_family.utils.trainer.base import BaseTrainer

tf.random.set_seed(seed=42)
tf.keras.backend.set_floatx("float64")


class NNTrainer(BaseTrainer):

    tasks = ["regression", "linear-classification", "logistic-classification"]

    """
    Tools for training neural networks for the two tasks needed in process family design.
    """

    def __init__(self, params, label, fit_epoch: Union[int, None] = 5000):
        """
        Args:
            fit_epoch: int | None
                Number of epochs to fit the model for.
        """
        self.fit_epoch = fit_epoch

        super().__init__(params, label)

    def _build_nn(self, hp=None):
        """
        Builds the neural network with the necessary dimensions / hp tuning metrics

        Args:
            hp : type
                If None, this indicates we are building a regular NN.
                If !None, this function is used by built in Keras functionality to perform
                hyperparameter tuning.
        """

        # num of inputs = num of process variant descriptors + num of common unit module types
        input_layer = len(self.process_variant_columns) + len(
            self.common_unit_type_columns
        )

        # set final layer activation, loss, and metrics based on the activation type
        if self.task == "regression":
            final_layer_activation = "linear"
            loss = "mean_squared_error"
            metrics = ["mean_squared_error", "mae"]
        if self.task == "linear-classification":
            final_layer_activation = "linear"
            loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=True, label_smoothing=0.1
            )
            metrics = ["accuracy"]
        if self.task == "logistic-classification":
            final_layer_activation = "sigmoid"
            loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=True, label_smoothing=0.1
            )
            metrics = ["accuracy"]

        # create model
        model = Sequential()

        if hp != None:
            # add 1,2 layers with 5-20 nodes per each layer (this is left up to the hyperparameter tuning)
            for i in range(hp.Int("num_layers", 1, 2, 3)):
                model.add(
                    Dense(
                        units=hp.Int(f"units_{i}", min_value=5, max_value=30),
                        input_dim=input_layer,
                        activation="relu",
                    )
                )
        else:
            model.add(Dense(15, input_dim=input_layer, activation="relu"))
            # model.add(Dense(10,
            #                 input_dim=input_layer,
            #                 activation='relu'))

        model.add(Dense(1, activation=final_layer_activation))

        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(), metrics=metrics)

        return model

    def train_nn(self, directory, task, plot_metrics=False, hp_tuning=False):
        """
        Trains a neural network for the task at hand.
        Saves network to directory provided.

        Args:
            directory : str
                directory to the location where all information will be saved.
                in particular, the model itself and any performance metrics.
            task : str
                this should be one of three options
                    - "regression" --> trains a relu/linear activated network to predict cost
                    - "linear-classification" --> trains a relu/linear activated network to classify
                    - "logistic-classification" --> trains a relu/sigmoid activated network to classify
            plot_metrics : bool, optional
                indicates if, after training, metrics of interest will be plotted and saved to directory
                optional, default False
            hp_tuning : bool, optional
                indicates if the user would like a HP search done first for the NN
                optional, default False
        """

        # add the task to the object (this is overwritten every time this is called)
        assert task in self.tasks
        self.task = task

        # get the necessary datasets, based on if we want to perform regression or classification
        if self.task == "regression":
            x_train = self.X_train_regression
            y_train = self.y_train_regression
            x_test = self.X_test_regression
            y_test = self.y_test_regression
            x_val = self.X_val_regression
            y_val = self.y_val_regression
        else:
            x_train = self.X_train_classification
            y_train = self.y_train_classification
            x_test = self.X_test_classification
            y_test = self.y_test_classification
            x_val = self.X_val_classification
            y_val = self.y_val_classification

        # add early stopping
        if self.task == "regression":
            early_stopping = EarlyStopping(
                monitor="mean_squared_error", mode="max", verbose=0, patience=100
            )
        else:
            early_stopping = EarlyStopping(
                monitor="accuracy", mode="max", verbose=0, patience=100
            )

        # if we need to tune hyperparameters of the model
        if hp_tuning:

            # init hyperparameter tuner
            if self.task == "regression":
                tuner = keras_tuner.Hyperband(
                    hypermodel=self._build_nn,
                    objective="mean_squared_error",
                    max_epochs=self.fit_epoch,
                    hyperband_iterations=5,
                    seed=42,
                    directory=directory,
                    project_name=f"{task}-hp-tuning",
                    overwrite=True,
                )
            else:
                tuner = keras_tuner.Hyperband(
                    hypermodel=self._build_nn,
                    objective="accuracy",
                    max_epochs=self.fit_epoch,
                    hyperband_iterations=5,
                    seed=42,
                    directory=directory,
                    project_name=f"{task}-hp-tuning",
                    overwrite=True,
                )

            # tune & search
            tuner.search(
                x=x_train,
                y=y_train,
                validation_data=(x_test, y_test),
                epochs=100,
                callbacks=[early_stopping],
            )

            # get the best hyperparameters, build model
            best_hp = tuner.get_best_hyperparameters()[0]
            model = tuner.hypermodel.build(best_hp)

        # if we don't want to tune hyperparameters, just train normally
        else:
            model = self._build_nn()
        self.model = model
        # log the history
        csv_logger_path = directory + f"/{task}-nn-training.log"
        csv_logger = CSVLogger(csv_logger_path, separator=",", append=False)

        # change the early_stopping criteria to longer patience
        early_stopping = EarlyStopping(
            monitor="val_loss", mode="min", verbose=0, patience=50
        )

        # finally, train model using best hyperparameters
        history = model.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_test, y_test),
            epochs=self.fit_epoch,
            callbacks=[early_stopping, csv_logger],
        )

        # save model
        model.save(directory + f"/{task}-nn.keras")

        if plot_metrics:

            # evaluate the models
            test_pred = model.predict(x_test)
            val_pred = model.predict(x_val)

            if task == "regression":
                # plot loss
                self.plot_loss(
                    loss=history.history["loss"],
                    val_loss=history.history["val_loss"],
                    png_pathstring=directory + "/regression-nn-loss.png",
                )

                # plot mean square error
                self.plot_mse(
                    mse=history.history["mean_squared_error"],
                    png_pathstring=directory + "/regression-nn-mse.png",
                )

                # plot mean absolute error
                self.plot_mae(
                    mae=history.history["mae"],
                    png_pathstring=directory + "/regression-nn-mae.png",
                )
            else:
                self.plot_accuracy(
                    accuracy=history.history["accuracy"],
                    png_pathstring=f"{directory}/{task}-nn-accuracy.png",
                )
                self.plot_classification_prediction(
                    predictions=test_pred,
                    labels=y_test,
                    dataset="test-set",
                    directory=directory,
                )
                self.plot_classification_prediction(
                    predictions=val_pred,
                    labels=y_val,
                    dataset="val-set",
                    directory=directory,
                )

    def plot_accuracy(self, accuracy, png_pathstring):
        """
        plots the accuracy, from the Keras history object

        Args:
            accuracy : list
                list of percent accuracies on the testing data over epochs of training.
                accessed via the keras history object returned from model.fit
            png_pathstring : str
                pathstring location (+.png) for the plot to be saved.
        """

        fig = plt.figure()
        ax = fig.add_subplot()

        ax.plot(accuracy, label="train")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(True)
        fig.legend()

        ax.set_title("Accuracy vs. Epoch\n" + self.label)
        fig.savefig(png_pathstring, bbox_inches="tight", dpi=300)

    def plot_loss(self, loss, val_loss, png_pathstring):
        """
        plots the accuracy, from the Keras history object

        Args:
            loss : list
                list of training loss over epochs of training.
                accessed via the keras history object returned from model.fit
            val_loss : list
                list of validation loss over epochs of training.
                accessed via the keras history object returned from model.fit
            png_pathstring : str
                pathstring location (+.png) for the plot to be saved.
        """

        fig = plt.figure()
        ax = fig.add_subplot()

        ax.plot(loss, label="train")
        ax.plot(val_loss, label="test")
        ax.set_yscale("log")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("log(loss)")
        ax.grid(True)
        fig.legend()

        ax.set_title("Loss vs. Epoch\n" + self.label)
        fig.savefig(png_pathstring, bbox_inches="tight", dpi=300)

    def plot_mse(self, mse, png_pathstring):
        """
        plots the mean square error, from the Keras history object

        Args:
            mse : list
                list of mean square error over epochs of training.
                accessed via the keras history object returned from model.fit
            png_pathstring : str
                pathstring location (+.png) for the plot to be saved.
        """

        fig = plt.figure()
        ax = fig.add_subplot()

        ax.plot(mse, label="train")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Square Error")
        ax.grid(True)
        fig.legend()

        ax.set_title("MSE vs. Epoch\n" + self.label)
        fig.savefig(png_pathstring, bbox_inches="tight", dpi=300)

    def plot_mae(self, mae, png_pathstring):
        """
        plots the mean absolute error, from the Keras history object

        Args:
            mae : list
                list of mean absolute error over epochs of training.
                accessed via the keras history object returned from model.fit
            png_pathstring : str
                pathstring location (+.png) for the plot to be saved.
        """

        fig = plt.figure()
        ax = fig.add_subplot()

        ax.plot(mae, label="train")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Absolute Error")
        ax.grid(True)
        fig.legend()

        ax.set_title("MAE vs. Epoch\n" + self.label)
        fig.savefig(png_pathstring, bbox_inches="tight", dpi=300)
