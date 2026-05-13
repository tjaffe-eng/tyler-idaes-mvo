import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from process_family.type.base import ProcessFamilyBase

class BaseTrainer:
    def __init__(self, params, label):
        """
        Initializes the base information for training ML surrogates for process family design.
        Organizes and splits the training, testing, and validation data.

        Args:
            data_path : str
                location of data for training.
            process_variant_columns : list of str
                list of column names corresponding to the variables that define the boundary conditions for the 
                process variants
            common_unit_types_column : list of str
                list of column names coresponding to the common modules in the process platform.
            feasibility_column: str
                column name corresponding to True/False feasibility data
            annualized_cost_column : str
                column name corresponding to the total annualized cost of each boundary condition & unit
            label : str
                label for this trainer.
        
        """

        # build the process family class obj. with given information
        process_family = ProcessFamilyBase(params)
        self.process_family = process_family
        self.process_variant_columns = params.process_variant_columns
        self.common_unit_type_columns = params.common_unit_types_column
        self.label=label

        """ 
        CLASSIFICATION TASK 
            surrogate task: predict classification of feasible / infeasible simulation
            given process variant conditions & common unit type designs.
        """

        # get labels & data
        self.y_classification=np.asarray(self.process_family.success_data).astype("float64") # [=] (N x 1)
        self.X_classification=np.hstack((self.process_family.variant_data, self.process_family.common_unit_module_data)) # [=] (N x |V|+|C|)
        self.X_classification=np.asarray(self.X_classification).astype("float64")

        # ensure correct dimensions before preceeding
        assert len(self.y_classification)==len(self.X_classification)

        # normalize (in this case, only X data)
        self.scale_x=MinMaxScaler()
        self.scaled_X_classification=np.asarray(self.scale_x.fit_transform(self.X_classification)).astype("float64")

        # train/test split
        self.X_train_classification, self.X_test_classification, \
            self.y_train_classification, self.y_test_classification = train_test_split(self.scaled_X_classification, 
                                                                                       self.y_classification,
                                                                                       test_size=0.1, 
                                                                                       random_state=42)
        # ensure correct dimensions before preceeding
        assert len(self.y_train_classification)==len(self.X_train_classification)
        assert len(self.y_test_classification)==len(self.X_test_classification)

        # validation split
        self.X_train_classification, self.X_val_classification, \
            self.y_train_classification, self.y_val_classification = train_test_split(self.X_train_classification, 
                                                                                       self.y_train_classification,
                                                                                       test_size=0.11, 
                                                                                       random_state=42)
        # ensure correct dimensions before preceeding
        assert len(self.y_train_classification)==len(self.X_train_classification)
        assert len(self.y_val_classification)==len(self.X_val_classification)

        """ 
        REGRESSION TASK 
            surrogate task: predict total annualized OPEX+CAPEX
            given process variant conditions & common unit type designs.
        """

        # here, we only want data w/ successful simulations.
        self.X_regression=[]
        self.y_regression=[]
        for x in range(len(self.y_classification)):
            if self.y_classification[x]==1:
                self.X_regression.append(self.X_classification[x])
                self.y_regression.append(self.process_family.cost_data[x])
        self.X_regression=np.asarray(self.X_regression).astype("float32")
        self.y_regression=np.asarray(self.y_regression).astype("float32")

        # ensure they are the same num. of rows
        assert len(self.y_regression)==len(self.X_regression)

        # normalize
        self.scaled_X_regression=np.asarray(self.scale_x.fit_transform(self.X_regression)).astype("float64")
        self.scale_y=MinMaxScaler()
        self.scaled_y_regression=np.asarray(self.scale_y.fit_transform(self.y_regression)).astype("float64")

        # ensure they are the same
        assert len(self.scaled_y_regression)==len(self.scaled_X_regression)
        
        # train/test split
        self.X_train_regression, self.X_test_regression, \
            self.y_train_regression, self.y_test_regression = train_test_split(self.scaled_X_regression,
                                                                            #    self.y_regression,
                                                                               self.scaled_y_regression,
                                                                               test_size=0.1,
                                                                               random_state=17)
        
        # validation split
        self.X_train_regression, self.X_val_regression, \
            self.y_train_regression, self.y_val_regression = train_test_split(self.X_train_regression,
                                                                            #   self.y_regression,
                                                                              self.y_train_regression,
                                                                              test_size=0.11,
                                                                              random_state=17)

        self._add_omlt_scaling_info()

    def _add_omlt_scaling_info(self):
        """
        this is where all the omlt scaling block information is added.
        We need this later on for the optimization.
        Attaches the classification_scaling and regression_scaling to the current object.
        """
        
        # num of inputs = num of process variant descriptors + num of common unit module types
        num_input_layers=len(self.process_variant_columns)+len(self.common_unit_type_columns)

        self.classification_scaling={"scaled_input_bounds": {input_layer: (0, 1.01) for input_layer in range(num_input_layers)},
                                     "offset_inputs": self.scale_x.data_min_,
                                     "factor_inputs": self.scale_x.data_range_,
                                     "offset_outputs": [0],
                                     "factor_outputs": [1]}
        self.regression_scaling={"scaled_input_bounds": {input_layer: (0, 1.01) for input_layer in range(num_input_layers)},
                                "offset_inputs": self.scale_x.data_min_,
                                "factor_inputs": self.scale_x.data_range_,
                                "offset_outputs": self.scale_y.data_min_,
                                "factor_outputs": self.scale_y.data_max_ }
    
    def plot_classification_prediction(self, predictions, labels, dataset, directory):
        """
        plots the predictions

        Args:
            predictions : list
                list of predicted values from the model.
            labels : list
                list of actual labels from the dataset.
            dataset : str
                type of dataset we are working with (i.e., val-set, test-set, train-set)
            directory : str
                directory to the location where all information will be saved.
                in particular, the model itself and any performance metrics.
        """

        # separate predictions based on successful/unsuccesful
        successful_simulation=[]
        unsuccessful_simulation=[]
        for i in range(len(predictions)):
            if labels[i] == 1:
                successful_simulation.append(float(predictions[i]))
            elif labels[i] == 0:
                unsuccessful_simulation.append(float(predictions[i]))

        # plot a histogram, colored based on successful / unsuccessful
        fig_hist = plt.figure()
        ax_hist = fig_hist.add_subplot()

        ax_hist.hist(successful_simulation, color = 'green', label = 'successful')
        ax_hist.hist(unsuccessful_simulation, color = 'black', label = 'unsuccessful')

        fig_hist.legend(bbox_to_anchor=(1.29, 0.55))

        ax_hist.set_xlabel('Surrogate Output Values')
        ax_hist.set_title('Binned Output Values of Classification Surrogate\n'+self.label)

        fig_hist.savefig(f"{directory}/{self.task}-histogram.png",
                         bbox_inches="tight",
                         dpi=300)