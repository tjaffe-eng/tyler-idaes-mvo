import onnx
import keras
import pickle
import lineartree

from onnxmltools.convert.lightgbm.convert import convert
import onnxmltools as onnxmltools
from skl2onnx.common.data_types import FloatTensorType
import lightgbm
import tensorflow as tf

from process_family.utils.parameters.base import Parameters
from process_family.utils.trainer.base import BaseTrainer

class SurrogateParameters(Parameters):

    classification_type_kwargs = [
        "lmdt",
        "gbdt",
        "nn"
    ]

    classification_types = [
        # keras.engine.sequential.Sequential,
        tf.keras.models.Sequential,
        onnx.onnx_ml_pb2.ModelProto,
        lineartree.lineartree.LinearTreeRegressor,
        lineartree.lineartree.LinearTreeClassifier,
        dict
    ]

    regression_type_kwargs = [
        "lmdt",
        "gbdt",
        "nn"
    ]

    regression_types = [
        # keras.engine.sequential.Sequential,
        tf.keras.models.Sequential,
        onnx.onnx_ml_pb2.ModelProto,
        lineartree.lineartree.LinearTreeRegressor,
        lightgbm.sklearn.LGBMRegressor
    ]

    def __init__(self, system_name):
        """
        returns the case study specific parameters for instantiating the ProcessFamily objects.

        Args: 
            system_name : str
                name of a the system, which must be in the "system_names" list.

        Attrs:
            csv_filepath : str
                location of csv file containing the data
            process_variant_columns : list of str
                list of column names corresponding to the variables that define the boundary conditions for the 
                process variants
            common_unit_types_column : list of str
                list of column names coresponding to the common modules in the process platform.
            feasibility_column: str
                column name corresponding to True/False feasibility data
            annualized_cost_column : str
                column name corresponding to the total annualized cost of each boundary condition & unit
            num_common_unit_type_designs : dict
                the keys will correspond (*exactly) to each of the elements in the common_unit_types_column
                each corresponding entry will be an int, representing the max. num of designs allowed for that unit module type
            labels_for_common_unit_module_designs : dict
                Each of the keys will be all c \in C (i.e. each common unit module type, c)
                Each corresponding element is a list of labels, to identify the selected common unit module designs.
            """
        super().__init__(system_name)
    
    def add_surrogates(self, classification_type, classification_path,
                        regression_type, regression_path, classification_threshold = 0.5):
        """
        adds a surrogate to the object

        Args:
            classification_type : str
                kwd string corresponding to one of the classification_surrogates 
                options.
            classification_path : str
                path to the specified classification surrogate
            regression_type : str
                kwd string corresponding to one of the regression_surrogates 
                options.
            regression_path : str
                path to the specified regression surrogate
            classification_threshold : float, optional  
                value to constrain above to be considered feasible
                optional, default is 0.5
        """
        self.classification_threshold = classification_threshold

        if classification_type not in self.classification_type_kwargs:
            supported = ", ".join(self.classification_type_kwargs)
            raise ValueError(f"Classification surrogate type '{classification_type}' is not supported. Supported types: {supported}")
        if regression_type not in self.regression_type_kwargs:
            supported = ", ".join(self.regression_type_kwargs)
            raise ValueError(f"Regression surrogate type '{regression_type}' is not supported. Supported types: {supported}")
        
        # load and add surrogates
        self.classification_surrogate = self._load_model(type = classification_type,
                                                         path = classification_path)
        self.regression_surrogate = self._load_model(type = regression_type,
                                                     path = regression_path)
        
        # sanity check
        self._check_loaded_models()

        # add scaling
        self._get_scaling_info()
    
    def _load_model(self, type, path):
        """
        returns the loaded model, depending on the ML type indicated.
        """

        if type=="gbdt":
            try:
                return onnx.load(path)
            except Exception:
                pass
            try:
                # unpickle
                with open(path, "rb") as file:
                    unpickled_gbdt = pickle.load(file)

                # convert to onnx model
                print(dir(unpickled_gbdt))

                float_tensor_type = FloatTensorType([None, unpickled_gbdt.n_features_])
                initial_types = [('float_input', float_tensor_type)]
                onnx_model = convert(unpickled_gbdt,
                                     initial_types=initial_types,
                                     target_opset=8)
                return onnx_model
            except Exception:
                pass
            try:
                # unpickle
                with open(path, "rb") as file:
                    unpickled_gbdt = pickle.load(file)

                float_tensor_type = FloatTensorType([None, unpickled_gbdt.n_features_])
                initial_types = [('float_input', float_tensor_type)]
                onnx_model = onnxmltools.convert_lightgbm(unpickled_gbdt,
                                                          initial_types=initial_types)
                return onnx_model
            except Exception:
                raise Exception("Could not load GBDT as an ONNX model, LightGBMRegressor, or unpickling.\nPlease check file format and try again.")

        if type=="lmdt":

            # if the path is really a path, load
            if isinstance(path, str):
                try:
                    model = onnx.load(path)
                    print(f"loading via ONNX; {model = }")
                    return model
                except Exception:
                    pass
                try:
                    with open(path, "rb") as file:
                        print("loading via pickle")
                        model = pickle.load(file)
                        return model
                except Exception:
                    raise Exception("Could not load LMDT as an ONNX model or unpickling.\nPlease check file format and try again.")

            # otherwise, it should already be a model and we can return directly
            else:
                model = path
                return model

        if type=="nn":
            try:
                return onnx.load(path)
            except Exception:
                pass
            try:
                return keras.models.load_model(path)
            except Exception:
                raise Exception("Could not load NN as an ONNX model or via Keras.\nPlease check file format and try again.")
    
    def _check_loaded_models(self):
        """
        Checks that the loaded models have the proper instance types.
        These are used in building the MILP, to sense which formulation to use.
        """
        
        if not any(isinstance(self.classification_surrogate, t) for t in self.classification_types):
            viable = ", ".join(str(t) for t in self.classification_types)
            raise TypeError(f"Classification surrogate type {type(self.classification_surrogate)} not supported. Viable types: {viable}")

        if not any(isinstance(self.regression_surrogate, t) for t in self.regression_types):
            viable = ", ".join(str(t) for t in self.regression_types)
            raise TypeError(f"Regression surrogate type {type(self.regression_surrogate)} not supported. Viable types: {viable}")

    def _get_scaling_info(self): 
        """
        get the scaling information for OMLT
        """
        dummy_trainer = BaseTrainer(self, label="dummy")
        self.classification_scaling = dummy_trainer.classification_scaling
        self.regression_scaling = dummy_trainer.regression_scaling
    
    def linearize_logistic(self, display=False):
        """
        We can train a classification NN that has a sigmoidal activation.
        To solve a MILP rather than an MINLP, we need to replace this sigmoidal activation
        with a linear, and set the threshold for classification to zero.

        Args:
            display : bool, optional
                Displays the adjusted keras model overview.
                Optional, default is false.
        """
        if not hasattr(self, "classification_surrogate") or \
                not isinstance(self.classification_surrogate, tf.keras.models.Sequential):
            raise TypeError("linearize_logistic() requires an instantiated classification surrogate of type NN (tf.keras.models.Sequential) with a sigmoid activation.")

        # change sigmoid output layer activation to linear
        self.classification_surrogate.layers[-1].activation = keras.activations.linear

        # display model for santity check
        if display:
            print("SANITY CHECK: Activation of the final layer in the classification NN:", self.classification_surrogate.layers[-1].get_config()["activation"])