import os
import numpy as np

from process_family.type.surrogates import SurrogatesProcessFamily
from process_family.utils.parameters.surrogates import SurrogateParameters
from process_family.utils.trainer.nn import NNTrainer
from process_family.utils.trainer.tree import TreeTrainer

if __name__=="__main__":

    # initialize parameters
    params = SurrogateParameters("transcritical-co2")

    # init path to csv file with transcritical co2 data
    params.csv_filepath=os.path.join(params.cwd,"data/transcritical-co2-data.csv")

    # (1) factors defining each process variant (must match with .csv file names)
    params.process_variant_columns=[
                                "Capacity", 
                                "Outside Air Temperature"
                            ]

    # (2) common unit module types (must match with .csv file names)
    params.common_unit_types_column=[
                                "Evaporator Area",
                                "Condenser Area",
                                "Compressor Design Flow"
                            ]
    
    # (3) col. of .csv file that corresponds to the "success" of each simulation
    params.feasibility_column=["Success"]

    # (4) col. of .csv file that corresponds to the total annualized cost of each simulation
    params.annualized_cost_column=["Total Annualized Cost"]
    
    # (5) max. num of designs offered for each common unit module type
    #     one key per entry in common_unit_module_types
    params.num_common_unit_type_designs={
                                    "Evaporator Area":2,
                                    "Condenser Area":2,
                                    "Compressor Design Flow":2
                                }
    
    # define the labels for each of the common unit type designs allowable.
    params.labels_for_common_unit_module_designs ={
                                                "Evaporator Area": range(2),
                                                "Condenser Area": range(2),
                                                "Compressor Design Flow": range(2) 
                                            }
    
    # individual unit module design cost data
    params.unit_module_capex_columns = {
                                        "Evaporator Area": 'Annualized Capital Evaporator Cost',
                                        "Condenser Area": 'Annualized Capital Condenser Cost',
                                        "Compressor Design Flow": 'Annualized Capital Compressor Cost'
                                        }
    
    params.process_variant_column_names = ["Capacity (tons)", 
                                            "Max. Outside Air Temperature (deg. C)"]
    params.common_module_type_column_names = ["Evap. Area ($m^2$)",
                                            "Cond. Area ($m^2$)",
                                            "Compr. Flow (mol./s)"]
    params.make_results_dir("surrogates")

     # grab paths to surrogates
    cwd = os.getcwd()
    classification_surrogate = os.path.join(cwd, f"surrogates/classification-regression-gbdt")
    regression_surrogate = os.path.join(cwd, f"surrogates/regression-gbdt")

    # add surrogates we want
    params.add_surrogates(classification_type = "gbdt",
                        classification_path = classification_surrogate,
                        regression_type = "gbdt",
                        regression_path = regression_surrogate,
                        classification_threshold = 0.99)

    # add in the output scaling
    params.regression_scaling["offset_outputs"] = np.array([0], dtype="float32")
    params.regression_scaling["factor_outputs"] = np.array([1], dtype="float32")
    
    # create the surrogates based process family
    spfd = SurrogatesProcessFamily(params)

    # build & solve surrogates formulation
    spfd.build_model(params.labels_for_common_unit_module_designs)
    spfd.solve_model()

    # plot results, store in plot_pathstring
    plot_pathstring = os.path.join(params.results_dir, "surrogates-results.png")
    spfd.plot(directory=plot_pathstring,
              process_variant_column_names = params.process_variant_column_names,
              common_module_type_columns = params.common_module_type_column_names)

    results_pathstring=os.path.join(params.results_dir,"surrogates-results.txt")
    spfd.results_summary(directory=results_pathstring)
