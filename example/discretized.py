import os

from process_family.type.discretized import DiscretizedProcessFamily
from process_family.utils.parameters.base import Parameters

if __name__=="__main__":

    # initialize parameters
    params = Parameters("transcritical-co2")

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
    
    params.process_variant_column_names = ["Capacity (tons)", 
                                            "Max. Outside Air Temperature (deg. C)"]
    params.common_module_type_column_names = ["Evap. Area ($m^2$)",
                                            "Cond. Area ($m^2$)",
                                            "Compr. Flow (mol./s)"]
    params.make_results_dir("discretized")

    # initializes all sets
    dpfd = DiscretizedProcessFamily(params)
    
    # build & solve model
    dpfd.build_model()
    dpfd.solve_model()
    
    # plot results
    dpfd.plot(directory = os.path.join(params.results_dir, "discretized-results.png"),
              process_variant_column_names = params.process_variant_column_names,
              common_module_type_columns = params.common_module_type_column_names)

    # create results summary, store in results_pathstring
    dpfd.results_summary(directory = \
                         os.path.join(params.results_dir, "discretized-results.txt"))
