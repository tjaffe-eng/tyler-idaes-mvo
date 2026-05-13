"""
Creates a Parameter object, that holds all of the information we need to build any of the 
process family design optimization formulations.

Note, this was designed to have all attributes added in any order.
Some more logical checks are performed when a process_family.type object is built.
"""

import os

class Parameters():

    # acceptable optimization methods available
    methods = [
                "discretized", 
                "surrogates", 
            ]
    
    def __init__(self, system_name):
        """
        returns the case study specific parameters for instantiating the ProcessFamily objects.

        Args: 
            system_name : str
                name of a the system

        Attrs:
            cwd : str
                path to the current working directory
            system_name : str
                name of the system
            """
        self.system_name = system_name
        self.cwd = os.getcwd()
    
    def add_csv_filepath(self, csv_filepath):
        """
        Add the csv filepath to the object.

        Args:
            csv_filepath : str
                path string to the location of the csv which contains
                all of the data necessary.
        """
        # check path exists
        if not os.path.exists(csv_filepath):
            raise Exception("The file path cannot be accessed.")
        self.csv_filepath=csv_filepath
    
    def _check_list_of_strs(self, lst):
        """ 
        Checks that the type a list
        and that each element is a string

        Args:
            list : 
                Checking type
        """
        if lst and isinstance(lst, list):
            return all(isinstance(elem, str) for elem in lst)
        else:
            return False

    def add_process_variant_columns(self, process_variant_columns):
        """
        Add the process variant columns information.

        Args:
            process_variant_columns : list of str
                List of strings that corresponds to each of the factors
                describing the variants in the csv file.
        """
        if self._check_list_of_strs(process_variant_columns):
            self.process_variant_columns = process_variant_columns
        else:
            raise TypeError("process_variant_columns must be a non-empty list of strings.")
        
    def add_common_unit_types_column(self, common_unit_types_column):
        """
        Add the common unit module type columns information.

        Args:
            common_unit_types_column : dict
                Dict, where each key is a common unit module type and corresponding
                element is 
        """
        if self._check_list_of_strs(common_unit_types_column):
            self.common_unit_types_column = common_unit_types_column
        else:
            raise TypeError("common_unit_types_column must be a non-empty list of strings.")
    
    def add_feasibility_column(self, feasibility_column):
        """
        Add the feasibility column (i.e., if a simulation was
        successful or not.)

        Args:
            feasibility_column : str
                string corresponding to the feasible column in the 
                csv file.
        """
        if feasibility_column and isinstance(feasibility_column, str):
            self.feasibility_column = [feasibility_column]
        else:
            raise TypeError("feasibility_column must be a non-empty string.")
    
    def add_annualized_cost_column(self, annualized_cost_column):
        """
        Add the annualized cost column.

        Args:
            annualized_cost_column : str
                string corresponding to the annualized cost column in the 
                csv file.
        """
        if annualized_cost_column and isinstance(annualized_cost_column, str):
            self.annualized_cost_column = [annualized_cost_column]
        else:
            raise TypeError("annualized_cost_column must be a non-empty string.")
    
    def add_num_common_unit_type_designs(self, num_common_unit_type_designs):
        """
        Add the number of each common unit type designs.
        This should be a dictionary, where each key corrsponds
        to a common unit module type, and each corresponding element
        is an integer number.

        Args:
            num_common_unit_type_designs : dict
                dictionary that has one key for each common unit module type
                and corresponding int entries for each.
        """
        if isinstance(num_common_unit_type_designs, dict) and \
            all(isinstance(value, int) for value in num_common_unit_type_designs.values()):
            self.num_common_unit_type_designs = num_common_unit_type_designs
        else:
            raise TypeError("num_common_unit_type_designs must be a dict, with each element\
                            equal to a type of int.")
    
    def add_labels_for_common_unit_module_designs(self, labels_for_common_unit_module_designs):
        """
        Adds labels for the common unit module designs.
        These are created automatically when add_num_common_unit_type_designs 
        are created, but they can be replaced with more specific names here.

        Args:
            labels_for_common_unit_module_designs : dict 
                list of labels for each of the designs offered for the 
                common unit module types.
        """
        if isinstance(labels_for_common_unit_module_designs, dict):
            self.labels_for_common_unit_module_designs = labels_for_common_unit_module_designs
        else:
            raise TypeError("labels_for_common_unit_module_designs must be a dict.")
    
    def add_unit_module_capex_columns(self, unit_module_capex_columns):
        """
        Adds the columns corresponding to the capital cost of each unit module type.
        This is used for generating economies of numbers savings.

        Args:
            unit_module_capex_columns : dict
                Dict, where keys are the references to the unit module types,
                and the corresponding string is the capex column of that column
                unit module type in the csv file.
        """
        if isinstance(unit_module_capex_columns, dict):
            self.unit_module_capex_columns = unit_module_capex_columns
        else:
            raise TypeError("unit_module_capex_columns must be a non-empty list of strings.")
    
    def add_process_variant_column_names(self, process_variant_column_names):
        """
        For plotting.
        Adds the names we will use for displaying in plots.
        Not required; if not present, the ones from the 

        Args:
            process_variant_column_names : list or str
                List of strings corresponding to the name of the each variant 
                descriptor to be displayed in plots.
        """
        if self._check_list_of_strs(process_variant_column_names):
            self.process_variant_column_names = process_variant_column_names
        else:
            raise TypeError("process_variant_column_names must be a non-empty list of strings.")
    
    def add_common_module_type_column_names(self, common_module_type_column_names):
        """
        For plotting.
        Adds the names we will use for displaying in plots.
        Not required; if not present, the ones from the 

        Args:
            common_module_type_column_names : list or str
                List of strings corresponding to the name of the each unit
                module type to be displayed in plots.
        """
        if self._check_list_of_strs(common_module_type_column_names):
            self.common_module_type_column_names = common_module_type_column_names
        else:
            raise TypeError("common_module_type_column_names must be a non-empty list of strings.")

    def make_results_dir(self, method):
        """ 
        Initializes and makes all of the directories for results.

        Args:
            method : str
                str kwarg that corresponds to the method we plan to use
                for the directory.
        """

        # check that it is an acceptable method
        assert method in self.methods

        # set up results directory
        cwd=os.getcwd()
        base_results_dir = os.path.join(cwd,f"results/{self.system_name}/")
        results_dir=os.path.join(base_results_dir,f"{method}")
        os.makedirs(results_dir, exist_ok=True)

        # add attributes
        self.base_results_dir = base_results_dir
        self.results_dir = results_dir
