"""
This is the discretized formulation class, which is a child of the base class ProcessFamilyBase.
It builds the equivalent Pyomo model for the discretized formulation of the process family design problem.

See:    Stinchfield, Georgia, et al. 
        "A mixed integer linear programming approach for the design of chemical process families." 
        Computers & Chemical Engineering 183 (2024): 108620.
"""

import pyomo.environ as pyo
from process_family.type.base import ProcessFamilyBase
import csv

class DiscretizedProcessFamily(ProcessFamilyBase):
    def __init__(self, params):
        """
        Args:
            params : <class Parameters>
                A fully initialized Parameters class object that must have following attrs:

                    params.csv_filepath : str
                        location of csv file containing the data
                    params.process_variant_columns : list of str
                        list of column names corresponding to the variables that define the boundary conditions for the 
                        process variants
                    params.common_unit_types_column : list of str
                        list of column names coresponding to the common modules in the process platform.
                    params.feasibility_column: str
                        column name corresponding to True/False feasibility data
                    params.annualized_cost_column : str
                        column name corresponding to the total annualized cost of each boundary condition & unit
        Returns: 
            None.
        """
        
        # call the base class constructor.
        super().__init__(params)
        
        self.pfd_solution_method="Discretized"

        # add in N_c: max. num of each common module 
        try:
            print(params.common_unit_types_column)
            print(params.num_common_unit_type_designs.keys())
            assert sorted(params.common_unit_types_column) == sorted(params.num_common_unit_type_designs.keys())
        except:
            raise AttributeError("check that the shared_component_columns match the keys of the num_shared_component_designs dictionary.")
        
        self.N_c = params.num_common_unit_type_designs

    def build_model(self):
        """
        Builds pyomo model for the discretized product family design problem.
        """

        model = pyo.ConcreteModel()
        model.N_c = self.N_c

        # indices for x_ia: all possible combinations of installation i boundary conditions & alternative
        self.x_va_indices = [ tuple( (v,a) ) for v in self.V for a in self.A_v[v] ]
        model.x_va = pyo.Var(self.x_va_indices, bounds = (0,1)) # 0 <= x_ia <= 1

        # indices for z_ks: all possible possible designs for shared units
        self.z_cl_indices = [ tuple( (c,l) ) for c in self.C for l in self.L_c[c] ]
        model.z_cl = pyo.Var(self.z_cl_indices, within = pyo.Binary) # z_ks = {0,1}

        # obj. = min. total weighted cost of all installations, i
        model.family_cost = pyo.Expression( expr = sum( model.x_va[(v, a)] * self.cost_va[v + a] \
                                                for v in self.V for a in self.A_v[v]) )
        model.obj = pyo.Objective( expr = model.family_cost )

        # only manufacture a certain number of each unit type
        @model.Constraint(self.C)
        def max_number_of_units_to_manufacture(model, c):
            return sum( model.z_cl[c, l] for l in self.L_c[c] ) <= self.N_c[c]
        
        # only 1 alternative can be selected for each installation
        @model.Constraint(self.V)
        def only_select_one_alternative(model, *args):
            v = args # arguments represent tuple entries for installation i
            return sum( model.x_va[v, a] for a in self.A_v[v] ) == 1
        
        # create new list of all installation i, alternative a, Q_a[a] for final constraint
        self.alternative_selectability_data = []
        for v in self.V:
            for a in self.A_v[v]:
                for q in self.Q_a[a]:
                    self.alternative_selectability_data.append(tuple([v, a, q]))
        self.alternative_selectability_data = set(self.alternative_selectability_data)

        # only select alt.'s if all of their individual units are selected for manufacture
        @model.Constraint(self.alternative_selectability_data)
        def alternative_selectability(model, *args):
            va = tuple((args[0:-2])) # all elements, except last two, hold (i,a) data
            c = args[-2] # second to last element holds shared unit name
            l = args[-1] # last element holds shared unit design
            return ( model.x_va[ va ] <= model.z_cl[( c,l )] )

        self.model=model
    
    def solve_model(self, solver_name="gurobi"):
        """ Solve the discretized MIP, using the sets initiatlized in the current obj. """
        
        # build the model, if not done so already.
        try:
            model = self.model
        except:
            print("The MIP for this problem has not been built yet.\nBuilding now.")
            self.build_discretized_mip()

        # solve model instance
        opt=pyo.SolverFactory(solver_name)
        self.results=opt.solve(self.model, tee=True)
    
    def get_results_dict(self, round_accuracy=3):
        """
        returns a dictionary of results from the optimization of the process family design.
        each key corresponds to a process variant, with an associated list of designs.

        Args : 
            round_accuracy : int, optional
                This is to indicate the precision digits of saved design results.
                Optional, defualt is 3
        Returns :
            sol_dict : dict
                For each process variant, there is a key in the dict that corresponds the selected alternative
                i.e. set of designs selected for that process variant
        """

        # create dict to hold selected alt for each installation i
        sol_dict={v:[] for v in self.V}

        # loop through each installation & alternative to grab which was selected
        for v in self.V:
            for a in self.A_v[v]:

                # if x_{v,a}==1, then alternative was selected for this variant
                if pyo.value(self.model.x_va[v,a]) >= 0.98:

                    # add to dict
                    sol_dict[v]=a       
        
        # add to object
        self.sol_dict=sol_dict
        return sol_dict
    
    def save_var_results(self, fpath, round_accuracy=3):
        """
        saves a csv of results from ALL variables in the opt (i.e., z_cl and x_va).
        each key corresponds to a variable, which corresponds to a dict where each key is indeces.

        Args : 
            round_accuracy : int, optional
                This is to indicate the precision digits of saved design results.
                Optional, defualt is 3
        Returns :
            None
        """
        # open file writer
        csv_writer=csv.writer(open(fpath,"w"))
        
        # add all values for x_va and z_cl
        csv_writer.writerow(["var_name", "indices", "value"])
        for v in self.V:
            for a in self.A_v[v]:
                v = list(v)
                a = list(a)
                va = v+a
                va = tuple(va)
                csv_writer.writerow(["x_va", va, pyo.value(self.model.x_va[v,a])])
        for c in self.C:
            for l in self.L_c[c]:
                cl = [c,l]
                cl = tuple(cl)
                csv_writer.writerow(["z_cl", cl, pyo.value(self.model.z_cl[c,l])])
    
    def plot(self, process_variant_column_names=None, common_module_type_columns=None,
                more_markers=False, set_ticks=True, plot_title=None, show=False, directory=None):
        """
        plotting results functionality. 
        also ref plot_results in ProcessFamily

        Args:
            process_variant_column_names : list, optional
                list of names corresponding to variables defining requirements for process variants. 
                by default None. 
                If None, will be set to the process variant names in the .csv file data.
            common_module_type_columns : list, optional
                list of legend variable names corresponding to the unit module types that make up each process variant.
                by default None. 
                If None, will be set to the common unit module types in the .csv file data.
            plot_title : str, optional
                string title for the plot
                by default None
            more_markers : bool, optional
                indicates if the plot will need more colors / markers than those available by default.
                by default, False
            set_ticks : bool, optional
                indicates if the x,y,z ticks should be set based on variants or as default matplotlib
                by default True
            show : boolean, optional
                Show is True to output plotted results in terminal, otherwise False
                by default True
            directory : str, optional
                Path location (string) to where to save the png file 
                by default None
        Returns:
            None
        """

        # if no process_variant_column_names, set to process_variant_columns
        if process_variant_column_names==None:
            process_variant_column_names=self.process_variant_columns
        
        # if no common_module_type_columns, set to C
        if common_module_type_columns==None:
                common_module_type_columns=self.C

        # generate the results dictionary
        self.get_results_dict(round_accuracy=2)

        # call results function from parent class ProcessFamily
        self.plot_results(sol_dict=self.sol_dict,
                          process_variant_column_names=process_variant_column_names,
                          shared_module_type_column_names=common_module_type_columns,
                          more_markers=more_markers,
                          set_ticks=set_ticks,
                          plot_title=plot_title,
                          show=show,
                          directory=directory)
    
    def results_summary(self, show=True, directory=None):
        """
        summary of results.
        also ref plot_results in ProcessFamily

        Args:
            show : boolean, optional
                Show is True to output results in terminal, otherwise False
                by default True
            directory : str, optional
                Path location (string) to where to save the txt file 
                by default None
        Returns:
            None
        """
        
        # check if results dict is present, otherwise generate
        try:
            sol_dict=self.sol_dict
        except:
            sol_dict=self.get_results_dict()
    
        self.create_results_summary(sol_dict=sol_dict,
                                    pfd_solution_method=self.pfd_solution_method,
                                    show=show,
                                    directory=directory)

    def eon_summary(self, unit_module_capex_columns, 
                        alpha = 0.25, DF_max = 0.8, show=True, directory=None):
        """
        summary of results, given eon parameters & literature correlation 
        along with the optimal solution.
        also ref plot_results in ProcessFamily

        Args:
            unit_module_capex_columns : dict
                column names corresponding to the capital cost columns of data for each unit module type
                each key must be a c in C, and the corresponding (str) element should be the column name corresponding to that
                c's capital cost
            alpha : float between 0,1, optional
                the market "elasticity" parameters; fractional, between 0,1
                Default, 0.25
            DF_max : float between 0,1 , optional
                maximum discount factor that can be achieved by economies of numbers savings.
                Conceptualized as the % of CAPEX that is fixed / material costs (i.e. cannot save more than that base cost)
                Default, 0.8
            show : boolean, optional
                Show is True to output results in terminal, otherwise False
                by default True
            directory : str, optional
                Path location (string) to where to save the txt file 
                by default None
        Returns:
            None
        """
        
        # check if results dict is present, otherwise generate
        try:
            sol_dict=self.sol_dict
        except:
            sol_dict=self.get_results_dict()
    
        if self.pfd_solution_method=="Discretized":

            # generate the necessary cost dictionary (attaches to class in function)
            self.unit_module_capex_columns=unit_module_capex_columns
            self._individual_unit_design_costs()
        
        else:
            alpha = self.alpha
            DF_max = DF_max

        # now we can generate
        self.create_eon_summary(pfd_solution_method=self.pfd_solution_method,
                                sol_dict=sol_dict,
                                alpha=alpha,
                                DF_max=DF_max,
                                show=show,
                                directory=directory)