"""
This is the surrogates formulation class, which is a child of the base class ProcessFamilyBase.
It builds the equivalent Pyomo model for the surrogates driven formulation of the process family design problem.

See:    Stinchfield, Georgia, et al. 
        "Mixed-Integer Linear Programming Formulation with Embedded Machine Learning Surrogates 
        for the Design of Chemical Process Families." 
        Industrial & Engineering Chemistry Research (2025).
"""

import pyomo.environ as pyo
from pyomo.gdp import Disjunct, Disjunction

try:
    import omlt
    from omlt.io import load_keras_sequential
    from omlt.neuralnet import ReluBigMFormulation
    from omlt import OmltBlock
    from omlt.gbt import GBTBigMFormulation, GradientBoostedTreeModel
    from omlt.linear_tree import LinearTreeGDPFormulation, LinearTreeDefinition
except:
    print("No OMLT import.")
    omlt = None

try: 
    import lineartree
except:
    print("No lineartree import.")
    lineartree = None

try: 
    import keras
except:
    print("No keras import.")
    keras = None

try: 
    import onnx
except:
    print("No onnx import.")
    onnx = None

try: 
    import tensorflow as tf
except:
    print("No tensorflow import.")
    tf = None

from process_family.type.base import ProcessFamilyBase

class SurrogatesProcessFamily(ProcessFamilyBase):

    # set method
    pfd_solution_method = "Surrogates"

    def __init__(self, params):
        """
        Organize raw data from a csv file into the sets for PFD optimization. 
        This is based on specified product headers & component headers passed by the user.
        Add the classification / regression surrogates attributes.

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

                    classification_surrogate : keras.engine.sequential.Sequential
                        trained neural network for predicting the indicator variable.
                    classification_scaling : dict
                        scaled_input_bounds (str) : dict
                            scaled range for each input neuron in classification_surrogate
                        offset_inputs (str) : list of float
                            min. data value for each input neuron in classification_surrogate
                        factor_inputs (str) : list of float
                            max. data value for each input neuron in classification_surrogate
                        offset_outputs (str) : list of float
                            min. data value for each output neuron in classification_surrogate
                        factor_outputs (str) : list of float
                            max. data vluae for each output neuron in classification_surrogate
                    classification_threshold : float
                        value for which the output of the classification neural net must be at min.

                    regression_surrogate : keras.engine.sequential.Sequential
                        trained neural network for predicting the cost variable.
                    regression_scaling : dict
                        scaled_input_bounds (str) : dict
                            scaled range for each input neuron in regression_surrogate
                        offset_inputs (str) : list of float
                            min. data value for each input neuron in regression_surrogate
                        factor_inputs (str) : list of float
                            max. data value for each input neuron in regression_surrogate
                        offset_outputs (str) : list of float
                            min. data value for each output neuron in regression_surrogate
                        factor_outputs (str) : list of float
                            max. data vluae for each output neuron in regression_surrogate
        Returns:
            None.
        """
        
        # call base class init
        super().__init__(params)
        
        # add info ab classification surrogate
        self.classification_surrogate = params.classification_surrogate
        self.classification_scaling = params.classification_scaling
        self.classification_threshold = params.classification_threshold

        # add info ab regression surrogate
        self.regression_surrogate = params.regression_surrogate
        self.regression_scaling = params.regression_scaling
    
    def _build_classification_mip(self,model):
        """
        Builds the MIP equivalent of the trained classification surrogate.
        Creates OMLT block, attaches surrogate to Pyomo model, and connects input/output nodes.
        Acceptable surrogate types: PWL neural network (ReLU and/or linear activations), gradient boosted decision tree, linear model decision tress
        
        Args:
            model : Pyomo ConcreteModel
                Pyomo model for defining & solving the surrogates PFD problem.
        Returns:
            None.
        """

        # if we have a NN for a classification surrogate, add appropriate omlt block
        # if isinstance(self.classification_surrogate, keras.engine.sequential.Sequential):
        if isinstance(self.classification_surrogate, tf.keras.models.Sequential):

            print("\tthe classification surrogate is a neural network.")

            def indicator_rule(block,*args):
                scaler_classification=omlt.OffsetScaling( offset_inputs=self.classification_scaling["offset_inputs"],
                                                          factor_inputs=self.classification_scaling["factor_inputs"],
                                                          offset_outputs=self.classification_scaling["offset_outputs"],
                                                          factor_outputs=self.classification_scaling["factor_outputs"] )
                # load the model with appropriate scaling information
                indicator_nn=load_keras_sequential(self.classification_surrogate,
                                                   scaling_object=scaler_classification,
                                                   scaled_input_bounds=self.classification_scaling["scaled_input_bounds"])
                # transform using ReLUBigM
                indicator_nn_mip=ReluBigMFormulation(indicator_nn)
                block.build_formulation(indicator_nn_mip)

        # if we have a GBDT for a classification surrgate, add appropriate omlt block
        # NOTE: as of OMLT 1.1 - classification GBDT should be trained as a REGRESSION model, as CLASSIFICATION is not supported currently.
        elif isinstance(self.classification_surrogate, onnx.onnx_ml_pb2.ModelProto):

            print("\tthe classification surrogate is a gradient boosted tree.")

            def indicator_rule(block,*args):
                scaler_classification=omlt.OffsetScaling( offset_inputs=self.classification_scaling["offset_inputs"],
                                                          factor_inputs=self.classification_scaling["factor_inputs"],
                                                          offset_outputs=self.classification_scaling["offset_outputs"],
                                                          factor_outputs=self.classification_scaling["factor_outputs"] )
                # get the gbt model (onnx format)
                gbt_model=GradientBoostedTreeModel(onnx_model=self.classification_surrogate,
                                                   scaling_object=scaler_classification,
                                                   scaled_input_bounds=self.classification_scaling["scaled_input_bounds"])
                # use big-M to reformulate the trees
                gbt_reformulation=GBTBigMFormulation(gbt_model)
                block.build_formulation(gbt_reformulation)
                print("hey look, I built the thing!")

        # if we have a LMDT for a classification surrogate, add appropriate omlt block
        if isinstance(self.classification_surrogate, lineartree.lineartree.LinearTreeRegressor) or \
            isinstance(self.classification_surrogate, lineartree.lineartree.LinearTreeClassifier) or \
                isinstance(self.classification_surrogate, dict):

            print("\tthe classification surrogate is a linear model decision tree.")            

            def indicator_rule(block,*args):
                scaler_classification=omlt.OffsetScaling( offset_inputs=self.classification_scaling["offset_inputs"],
                                                          factor_inputs=self.classification_scaling["factor_inputs"],
                                                          offset_outputs=self.classification_scaling["offset_outputs"],
                                                          factor_outputs=self.classification_scaling["factor_outputs"] )
                # load the model with appropriate scaling information
                print(f"{type(self.classification_surrogate) = }")
                indicator_lmdt=LinearTreeDefinition( lt_regressor=self.classification_surrogate,
                                                     scaling_object = scaler_classification,
                                                     scaled_input_bounds = self.classification_scaling["scaled_input_bounds"] )
                # transform using ReLUBigM
                indicator_lmdt_mip=LinearTreeGDPFormulation(indicator_lmdt)
                block.build_formulation(indicator_lmdt_mip)

        # add the surrogate with the correct rule, based on type of surrogate
        model.indicator_surrogate=OmltBlock(self.V, 
                                            rule=indicator_rule)

        # num. of inputs to surrogate = num. process variant descriptors + num. of common unit module types
        num_inputs=len(self.process_variant_columns)+len(self.C)

        # connect inputs
        def connect_indicator_surrogate_inputs(model, *args):

            # extract info- last element = node index, all else = v
            node_index=args[-1]
            v=args[:-1]

            # first |v| inputs must equal the process variant descriptor, corresponding to current node
            if node_index < len(v):
                return ( v[node_index] == model.indicator_surrogate[v].inputs[node_index] )
            
            # once we have gone through all |v|, the remaining nodes must be connected to the d_vc variables
            if node_index >= len(v):
                # to get c's index, subtract |v| from the current node_index
                c_index=node_index-len(v)
                return ( model.d_vc[v,self.C[c_index]] == model.indicator_surrogate[v].inputs[node_index] )
        model.connect_indicator_surrogate_inputs=pyo.Constraint( self.V, range(num_inputs), 
                                                                 rule=connect_indicator_surrogate_inputs )

        # connect output node
        def connect_indicator_surrogate_output(model, *args):
            v=args
            return (model.i_v[v]==model.indicator_surrogate[v].outputs[0])
        model.connect_indicator_surrogate_output=pyo.Constraint( self.V, 
                                                                 rule=connect_indicator_surrogate_output )

    def _build_regression_mip(self,model):
        """
        Builds the MIP equivalent of the trained regression surrogate.
        Creates OMLT block, attaches surrogate to Pyomo model, and connects input/output nodes.
        Acceptable surrogate types: PWL neural network (ReLU and/or linear activations), gradient boosted decision tree, linear model decision tress

        Args:
            model : Pyomo ConcreteModel
                Pyomo model for defining & solving the surrogates PFD problem.
        Returns:
            None.
        """

        # if we have a NN for a regression surrogate, add appropriate omlt block
        # if isinstance(self.regression_surrogate, keras.engine.sequential.Sequential):
        if isinstance(self.regression_surrogate, tf.keras.models.Sequential):
            
            print("\tthe regression surrogate is a neural network.\n")

            def cost_rule(block,*args):
                scaler_regression=omlt.OffsetScaling( offset_inputs=self.regression_scaling["offset_inputs"],
                                                      factor_inputs=self.regression_scaling["factor_inputs"],
                                                      offset_outputs=self.regression_scaling["offset_outputs"],
                                                      factor_outputs=self.regression_scaling["factor_outputs"] )
                # load the model with appropriate scaling information
                cost_nn=load_keras_sequential( self.regression_surrogate,
                                               scaler_regression,
                                               self.regression_scaling["scaled_input_bounds"] )
                # transform using ReLUBigM
                cost_nn_mip=ReluBigMFormulation(cost_nn)
                block.build_formulation(cost_nn_mip)
        
        # if we have a LMDT for a regression surrogate, add appropriate omlt block
        if isinstance(self.regression_surrogate, lineartree.lineartree.LinearTreeRegressor):

            print("\tthe regression surrogate is a linear model decision tree.\n")            

            def cost_rule(block,*args):
                scaler_regression=omlt.OffsetScaling( offset_inputs=self.regression_scaling["offset_inputs"],
                                                      factor_inputs=self.regression_scaling["factor_inputs"],
                                                      offset_outputs=self.regression_scaling["offset_outputs"],
                                                      factor_outputs=self.regression_scaling["factor_outputs"] )
                # load the model with appropriate scaling information
                cost_lmdt=LinearTreeDefinition( self.regression_surrogate,
                                                scaler_regression,
                                                self.regression_scaling["scaled_input_bounds"] )
                # transform using ReLUBigM
                cost_lmdt_mip=LinearTreeGDPFormulation(cost_lmdt)
                block.build_formulation(cost_lmdt_mip)

        # if we have a GBDT for a classification surrgate, add appropriate omlt block
        if isinstance(self.regression_surrogate, onnx.onnx_ml_pb2.ModelProto):

            print("\tthe regression surrogate is a gradient boosted tree.")

            def cost_rule(block,*args):
                scaler_regression=omlt.OffsetScaling( offset_inputs=self.regression_scaling["offset_inputs"],
                                                      factor_inputs=self.regression_scaling["factor_inputs"],
                                                      offset_outputs=self.regression_scaling["offset_outputs"],
                                                      factor_outputs=self.regression_scaling["factor_outputs"] )
                # get the gbt model (onnx format)
                cost_gbdt=GradientBoostedTreeModel(onnx_model=self.regression_surrogate,
                                                   scaling_object=scaler_regression,
                                                   scaled_input_bounds=self.regression_scaling["scaled_input_bounds"])
                # use big-M to reformulate the trees
                cost_gbdt_reformulation=GBTBigMFormulation(cost_gbdt)
                block.build_formulation(cost_gbdt_reformulation)

        model.cost_surrogate=OmltBlock(self.V, 
                                       rule=cost_rule)

        # num. of inputs to nn = num. process variant descriptors + num. of common unit module types
        num_inputs=len(self.process_variant_columns)+len(self.C)

        # connect input/outputs nodes
        def connect_cost_surrogate_inputs(model, *args):

            # args = [v,node_index]
            node_index=args[-1]
            v=args[:-1]

            # first |v| # of inputs = process variant descriptor corresponding to current nn node
            if node_index<len(v):
                return ( v[node_index] == model.cost_surrogate[v].inputs[node_index] )
            
            # once we have gone through all |v|, the remaining nodes must be connected to the d_vc variables
            # i.o.w., now the node input is unit module design d_vc corresponding to unit module type c, variant v
            else:
                # to get c's index, subtract |v| from the current node_index
                c_index=node_index-len(v)
                return ( model.d_vc[v,self.C[c_index]] == model.cost_surrogate[v].inputs[node_index] )
        model.connect_cost_surrogate_inputs=pyo.Constraint( self.V, range(num_inputs),
                                                            rule=connect_cost_surrogate_inputs )
        
        # connect output node
        def connect_cost_surrogate_output(model, *args):
            v=args
            return (model.p_v[v]==model.cost_surrogate[v].outputs[0])
        model.connect_cost_surrogate_output=pyo.Constraint( self.V, 
                                                            rule=connect_cost_surrogate_output)

    def _build_common_unit_module_design_disjuncts(self,model):
        """
        Builds all disjuncts to be included in disjunctions. 
        These represent the decision for which unit module type $l$ is manufactured for each process variant $v$, common unit module type $c$.

        Args:
            model : Pyomo ConcreteModel
                Pyomo model for defining & solving the surrogates PFD problem.
        Returns:
            None.
        """
        
        # get the indices for the selected common unit module designs (i.e. \hat{d}_{c,l})
        d_hat_cl_ind=[]
        for c in self.C:
            for l in self.labels_for_common_unit_module_designs[c]:
                d_hat_cl_ind.append(tuple((c,l)))

        # define UB/LB for BigM transformation purposes
        # note: bounds are the same as other d variable, so here we just adjust dict keys.
        d_hat_cl_LB={tuple((c,l)): self.d_vc_LB[c] for c in self.C for l in self.labels_for_common_unit_module_designs[c]}
        d_hat_cl_UB={tuple((c,l)): self.d_vc_UB[c] for c in self.C for l in self.labels_for_common_unit_module_designs[c]}
        def d_hat_cl_bounds(model,*args):
            return (d_hat_cl_LB[args], d_hat_cl_UB[args])

        # now we can can create the d_hat_cl variable, setting bounds properly.
        model.d_hat_cl=pyo.Var( d_hat_cl_ind,
                                 bounds=d_hat_cl_bounds )

        # generate the indicies for the disjuncts- these are all possible combos of v,c,l
        disjunct_ind=[]
        for v in self.V:
                v_elements=list(v)
                for c in self.C:
                    c_elements=[c]
                    for l in self.labels_for_common_unit_module_designs[c]:
                        l_elements=[l]
                        ind_list=v_elements+c_elements+l_elements
                        disjunct_ind.append(tuple((ind_list)))

        # create disjuncts; we have |L_c| x |C| num. of disjuncts per each variant, v
        # if a particular disjunct is selected, then variant v will have unit design l for common unit type c
        def unit_module_c_design_l_disjunct(disjunct,*args):

            # extract *args
            len_v=len(self.process_variant_columns)
            len_c=len(self.C)
            v=args[:len_v]
            c=args[len_v:len_v+1]
            l=args[-1]
            
            # add disjunct to model
            m=disjunct.model()

            # add disjunct rule to disjunct obj on model
            disjunct.select_common_unit_module_design_l = pyo.Constraint( expr = m.d_vc[v,c] == m.d_hat_cl[c,l] )
        model.unit_module_c_design_l_disjunct = Disjunct( disjunct_ind,
                                                          rule=unit_module_c_design_l_disjunct )

        # package disjunctions; we should have |C| x |V| num. of disjunctions
        def unit_module_c_variant_v_disjunction(model,*args):

            #extract *args
            len_v=len(self.process_variant_columns)
            v=args[:len_v]
            c=args[len_v:][0]

            # grab all disjuncts associated with v,c (i.e. all those with labels l for common unit type c)
            disjunct_list=[]
            for l in self.labels_for_common_unit_module_designs[c]:
                disjunct_list.append(model.unit_module_c_design_l_disjunct[v,c,l])

            return disjunct_list
        model.unit_module_c_variant_v_disjunction=Disjunction( self.V, self.C,
                                                               rule=unit_module_c_variant_v_disjunction )
        
        # to add ordering constraints, generate the proper indices
        ordering_constraints_ind=[]
        for c in self.C:
            for l_ind in range(len(self.labels_for_common_unit_module_designs[c])-1):
                ordering_constraints_ind.append(tuple((c,l_ind)))

        # add ordering constraints for d_hat_cl_ordering_constraints to avoid any potential issues with degeneracy
        def d_hat_cl_ordering_constraints(model, *args):

            # extract *args
            c=args[0]
            l_ind=args[-1]

            return ( model.d_hat_cl[ (c,self.labels_for_common_unit_module_designs[c][l_ind]) ] <= model.d_hat_cl[ (c,self.labels_for_common_unit_module_designs[c][l_ind+1]) ] )
        model.d_hat_cl_ordering_constraints=pyo.Constraint( ordering_constraints_ind,
                                                            rule=d_hat_cl_ordering_constraints )

    def build_model(self, labels_for_common_unit_module_designs):
        """
        Builds the overall MIP Pyomo model for solving the surrogates based PFD problem.
        Attaches built model to the SurrogatesProcessFamily object.

        Args:
            labels_for_common_unit_module_designs : dict
                Each of the keys will be all c \in C (i.e. each common unit module type, c)
                Each corresponding element is a list of labels, to identify the selected common unit module designs.
        Returns:
            None.
        
        """

        # add max designs to object
        self.labels_for_common_unit_module_designs=labels_for_common_unit_module_designs

        # instantiate pyomo concrete model
        model=pyo.ConcreteModel()

        """ (1) add bounds for the continuous design variables.
                NOTE: this is essential for allowing auto big-M transformation later on. """

        # get the lower and upper bounds for each common unit module type (for bigM Transformation.)
        self.d_vc_LB, self.d_vc_UB = {}, {}
        for i,c in enumerate(self.C):
            self.d_vc_LB[c]=min(self.common_unit_module_data.T[i]) # LB for common unit module type c = min(all designs for c considered)
            self.d_vc_UB[c]=max(self.common_unit_module_data.T[i]) # UB for common unit module type c = max(all designs for c considered)

        # add the bounds as constraints
        def d_vc_bounds(model,*args):
            # the args contain v,c: extract c
            c=args[-1]
            return (self.d_vc_LB[c],self.d_vc_UB[c])
        
        # create variables themselves, including rule for bounds.
        model.d_vc=pyo.Var( self.V, self.C,
                            bounds=d_vc_bounds)

        """ (2) add variables for cost & indicator """

        model.i_v=pyo.Var(self.V,
                          within=pyo.Reals) # indicator variable of the variants
        model.p_v=pyo.Var(self.V, 
                          within=pyo.NonNegativeReals) # cost of the variants

        """ (3) create & connect OMLT blocks """
        
        self._build_classification_mip(model)
        self._build_regression_mip(model)
        
        """ (4) build disjunctions for common unit module design decisions """

        self._build_common_unit_module_design_disjuncts(model)

        """ (5) add in thresholding constraint to ensure feasible designs are selected """

        def indicator_threshold_rule(model,*args):
            v=args
            return ( model.i_v[v] >= self.classification_threshold )
        model.indicator_threshold_rule=pyo.Constraint( self.V, rule=indicator_threshold_rule )
        
        """ (6) finally, we add the objective """

        model.obj=pyo.Objective( expr=sum(model.p_v[v] for v in self.V) )

        # add to our object.
        self.model=model


    def solve_model(self, transformation_type="gdp.bigm", solver_name="gurobi"):
        """
        Solves the surrogates mip formulation.
        Assumes build_surrogates_mip called first.
        Attaches solver results to object.

        Args:
            transformation_type : str, optional
                passed directly to pyo.TransformationFactory to specify type of transformation
                optional, big-M by default
            solver_name : str, optional
                passed directly to the pyo.SolverFactory to specify solver name
                optional, gurobi by default
        Returns:
            None.
        """

        # build the model, if not done so already.
        try:
            model=self.model
        except:
            print("The MIP for this problem has not been built yet.\nBuilding now.")
            self.build_surrogates_mip()
        
        # transform model- bigM by default.
        pyo.TransformationFactory(transformation_type).apply_to(self.model)

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
                For each process variant, there is a key in the dict that corresponds to a list
                of the common unit module type designs. 
                Note- designs correspond to the SAME ORDER as how the shared_module_type_columns appear.
        """

        # create dict to hold selected alt for each installation i
        sol_dict={v:[] for v in self.V}

        # loop through each installation & alternative to grab which was selected
        for v in self.V:
            for c in self.C:
                sol_dict[v].append(round(pyo.value(self.model.d_vc[v,c]),round_accuracy))
        
        # add to object
        self.sol_dict=sol_dict
        return sol_dict
    
    def plot(self, process_variant_column_names=None, common_module_type_columns=None,
                set_ticks=True, plot_title=None, show=False, directory=None):
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
                          set_ticks=set_ticks,
                          plot_title=plot_title,
                          show=show,
                          directory=directory)
    
    def results_summary(self, show=True, directory=None):
        """
        plotting results functionality. 
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