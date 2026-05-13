"""
This is the parent class for all other ProcessFamily types.
It organizes the specified data into the necessary sets used across the different process family design optimizations.
"""

import pandas as pd
import numpy as np
import random
random.seed(42)
import datetime
import matplotlib.pyplot as plt
import pyomo.environ as pyo

class ProcessFamilyBase:
    def __init__(self, params):
        """
        Organize raw data from a csv file into the sets for PFD optimization.
        This is based on specified product headers & component headers passed by the user.

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

        # SPLIT DATA 

        # read data from csv file into panda dictionary
        data = pd.read_csv(params.csv_filepath)

        # grab data we need, organize into arrays
        variant_data = np.vstack( [ data[p] for p in params.process_variant_columns ] ).T
        common_unit_module_data = np.vstack( [ data[c] for c in params.common_unit_types_column ] ).T
        cost = np.array( data[params.annualized_cost_column] )
        success = np.array( data[params.feasibility_column] )
        n_rows = len(success)

        # make sure dimensions of data are correct
        if variant_data.shape[0]!=n_rows or common_unit_module_data.shape[0]!=n_rows or len(cost)!=n_rows:
            print('Error: Not all columns contain the same number of data points.')
            print('Check dataset.')
            quit()

        # get the size options for each shared component
        L_c = { nm:sorted( set( common_unit_module_data[:,c] ) ) for c,nm in enumerate(params.common_unit_types_column) }

        # get the set of all possible process variants from the data
        V = map(tuple,variant_data.tolist()) # list of tuples of all combinations in the csv file
        V = sorted(set(V)) # sorted set of unique tuples

        # set of feasible alternatives for each process variant
        A_v = {v:[] for v in V}
        cost_va = {}

        # set of feasible alternatives, indexed over v,c, and s (for use in the economies of numbers formulations)
        C=list(params.common_unit_types_column)
        A_vcl = {tuple((v,c,l)):[] for v in V for c in C for l in L_c[c]}

        # loop through data to store alternatives & costs
        for row in range(n_rows):
            # grab & store alternative data and cost data
            variant_specs = tuple(variant_data[row])
            unit_specs = tuple(common_unit_module_data[row])
            va = variant_specs + unit_specs # concatenate process variant and shared component tuples
            cost_va[ va ] = cost[row]

            # alternative is only stored if success = True
            if success[row] == True:
                A_v[variant_specs].append(unit_specs)

                # add the variant also to the A_vcl dict
                for c_ind,c in enumerate(C):
                    l = unit_specs[c_ind]
                    A_vcl[variant_specs,c,l].append(unit_specs)

        # Q_a = list of tuples of (shared_component_name, size) for all shared components that are utilized within a particular alternative
        # Each entry in Q_a should be the same length as the number of shared components
        Q_a = dict()
        for r in range(n_rows):
            Q_a[tuple(common_unit_module_data[r])] = [ (nm,common_unit_module_data[r][c]) for c,nm in enumerate(params.common_unit_types_column) ]

        # add attributes.
        self.C=C
        self.L_c = L_c
        self.V=V
        self.A_v=A_v
        self.A_vcl = A_vcl
        self.Q_a=Q_a
        self.cost_va=cost_va
        self.data=data
        self.variant_data=variant_data
        self.common_unit_module_data=common_unit_module_data
        self.cost_data=cost
        self.success_data=success
        self.process_variant_columns=params.process_variant_columns
        self.n_rows=n_rows

        # check that data makes sense
        self._check_data()

    def _check_data(self):
        """
        Check data & constructed sets to ensure P-Median viability & alternatives availability.

        Args:
            data : pandas df
                Data from csv file
            cost : numpy array
                Array containing costing data for each data point
            sets : dict
                Dictionary containing the sets and parameters to create the model 
                (see :meth:`organize_data_for_product_family_design`)
        Returns:
            None
        """

        # (1) check if all costs are different- if not, give warning that P-median may fail
        # find set of all cost data (i.e. eliminate duplicates)
        list_of_costs=self.cost_data.T.tolist()[0]
        set_of_costs=set(list_of_costs)

        # if the length(set of cost) is not the same as the length(list of costs) there are duplicates
        if len(set_of_costs)<len(self.cost_data): 
            print("total number of costs =", len(self.cost_data))
            print("total number of unique costs =", len(set_of_costs))
            print('\nWarning: not all costs are unique, which means the P-median formulation will potentially fail.\n')

        # (2) Check that all products have at least ONE feasible alternative
        cannot_design=[]
        for v in self.V:
            if len(self.A_v[v])==0:
                print('\nWarning: installation', v, 'does not have any feasible alternatives.')
                print('Removing this installation specification from the set, cannot design for this installation.\n')
                cannot_design.append(v)

        # for those that we could not design, remove from set i in I and add to "cannot design" set
        self.cannot_design=[]
        if len(cannot_design)!=0:
            for v in cannot_design:
                self.cannot_design.append(v)
                self.V.remove(v)
                self.A_v.pop(v)

    def plot_results(self, sol_dict, process_variant_column_names=None, shared_module_type_column_names=None,
                        more_markers=False, set_ticks=True, plot_title=None, show=False, directory=None):
        """
        Plots all process variants with their optimal common unit module designs. 
        Optionally creates a png file.
        Optionally shows plot.

        Args:
            sol_dict : dict
                For each process variant, there is a key in the dict that corresponds to a list
                of the common unit module type designs. 
                Note- designs correspond to the SAME ORDER as how the shared_module_type_columns appear.
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
                Show is True to output plotted results in terminal, otherwise Falseß
                by default True
            directory : str, optional
                Path location (string) to where to save the png file 
                by default None
        Returns:
            None
        """
        # check that there are only 1 OR 2 process variant descriptors, otherwise this func. won't work.
        try:
            assert( len(self.V[0]) <= 2 )
        except:
            print("you are trying to plot a process variant with more than 2 process descriptors.")

        fig=plt.figure()
        ax=fig.add_subplot()

        color_options=['r', 'y', 'g', 'b', 'm', 'k', 'c', 
                        'xkcd:orangered', 'xkcd:candy pink', 'xkcd:aqua', 'xkcd:tangerine', 'xkcd:royal purple']
        marker_options=['.', ',', 'o', 'v', '^', '<', '>', 'p', '*', '+', '_', '|', '8', 'h', 'D']

        # HACK: sometimes we run out of options...
        if more_markers:
            num_perterbs=25
            for _ in range(num_perterbs):
                more_color_options=random.shuffle(color_options)
                color_options.append(more_color_options)
                more_marker_options=random.shuffle(marker_options)
                marker_options.append(more_marker_options)

        xaxis_ticks=[]
        yaxis_ticks=[]

        # how many different combos of units do we have to plot
        all_combos=[]
        for v in self.V:
            all_combos.append(sol_dict[v])
        all_combos=list(set(map(tuple,all_combos)))

        for v in self.V:
            
            # get the label name
            label_name=""
            for c, c_name in enumerate(self.C):
                label_name+=shared_module_type_column_names[c]+"="+str(round(sol_dict[v][c],2))+"\n"

            # find index of current unit combos in all_combos list (this allows for assignment of color/shape)
            combination_index=all_combos.index(tuple(sol_dict[v]))

            # plot point with color / shape corresponding to the index relationship + current label
            plt.scatter( 
                        v[0], v[1], 
                        marker=marker_options[combination_index], 
                        s=50, 
                        color=color_options[combination_index], 
                        label=label_name 
                        )
        
            xaxis_ticks.append(v[0])
            yaxis_ticks.append(v[1])
        
        def legend_without_duplicate_labels(figure):
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            figure.legend(by_label.values(), by_label.keys(),\
                          bbox_to_anchor=(1.35, 0.85), loc="upper right")
        legend_without_duplicate_labels(fig)

        ax.set_ylabel(process_variant_column_names[1])
        ax.set_xlabel(process_variant_column_names[0])

        if set_ticks==True:
            ax.yaxis.set_ticks(sorted(set(yaxis_ticks)))
            ax.xaxis.set_ticks(sorted(set(xaxis_ticks)))

        if plot_title!=None:
            ax.set_title(plot_title)

        if show==True:
            fig.show()
        
        if directory!=None:

            # check if this has a file name and create plot_pathstring
            if directory[-4:]==".png":
                plot_pathstring=directory
            else:
                plot_pathstring=directory+"surrogates-pfd-results.png"

            # save figure
            fig.savefig(
                fname=plot_pathstring,
                dpi=300,
                transparent=True,
                bbox_inches="tight"
            )

    def create_results_summary(self, pfd_solution_method, sol_dict,
                                    show=True, directory=None):
        """
        Prints all installations with their assigned alternatives 
        Optionally creates a csv file of results with printed information.

        Args:
            pfd_solution_method : str
                indicates which of the methods of PFD was used to solve this.
            sol_dict : dict
                For each process variant, there is a key in the dict that corresponds to a list
                of the common unit module type designs. 
                Note- designs correspond to the SAME ORDER as how the shared_module_type_columns appear.
            show : boolean, optional
                Show is True to output organized results in terminal, otherwise False
                by default True
            csv_pathstring : str, optional
                Path location (string) to where to write the csv file 
                by default None
        Returns:
            None
        """
        # check that the model has been solved.
        try:
            results=self.results
        except:
            print("You need to create & solve the MIP before creating the results summary.\n \
                   Execute the func solve_discretized_mip, then execute create_results_summary.")

         # make sure the solution method string matches what we expect to recieve   
        sol_methods=["Surrogates", 
                     "Discretized", 
                     "Discretized-EON"]
        assert pfd_solution_method in sol_methods
        
        if directory is None:
            return

        if directory[-4:]==".txt":
            csv_pathstring=directory
        else:
            csv_pathstring=directory+"opt-results.txt"

        with open(csv_pathstring, 'w') as results_file:

            results_file.write("--------------------------------------------------------------------------------------\n")
            results_file.write(" Process Family Design Results: " + pfd_solution_method + "\n")
            results_file.write("--------------------------------------------------------------------------------------\n")

            current_time = datetime.datetime.now()
            results_file.write( str('Date: ' + str(current_time.month) + '/' + str(current_time.day) + '/' + str(current_time.year) + '\n') )
            results_file.write( str('Time: ' + str(current_time.hour) + ':' + str(current_time.minute) + ':' + str(current_time.second) + '\n'))
            results_file.write("--------------------------------------------------------------------------------------\n")

            # add total annualzied cost (objective)
            results_file.write( str('Total Annualized Cost (i.e. objective) = ' + str(pyo.value(self.model.obj)) + '\n'))
            results_file.write("--------------------------------------------------------------------------------------\n")

            # add solver information
            results_file.write("Optimization Statistics: \n")
            results_file.write("\tTermination condition: " + str(self.results.solver.termination_condition) + "\n")
            results_file.write("\tSolver status: " + str(self.results.solver.status) + "\n")
            results_file.write("--------------------------------------------------------------------------------------\n")

            # add the parameters for N_c
            results_file.write( "Max num. of each common unit type allowed in platform:\n" )
            if pfd_solution_method=="Surrogates":
                for unit in self.labels_for_common_unit_module_designs.keys():
                    results_file.write("\tUnit type = " + str(unit) + " | Max. num. allowed = " + str(self.labels_for_common_unit_module_designs[unit][-1]) + '\n')
            if pfd_solution_method=="Discretized" or pfd_solution_method=="Discretized-EON":
                for c in self.C:
                    results_file.write("\tUnit type = " + str(c) + " | Max. num. allowed = " + str(self.N_c[c]) +" | Max. possible = " + str(len(self.L_c[c])) +'\n')
            results_file.write("--------------------------------------------------------------------------------------\n")

            # add the process variants that could not be designed.
            results_file.write( "Process variants that could NOT be designed (i.e. no feasible alternatives in dataset.) \n" )
            results_file.write( "\tTotal num. variants infeasible = " + str(len(self.cannot_design)) + "\n" )
            for infeasible_variant in self.cannot_design:
                results_file.write( '\n\tVariant :\n' )
                for i_index, i_name in enumerate(self.process_variant_columns):
                    results_file.write( str('\t\t' + str(i_name) + '=' + str(infeasible_variant[i_index]) + '\n') )

            results_file.write("--------------------------------------------------------------------------------------\n")

            if pfd_solution_method=="Discretized-EON":
                results_file.write( "Economies of Numbers Parameters: \n" )
                results_file.write( "\talpha (market elasticity) = " +str(self.alpha) + "\n" )
                results_file.write( "\tDF_max (largest achievable discount) = " +str(self.DF_max) + "\n" )
                results_file.write("--------------------------------------------------------------------------------------\n")

            results_file.write('\nThe assignments of alternatives to variants are:\n')
            for v in sol_dict:

                if show:
                    print('\nVariant:')
                    for v_index, v_name in enumerate(self.process_variant_columns):
                        print('\t', v_name, '=', v[v_index])
                    print('Common Unit Module Designs Selected:')
                    for design_index, design_name in enumerate(self.C):
                        print('\t', design_name, '=', sol_dict[v][design_index])

                results_file.write( '\nVariant :\n' )
                for v_index, v_name in enumerate(self.process_variant_columns):
                    results_file.write( str('\t' + str(v_name) + '=' + str(v[v_index]) + '\n') )
                results_file.write( 'Common Unit Module Designs Selected:\n' )
                for a_index, a_name in enumerate(self.C):
                    results_file.write( str('\t' + str(a_name) + '=' + str(sol_dict[v][a_index]) + '\n') )

    def _individual_unit_design_costs(self):
        """
        Attaches a dictionary, name c_cap_cs, which corresponds to the undiscounted capital cost
        of each of the unit module designs.
        NOTE: this is an internal function because unit_module_capex_columns needs to be added
               before this is called (only is attached already for eon based classes)

        Args:
            unit_module_capex_columns : dict
                column names corresponding to the capital cost columns of data for each unit module type
                each key must be a c in C, and the corresponding (str) element should be the column name corresponding to that
                c's capital cost
        Returns:
            None
        """
        self.c_cap_cl={}
        for c in self.C:
            for l in self.L_c[c]:

                # extract the individual unit module type cost header
                c_capex_header = self.unit_module_capex_columns[c]

                # grab all data that has unit module type c, design s, AND unit module cost != 0 (i.e. feasible)
                cl_data = self.data.loc[ (self.data[c] == l) & (self.data[c_capex_header] > 0.0) ]
                cl_cost = set(list(cl_data[c_capex_header]))

                # check if there was ever a feasible alt.
                isEmpty = (len(cl_cost) == 0)

                if isEmpty:
                    self.c_cap_cl[tuple((c,l))] = 0
                else:
                    self.c_cap_cl[tuple((c,l))] = list(cl_cost)[0]

    def _number_of_manufactured_unit_designs(self, sol_dict):
        """
        Returns a dict, indexed by (common unit module type c, common unit module design labeled l)
        with a single int as the dict value, corresponding to the number of times (c,l) was 
        elected to be manufactured as apart of the optimal solution.

        Args:
            sol_dict : dict
                For each process variant, there is a key in the dict that corresponds to a list
                of the common unit module type designs. 
                Note- designs correspond to the SAME ORDER as how the shared_module_type_columns appear.
        
        Returns:
            manufactured_unit_designs : dict
                For each possible unit module design for each of the common unit module types, there is a 
                key in the dict that will return the (int) value of the num. of times it was selected
                for manufacture, according to the solution.
            designs_selected : list of tuples
                Tuples of (c,l) corresponinding to unit module designs that were manufactured at LEAST 
                one time as apart of the optimal sol (according to sol_dict)
        """

        # dict for holding results
        manufactured_unit_designs = {tuple((c,l)):0 for c in self.C for l in self.L_c[c]}
        designs_selected = []

        # iterate through each sol.
        for v in self.V:

            # grab current sol.
            sol_v = sol_dict[v]

            # the solution is a tuple, where element corresponds to the unit module design of the 
            # common unit module type of the same index as the set initialized as C
            for c_ind, c in enumerate(self.C):

                # grab design value
                design_of_c = sol_v[c_ind]

                # add a tally to the results
                cl = tuple((c,design_of_c))
                manufactured_unit_designs[cl] += 1
                designs_selected.append(cl)

        
        # we only need the set(designs_selected), repeats don't do much for us.
        designs_selected=set(designs_selected)

        return manufactured_unit_designs, designs_selected
    

    def _calculate_discounted_unit_costs(self, alpha, DF_max, 
                                             manufactured_unit_designs):
        """
        Calculates the discounted individual unit costs of each unit module type.
        Also calcualtes the total discount off of the undiscounted cost, attributed
        to economies of numbers correlation savings.

        Args:
            alpha : float between 0,1
                the market "elasticity" parameters; fractional, between 0,1
            DF_max : float between 0,1
                maximum discount factor that can be achieved by economies of numbers savings.
                Conceptualized as the % of CAPEX that is fixed / material costs (i.e. cannot save more than that base cost)
            manufactured_unit_designs : dict
                For each possible unit module design for each of the common unit module types, there is a 
                key in the dict that will return the (int) value of the num. of times it was selected
                for manufacture, according to the solution.
        Returns:
            c_cap_cl_discounted : dict
                keys are (c,l), values are the discounted unit module design cost
            total_discount : float
                total valuation of the discount across all designs
            total_unit_capex : float
                total valuation of the capital cost
        """
        
        # new dict for all of the discounted unit module designs
        c_cap_cl_discounted = {tuple((c,l)):None for c in self.C for l in self.L_c[c]}

        # keep track of running sum of all discounts
        total_discount = 0
        total_unit_capex = 0

        # calculate discounted cost and add to dict
        for c in self.C:
            for l in self.L_c[c]:
                cl = tuple((c,l))
                if manufactured_unit_designs[cl] == 0 or manufactured_unit_designs[cl] == 1:
                    DF_cl = 1
                else:
                    DF_cl = DF_max + (1-DF_max) * manufactured_unit_designs[cl] ** (-alpha)
                c_cap_cl_discounted[cl] = DF_cl * self.c_cap_cl[cl]

                total_discount += manufactured_unit_designs[cl] * (self.c_cap_cl[cl] - c_cap_cl_discounted[cl])
                total_unit_capex +=  manufactured_unit_designs[cl] * self.c_cap_cl[cl]
        
        return c_cap_cl_discounted, total_discount, total_unit_capex
    
    def create_eon_summary(self, pfd_solution_method, sol_dict,
                                alpha = 0.25, DF_max = 0.8,
                                    show = True, directory = None):
        """
        Prints expected savings calculated based on the economies of numbers  
        Optionally creates a csv file of results with printed information.

        Args:
            pfd_solution_method : str
                indicates which of the methods of PFD was used to solve this.
            sol_dict : dict
                For each process variant, there is a key in the dict that corresponds to a list
                of the common unit module type designs. 
                Note- designs correspond to the SAME ORDER as how the shared_module_type_columns appear.
            alpha : float between 0,1, optional
                the market "elasticity" parameters; fractional, between 0,1
                Default, 0.25
            DF_max : float between 0,1 , optional
                maximum discount factor that can be achieved by economies of numbers savings.
                Conceptualized as the % of CAPEX that is fixed / material costs (i.e. cannot save more than that base cost)
                Default, 0.8
            show : boolean, optional
                Show is True to output organized results in terminal, otherwise False
                by default True
            csv_pathstring : str, optional
                Path location (string) to where to write the csv file 
                by default None
        Returns:
            None
        """

        # get the num. of times each unit type was manufactured
        manufactured_unit_designs, designs_selected = self._number_of_manufactured_unit_designs(sol_dict)

        # calculate the discounted cost of each unit vs. original
        c_cap_cl_discounted, total_discount, total_unit_capex = \
            self._calculate_discounted_unit_costs(alpha=alpha,
                                                DF_max=DF_max,
                                                manufactured_unit_designs=manufactured_unit_designs)
        
        if directory is None:
            return

        if directory[-4:]==".txt":
            csv_pathstring=directory
        else:
            csv_pathstring=directory+"opt-eon-results.txt"

        with open(csv_pathstring, 'w') as results_file:

            results_file.write("--------------------------------------------------------------------------------------\n")
            results_file.write(" Process Family Design: " + pfd_solution_method + "\n")
            results_file.write(" Analysis: Economies of Numbers Savings \n")
            results_file.write("--------------------------------------------------------------------------------------\n")

            current_time = datetime.datetime.now()
            results_file.write( str('Date: ' + str(current_time.month) + '/' + str(current_time.day) + '/' + str(current_time.year) + '\n') )
            results_file.write( str('Time: ' + str(current_time.hour) + ':' + str(current_time.minute) + ':' + str(current_time.second) + '\n'))
            results_file.write("--------------------------------------------------------------------------------------\n")

            # add total annualzied cost (objective)
            results_file.write( str('Total Annualized Cost (i.e. objective) = ' + str(pyo.value(self.model.obj)) + '\n'))
            results_file.write( "\tTotal CAPEX  $" + str(total_unit_capex) + "\n" )
            results_file.write( "\tTotal OPEX  $" + str(pyo.value(self.model.obj) - total_unit_capex) + "\n" )
            results_file.write("--------------------------------------------------------------------------------------\n")

            results_file.write( "Economies of Numbers Parameters: \n" )
            results_file.write( "\talpha (market elasticity) = " +str(alpha) + "\n" )
            results_file.write( "\tDF_max (largest achievable discount) = " +str(DF_max) + "\n" )
            results_file.write("--------------------------------------------------------------------------------------\n")

            results_file.write( "Economies of Numbers Results: \n" )

            if pfd_solution_method=="Discretized" or pfd_solution_method=="Surrogates":
                results_file.write("Total Discount = $" + str(total_discount) + "\n" )
                percent_total_cost_savings = total_discount / pyo.value(self.model.obj)
                percent_total_cost_savings = round(percent_total_cost_savings, 3) * 100
                results_file.write("This optimization formulation did not include EON savings within the formulation.\n")
                results_file.write("  The percentage of cost savings (from TOTAL cost) that WOULD have been recognized given this scheme is " + str(percent_total_cost_savings) + "%\n")
            else:
                results_file.write("Total Discount = $" + str(pyo.value(self.model.gamma)) + "\n")
                percent_total_cost_savings = pyo.value(self.model.gamma) / (pyo.value(self.model.obj) + pyo.value(self.model.gamma))
                percent_total_cost_savings = round(percent_total_cost_savings, 3)*100
                results_file.write("  The percentage of cost savings (from TOTAL cost) that this scheme gives is " + str(percent_total_cost_savings) + "%\n")
            percent_cap_cost_savings = total_discount / total_unit_capex
            percent_cap_cost_savings = round(percent_cap_cost_savings, 3)*100
            results_file.write("  The percentage of costs savings (from CAPEX only) that this scheme gives is " + str(percent_cap_cost_savings) + "%\n")

            results_file.write( "\nPer Unit Cost Comparison \n")
            for c,l in designs_selected:
                results_file.write("\tUnit module type = " + str(c) + " | Design = " + str(l)+ " \n")
                results_file.write("\t\tNum. Manufactured = " + str(manufactured_unit_designs[c,l]) + "\n")
                results_file.write("\t\tOriginal per Unit Cost = $" + str(self.c_cap_cl[c,l]) + "\n")
                results_file.write("\t\tDiscounted per Unit Cost = $" + str(c_cap_cl_discounted[c,l]) + "\n" )
                results_file.write("\n")