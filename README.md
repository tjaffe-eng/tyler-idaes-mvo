# Process Family Design

This Python repository contains generalized optimization formulations (written in the Algebraic Modeling Language [Pyomo](https://github.com/Pyomo/pyomo)) to solve process family design problems. The methodology has been combined into a Python package for ease of use. This problem can be solved in various ways – currently, in this repository, we can solve this problem using a discretization method or embedded machine-learning surrogates. 

Please reference the paper for this software package as:

    @article{stinchfield2024mixed,
                title={A mixed integer linear programming approach for the design of chemical process families},
                author={Stinchfield, Georgia and Morgan, Joshua C and Naik, Sakshi and Biegler, Lorenz T and Eslick, John C and Jacobson, Clas and Miller, David C and Siirola, John D and Zamarripa, Miguel and Zhang, Chen and others},
                journal={Computers \& Chemical Engineering},
                volume={183},
                pages={108620},
                year={2024},
                publisher={Elsevier}
            }

Please consider also citing the following paper if using the ML surrogates portion of this work:

    @article{stinchfield2025mixed,
                title={Mixed-Integer Linear Programming Formulation with Embedded Machine Learning Surrogates for the Design of Chemical Process Families},
                author={Stinchfield, Georgia and Khalife, Natali and Ammari, Bashar L and Morgan, Joshua C and Zamarripa, Miguel and Laird, Carl D},
                journal={Industrial \& Engineering Chemistry Research},
                year={2025},
                publisher={ACS Publications}
            }
     
## Problem Statement

Process systems is a broad term that captures a variety of systems based on chemical and mechanical processes. Some are small and relatively simple, like air conditioning units, while others are vast and complex, like a power plant. Traditionally, process systems engineers focus on optimizing the complex, nonlinear equations that govern the physics of these systems to find ideal operating conditions and schedules and overall designs of the systems to minimize expenses while meeting customers’ and company requirements.

Manufacturing process systems are rarely considered in the context of optimization in process systems engineering. When designing a singular instance of a system, the deployment context and system itself are used to determine optimal decisions (e.g., operations, schedules, designs). Manufacturing schemes for mass deployment are rarely considered. However, they are key areas for saving capital and deployment costs as it is not likely a process system will only be deployed once.

For example, take a refrigeration system to be deployed at a factory. A team of engineers will decide each possible optimization variable for this particular setting. This is time-consuming; additionally, manufacturing one-off designs is increasingly costly compared to standardized manufacturing lines.
Consider now that this refrigeration system is also being deployed at multiple factories with different load requirements. Rather than viewing each deployment as a singular, isolated optimization problem, we propose designing all the refrigeration systems simultaneously in a way that leads to standardization at the manufacturing levels.

In process family design, we aim to develop an optimization-backed manufacturing scheme for deploying a "family" of process systems. We design elements of these systems commonly, designing a platform of standardized units from which aspects of the system must be selected. We design other system elements uniquely, aiming to balance the trade-offs between standardization and customization.

_Please see the References section below for a more comprehensive problem statement._

## Optimization Techniques

We pose a general problem, coined "process family design", and have developed four subsequent optimization-based techniques to solve it (where each named approach corresponds to the same numbered paper reference below):

1. Discretization approach
2. MILP-representable ML Surrogates approach
3. Decomposition of the Discretization approach
4. Discretization approach + consideration for Economies of Numbers savings

This package and repository demonstrate approaches 1 and 2.

### Approach 1: Discretization<sup>1</sup>

Searching the continuous design space is impractical for complex, non-linear systems. Our first approach involved discretizing across the design space, simulating, and making optimal selections among these discrete design choices. In addition, the formulation shares the structure and properties of the P-median optimization formulation.

_See the discretized.py file in the example directory_

### Approach 2: Embedded MILP-representable Machine Learning Surrogates<sup>2</sup>

Constraining our decisions to discrete designs limits our ability to customize the manufacturing scheme fully; additionally, we have no guarantees that the optimal design is among the discrete options we pre-selected. Our second approach involved training and embedding machine learning surrogates to replace the complex, non-linear system of equations that govern the physics of these models. This allowed us to search the surrogated continuous design space.  

_See the surrogates.py file in the examples directory_

# References

1. G. Stinchfield, J.C Morgan, S. Naik, L.T. Biegler, J.C. Eslick, C. Jacobson, D.C. Miller, J.D. Siirola, M.A. Zamarripa, C. Zhang, Q. Zhang, C.D. Laird, “A Mixed Integer Linear Programming Approach for the Design of Chemical Process Families”. Computers & Chemical Engineering (2024): 108620.

2. G. Stinchfield, N. Khalife, B.L. Ammari, J.C. Morgan, M. Zamarripa, C.D. Laird, "Mixed-Integer Linear Programming Formulation with Embedded Machine Learning Surrogates for the Design of Chemical Process Families". Industrial & Engineering Chemistry Research (2025).

3. G. Stinchfield, J.P. Watson, C.D. Laird, “Progressive Hedging Decomposition for Solutions of Large-Scale Process Family Design Problems”. In proceedings, joint conference of the 34th Annual European Symposium on Computer-Aided Process Engineering (ESCAPE) and Process Systems Engineering (PSE) 2024.

4. G. Stinchfield, S. Jan, J.C. Morgan, M.A. Zamarripa, C.D. Laird, “Optimal Design Approaches for Rapid, Cost-Effective Manufacturing and Deployment of Chemical Process Families with Economies of Numbers”. Systems and Control Transactions 3 (2024) 208-214. 