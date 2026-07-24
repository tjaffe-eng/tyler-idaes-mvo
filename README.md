# IDAES-MVO Toolkit

## Project Build and Download Statuses
[![Tests](https://github.com/tjaffe-eng/tyler-idaes-mvo/actions/workflows/deploy-book.yml/badge.svg)](https://github.com/tjaffe-eng/tyler-idaes-mvo/actions/workflows/deploy-book.yml)
[![Documentation](https://img.shields.io/badge/docs-passing-brightgreen)](https://tjaffe-eng.github.io/tyler-idaes-mvo/)
[![GitHub contributors](https://img.shields.io/github/contributors/tjaffe-eng/tyler-idaes-mvo.svg)](https://github.com/tjaffe-eng/tyler-idaes-mvo/graphs/contributors)
[![Merged PRs](https://img.shields.io/github/issues-pr-closed-raw/tjaffe-eng/tyler-idaes-mvo.svg?label=merged+PRs)](https://github.com/tjaffe-eng/tyler-idaes-mvo/pulls?q=is:pr+is:merged)
[![Issue stats](http://isitmaintained.com/badge/resolution/tjaffe-eng/tyler-idaes-mvo.svg)](http://isitmaintained.com/project/tjaffe-eng/tyler-idaes-mvo)
<!-- END Status badges -->

## Overview
The IDAES-MVO Toolkit provides open-source, Python-based, multi-scale tools for multi-variant process modeling. Developed in support of the **Institute for the Design of Advanced Energy Systems (IDAES)**, the toolkit addresses **process family design**: developing optimization-backed manufacturing strategies for deploying families of process systems. By designing a platform of standardized, shared components alongside unique, site-specific elements, the framework balances the trade-offs between standardization and customization.

### Motivation
Imagine deploying industrial refrigeration units across multiple factories with different load demands. Optimizing each installation independently leads to costly, one-off manufacturing runs. Instead of treating each deployment as an isolated problem, this package optimizes the entire family simultaneously to enable standardized manufacturing.

### Solution Methods
Built on the [Pyomo](https://github.com/Pyomo/pyomo) Algebraic Modeling Language, this Python package offers generalized optimization formulations that can be solved via:
* **Discretization Methods**
* **Embedded Machine-Learning Surrogates**

For detailed mathematical models and motivations, review the [explanation notebook](docs/explanation.ipynb) and [reference guide](docs/reference_guides.ipynb)


## Getting Started

* **Installation:** Follow the [Getting Started guide](docs/getting_started.ipynb#Installation) to set up the IDAES-MVO Python package locally.
* **Examples:** Browse the [Examples section](docs/examples.ipynb#Installation) for an overview of available tools. You can also inspect and run these files directly from the [`examples`](examples) folder.
* **Framework & Background:** Read the [reference guide](docs/reference_guides.ipynb) for a detailed breakdown of the underlying mathematical framework.

## System requirements
The code and examples have been tested with the following operating systems:

|Operating system|Supported versions  |
|----------------|--------------------|
| Linux          | Any modern Linux   |
| Windows        | Windows 10         |
| macOS          | Partly supported*  |

Most of the functionality is implemented in Python. In accordance with
the end-of-life for many Python 2 libraries, the IDAES/IDAES-MVO Toolkit is written
for Python 3. The following sub-versions are supported:

* Python 3.13


## Contacts and more information

General, background and overview information on IDAES is available at the [IDAES main website](https://www.idaes.org).
Framework development happens at our [GitHub repo](https://github.com/IDAES/idaes-mvo) where you can [report issues/bugs](https://github.com/IDAES/idaes-mvo/issues) or [make contributions](https://github.com/IDAES/idaes-mvo/pulls).

## Funding acknowledgements

This work was conducted as part of the [Institute for the Design of Advanced Energy Systems (IDAES)](https://idaes.org)
with support through the [Simulation-Based Engineering, Crosscutting Research Program](https://netl.doe.gov/coal/simulation-based-engineering)
within the U.S. Department of Energy’s [Office of Fossil Energy and Carbon Management (FECM)](https://www.energy.gov/fecm/office-fossil-energy-and-carbon-management).

## Contributing

Please see our [Getting Started](docs/getting_started.ipynb) and [Examples section](docs/examples.ipynb#Installation) on how to work with the idaes-mvo source code and contribute changes to the project.

**By contributing to this repository, you are agreeing to all the terms set out in the LICENSE.md and COPYRIGHT.md files in this directory.**
  
## References

1. G. Stinchfield, J.C Morgan, S. Naik, L.T. Biegler, J.C. Eslick, C. Jacobson, D.C. Miller, J.D. Siirola, M.A. Zamarripa, C. Zhang, Q. Zhang, C.D. Laird, “A Mixed Integer Linear Programming Approach for the Design of Chemical Process Families”. Computers & Chemical Engineering (2024): 108620.

2. G. Stinchfield, N. Khalife, B.L. Ammari, J.C. Morgan, M. Zamarripa, C.D. Laird, "Mixed-Integer Linear Programming Formulation with Embedded Machine Learning Surrogates for the Design of Chemical Process Families". Industrial & Engineering Chemistry Research (2025).

3. G. Stinchfield, J.P. Watson, C.D. Laird, “Progressive Hedging Decomposition for Solutions of Large-Scale Process Family Design Problems”. In proceedings, joint conference of the 34th Annual European Symposium on Computer-Aided Process Engineering (ESCAPE) and Process Systems Engineering (PSE) 2024.

4. G. Stinchfield, S. Jan, J.C. Morgan, M.A. Zamarripa, C.D. Laird, “Optimal Design Approaches for Rapid, Cost-Effective Manufacturing and Deployment of Chemical Process Families with Economies of Numbers”. Systems and Control Transactions 3 (2024) 208-214. 