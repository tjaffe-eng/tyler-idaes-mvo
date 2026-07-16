# IDAES Multi-Variant Optimization

Welcome to the documentation for the IDAES Multi-Variant Optimization (idaes_mvo) software package.

This documentation provides the motivation behind the work, how to install indaes_mvo, API references, and how to get started working with the `idaes_mvo` package.

**Get started**

- **Installtion instructions:** See [Getting Started](getting_started.ipynb) section.
- **API Reference:** See [API Reference](getting_started.ipynb#api-reference) section.
- **Examples:** See where to being working with `idaes_mvo` package go to the [Examples](examples.ipynb)
- **Detialed Model Descriptions and Papers** See [Reference Guides](reference_guides.ipynb)

## About IDAES

The Institute for Design of Advanced Energy Systems (IDAES) was originated to bring the most advanced modeling and optimization capabilities to the challenges of transforming and decarbonizing the world’s energy systems to make them environmentally sustainable while maintaining high reliability and low cost. For more information about the IDAES project, see the [IDAES](https://idaes.org/) website.

For more information on the IDAES platform and related projects, now called IDAES+, see the [IDAES+](https://idaesplus.readthedocs.io/latest/) web pages.

For more information on the code and modeling library see the IDAES gibhub page [idaes-pse](https://github.com/IDAES/idaes-pse).

## About IDAES-MVO

This is a python package under the IDAES umbrella that implements optimization-based formulations for designing a family of related process systems rather than a single isolated design. The goal is to balance standardization and customization so that a set of process units can be manufactured and deployed more efficiently.

The repository currently supports two modeling approaches:

- Discretized Approach
- MILP-representable ML Surrogates approach

These approaches are described in more detail in the [Core Models](reference_guides.ipynb#core-models) section, and papers covering the development work are cited in the [Further Reading](reference_guides.ipynb#further-reading)section.