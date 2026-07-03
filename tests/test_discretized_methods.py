import json
import os
from pathlib import Path

import pyomo.environ as pyo
import pytest
from pyomo.opt import SolverFactory

from process_family.type.discretized import DiscretizedProcessFamily
from process_family.utils.parameters.base import Parameters

pd = pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "example"
NOTEBOOK_PATH = EXAMPLE_DIR / "discretized.ipynb"
CSV_PATH = EXAMPLE_DIR / "data" / "transcritical-co2-data.csv"

EXPECTED_DISCRETIZED_OBJECTIVE = 4173030.39856
EXPECTED_DISCRETIZED_SHARED_DESIGNS = {
    "Evaporator Area": {50, 80},
    "Condenser Area": {20, 25},
    "Compressor Design Flow": {60, 105},
}


def _build_notebook_parameters():
    """Return a configured Parameters object matching the discretized example notebook."""
    params = Parameters("transcritical-co2")
    params.csv_filepath = str(CSV_PATH)
    params.process_variant_columns = ["Capacity", "Outside Air Temperature"]
    params.common_unit_types_column = [
        "Evaporator Area",
        "Condenser Area",
        "Compressor Design Flow",
    ]
    params.feasibility_column = ["Success"]
    params.annualized_cost_column = ["Total Annualized Cost"]
    params.num_common_unit_type_designs = {
        "Evaporator Area": 2,
        "Condenser Area": 2,
        "Compressor Design Flow": 2,
    }
    params.labels_for_common_unit_module_designs = {
        "Evaporator Area": range(2),
        "Condenser Area": range(2),
        "Compressor Design Flow": range(2),
    }
    params.process_variant_column_names = [
        "Capacity (tons)",
        "Max. Outside Air Temperature (deg. C)",
    ]
    params.common_module_type_column_names = [
        "Evap. Area ($m^2$)",
        "Cond. Area ($m^2$)",
        "Compr. Flow (mol./s)",
    ]
    return params


def _parse_notebook_source_contains_csv_reference():
    """Return True if the discretized notebook contains the expected CSV path reference."""
    source = NOTEBOOK_PATH.read_text()
    notebook = json.loads(source)
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if "data/transcritical-co2-data.csv" in "".join(cell.get("source", [])):
            return True
    return False


def _available_solver():
    """Return the first available supported solver name, or None if none are available."""
    for solver_name in ("gurobi", "cbc"):
        solver = SolverFactory(solver_name)
        if solver.available(False):
            return solver_name
    return None


def test_notebook_exists_and_references_example_data():
    """Verify the discretized notebook exists and references the example data file."""
    assert NOTEBOOK_PATH.exists(), f"Notebook file missing: {NOTEBOOK_PATH}"
    # Confirm the notebook source references the expected CSV path.
    assert _parse_notebook_source_contains_csv_reference()
    # Confirm the referenced example CSV exists on disk.
    assert CSV_PATH.exists(), f"CSV file missing: {CSV_PATH}"


def test_discretized_notebook_parameters_match_example_configuration():
    """Confirm the helper parameters match the configuration used in the discretized example notebook."""
    params = _build_notebook_parameters()
    # Validate the variant and common unit type columns used by the example.
    assert params.process_variant_columns == ["Capacity", "Outside Air Temperature"]
    assert params.common_unit_types_column == [
        "Evaporator Area",
        "Condenser Area",
        "Compressor Design Flow",
    ]
    assert params.feasibility_column == ["Success"]
    assert params.annualized_cost_column == ["Total Annualized Cost"]
    assert params.num_common_unit_type_designs == {
        "Evaporator Area": 2,
        "Condenser Area": 2,
        "Compressor Design Flow": 2,
    }
    assert params.labels_for_common_unit_module_designs == {
        "Evaporator Area": range(2),
        "Condenser Area": range(2),
        "Compressor Design Flow": range(2),
    }
    assert params.process_variant_column_names == [
        "Capacity (tons)",
        "Max. Outside Air Temperature (deg. C)",
    ]
    assert params.common_module_type_column_names == [
        "Evap. Area ($m^2$)",
        "Cond. Area ($m^2$)",
        "Compr. Flow (mol./s)",
    ]


def test_discretized_example_builds_expected_model():
    """Build the discretized example model and verify the expected model structure and variables."""
    params = _build_notebook_parameters()
    dpfd = DiscretizedProcessFamily(params)

    # Ensure the process family was initialized with variants and common module types.
    assert dpfd.V
    assert dpfd.C == [
        "Evaporator Area",
        "Condenser Area",
        "Compressor Design Flow",
    ]
    assert all(isinstance(v, tuple) for v in dpfd.V)
    assert all(isinstance(alts, list) for alts in dpfd.A_v.values())

    dpfd.build_model()

    # Confirm the built Pyomo model contains the expected variables and objective.
    assert hasattr(dpfd, "model")
    assert hasattr(dpfd.model, "x_va")
    assert hasattr(dpfd.model, "z_cl")
    assert hasattr(dpfd.model, "obj")
    assert len(dpfd.x_va_indices) > 0
    assert len(dpfd.z_cl_indices) > 0
    assert dpfd.model.obj is not None
    assert hasattr(dpfd.model, "family_cost")
    assert str(dpfd.model.obj.expr) != ""


@pytest.mark.skipif(
    os.getenv("PRE_COMMIT") == "1", reason="Skipped in pre-commit"
)  # skipping this for pre-commit because it requires a solver to be available and can take time to run
def test_discretized_example_solves_and_returns_consistent_solution(tmp_path):
    """Solve the discretized example and verify the solution structure, summary output, and objective value."""
    solver_name = _available_solver()
    if solver_name is None:
        pytest.skip("No supported solver available for discretized solve")

    params = _build_notebook_parameters()
    dpfd = DiscretizedProcessFamily(params)
    dpfd.build_model()
    dpfd.solve_model(solver_name=solver_name)

    # Validate the solver returned a feasible or optimal result.
    assert dpfd.results.solver.termination_condition in (
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
    )
    assert dpfd.results.solver.status == pyo.SolverStatus.ok

    solution = dpfd.get_results_dict()
    # Ensure the returned solution covers every process variant and uses valid alternative combinations.
    assert set(solution.keys()) == set(dpfd.V)
    assert all(tuple(solution[v]) in dpfd.A_v[v] for v in dpfd.V)

    selected_designs = {c: set() for c in dpfd.C}
    for v, alt in solution.items():
        for index, c in enumerate(dpfd.C):
            selected_designs[c].add(alt[index])

    # Check the selected designs obey the shared design limits and match expected choices.
    assert all(
        len(selected_designs[c]) <= params.num_common_unit_type_designs[c]
        for c in dpfd.C
    )
    assert all(len(selected_designs[c]) >= 1 for c in dpfd.C)
    assert selected_designs == EXPECTED_DISCRETIZED_SHARED_DESIGNS

    results_file = tmp_path / "discretized-results.txt"
    dpfd.results_summary(show=False, directory=str(results_file))
    assert results_file.exists()
    result_text = results_file.read_text()
    assert "Total Annualized Cost" in result_text
    assert "Solver status" in result_text

    objective_line = [
        line
        for line in result_text.splitlines()
        if line.startswith("Total Annualized Cost")
    ]
    assert objective_line, "Objective value line not found in results summary"
    objective_value = float(objective_line[0].split("=")[-1].strip())

    assert abs(objective_value - EXPECTED_DISCRETIZED_OBJECTIVE) < 1e-2
