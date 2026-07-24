import os
from pathlib import Path

import numpy as np
import pyomo.environ as pyo
import pytest
from pyomo.opt import SolverFactory
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from process_family.type.surrogates import SurrogatesProcessFamily
from process_family.utils.parameters.surrogates import SurrogateParameters
from process_family.utils.trainer.tree import TreeTrainer

# ============================================================
# 1. Test Parameters for Initilaizing Model Parameters
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
CSV_PATH = TESTS_DIR / "data" / "transcritical-co2-data-small.csv"
# ============================================================
# 2. Test Parameters for Surrogate Model Methods
# ============================================================
CLASSIFICATION_MODEL_FILE = "classification-regression-gbdt"
REGRESSION_MODEL_FILE = "regression-gbdt"
TRAIN_RMSE = 7.010065779990453e-06
TEST_RMSE = 0.3447114891574108
MAE = 0.14072940703625744
ROC_AUC = 1.0
AVERAGE_PRECISION = 1.0
OFFSET_INPUTS_SCALING = [80.0, 28.0, 50.0, 25.0, 45.0]
FACTOR_INPUTS_SCALING = [120.0, 7.0, 60.0, 50.0, 105.0]
# ============================================================
# 3. Test Parameters for Optimization Methods
# ============================================================
BEST_FEASIBLE_OBJECTIVE = 12.31888854
BEST_OBJECTIVE_BOUND = 12.31888854
OBJECTIVE_VALUE = 12.31888854
SET_DESIGN_VARIABLE = {
    (80, 28),
    (80, 29),
    (80, 30),
    (80, 31),
    (80, 32),
    (80, 33),
    (80, 34),
    (80, 35),
    (100, 28),
    (100, 29),
    (100, 30),
    (100, 31),
    (100, 32),
    (100, 33),
    (100, 34),
    (100, 35),
    (120, 28),
    (120, 29),
    (120, 30),
    (120, 31),
    (120, 32),
    (120, 33),
    (120, 34),
    (120, 35),
    (140, 28),
    (140, 29),
    (140, 30),
    (140, 31),
    (140, 32),
    (140, 33),
    (140, 34),
    (140, 35),
    (160, 28),
    (160, 29),
    (160, 30),
    (160, 31),
    (160, 32),
    (160, 33),
    (160, 34),
    (160, 35),
    (180, 28),
    (180, 29),
    (180, 30),
    (180, 31),
    (180, 32),
    (180, 33),
    (180, 34),
    (180, 35),
    (200, 28),
    (200, 29),
    (200, 30),
    (200, 31),
    (200, 32),
    (200, 33),
    (200, 34),
    (200, 35),
}


def _build_surrogate_parameters(load_surrogates=False, model_dir=None):
    """Return a configured SurrogateParameters object matching the reduced surrogate example."""
    params = SurrogateParameters("transcritical-co2-data-small")
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
    params.unit_module_capex_columns = {
        "Evaporator Area": "Annualized Capital Evaporator Cost",
        "Condenser Area": "Annualized Capital Condenser Cost",
        "Compressor Design Flow": "Annualized Capital Compressor Cost",
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
    if load_surrogates:
        assert model_dir is not None
        params.add_surrogates(
            classification_type="gbdt",
            classification_path=str(model_dir / CLASSIFICATION_MODEL_FILE),
            regression_type="gbdt",
            regression_path=str(model_dir / REGRESSION_MODEL_FILE),
            classification_threshold=0.99,
        )
    return params


def _available_solver():
    """Return the first supported solver name, or None if none are available."""
    for solver_name in ("gurobi", "cbc"):
        solver = SolverFactory(solver_name)
        if solver.available(False):
            return solver_name
    return None


def test_surrogate_parameters_initialize_correctly():
    """Confirm surrogate parameters are initialized with the reduced example configuration."""
    params = _build_surrogate_parameters()

    assert params.csv_filepath == str(CSV_PATH)
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
    assert params.unit_module_capex_columns == {
        "Evaporator Area": "Annualized Capital Evaporator Cost",
        "Condenser Area": "Annualized Capital Condenser Cost",
        "Compressor Design Flow": "Annualized Capital Compressor Cost",
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


def test_tree_trainer_can_train_gbdt_surrogates(tmp_path):
    """Train and persist classification and regression GBDT surrogates on the reduced dataset."""
    model_dir = tmp_path / "surrogates_models"
    model_dir.mkdir()

    params = _build_surrogate_parameters()
    trainer = TreeTrainer(params, label="surrogate-test")

    # ============================================================
    # 1. GDBT Regression Tests
    # ============================================================
    regression_model = trainer.train_gbdt(
        task="regression", directory=str(model_dir), plot_metrics=False, hp_tune=True
    )

    # Did the model get built and is it located in the correct directory?
    assert regression_model is not None
    assert (model_dir / REGRESSION_MODEL_FILE).exists()

    # Did the class-reg GBDT model cosntruct the surrogate correctly and accuratly to what is expected?
    y_pred_train = regression_model.predict(trainer.X_train_regression)
    y_pred_test = regression_model.predict(trainer.X_test_regression)

    rmse_train = np.sqrt(mean_squared_error(trainer.y_train_regression, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(trainer.y_test_regression, y_pred_test))
    assert TRAIN_RMSE == pytest.approx(rmse_train)
    assert TEST_RMSE == pytest.approx(rmse_test)
    assert MAE == pytest.approx(
        mean_absolute_error(trainer.y_test_regression, y_pred_test)
    )

    # ============================================================
    # 2. GDBT Classification-Regression Tests
    # ============================================================
    classification_model = trainer.train_gbdt(
        task="classification-regression",
        directory=str(model_dir),
        plot_metrics=False,
        hp_tune=True,
    )
    # Did the model get built and is it located in the correct directory?
    assert classification_model is not None
    assert (model_dir / CLASSIFICATION_MODEL_FILE).exists()
    # Did the class-reg GBDT model cosntruct the surrogate correctly and accuratly to what is expected?
    scores = classification_model.predict(trainer.X_test_classification)
    assert (
        pytest.approx(roc_auc_score(trainer.y_test_classification, scores)) == ROC_AUC
    )
    assert (
        pytest.approx(average_precision_score(trainer.y_test_classification, scores))
        == AVERAGE_PRECISION
    )

    # Did the model pass the correct number of scaling factors and their expected values?
    assert len(trainer.scale_x.data_min_) == 5
    assert len(trainer.scale_x.data_range_) == 5
    assert pytest.approx(trainer.scale_x.data_min_) == OFFSET_INPUTS_SCALING
    assert pytest.approx(trainer.scale_x.data_range_) == FACTOR_INPUTS_SCALING


def test_load_trained_models_and_build_optimization_model(tmp_path):
    """Load trained surrogate models and build the surrogate optimization model."""
    model_dir = tmp_path / "surrogates_models"
    model_dir.mkdir()

    # Initialize parameters and train surrogate models to test loading
    params = _build_surrogate_parameters()
    trainer = TreeTrainer(params, label="surrogate-test")
    trainer.train_gbdt(
        task="classification-regression",
        directory=str(model_dir),
        plot_metrics=False,
        hp_tune=False,
    )
    trainer.train_gbdt(
        task="regression", directory=str(model_dir), plot_metrics=False, hp_tune=True
    )

    # Build optimization model
    params = _build_surrogate_parameters(load_surrogates=True, model_dir=model_dir)
    params.regression_scaling["offset_outputs"] = np.array([0], dtype="float32")
    params.regression_scaling["factor_outputs"] = np.array([1], dtype="float32")
    spfd = SurrogatesProcessFamily(params)
    spfd.build_model(params.labels_for_common_unit_module_designs)

    # Did the optimization model get built and does it have the expected attributes?
    assert hasattr(spfd, "model")
    assert hasattr(spfd.model, "d_vc")
    assert hasattr(spfd.model, "i_v")
    assert hasattr(spfd.model, "p_v")
    assert hasattr(spfd.model, "obj")
    assert spfd.model.obj is not None


@pytest.mark.skipif(
    os.getenv("PRE_COMMIT") == "1", reason="Skipped in pre-commit"
)  # skipping this for pre-commit because it requires a solver to be available and can take time to run
def test_surrogate_optimization_method_solves_returns_solution(tmp_path):
    """Solve the reduced surrogate model and verify returned solution structure."""

    model_dir = tmp_path / "surrogates_models"
    model_dir.mkdir()

    # Initialize parameters and train surrogates for solving
    params = _build_surrogate_parameters()
    trainer = TreeTrainer(params, label="surrogate-test")
    trainer.train_gbdt(
        task="classification-regression",
        directory=str(model_dir),
        plot_metrics=False,
        hp_tune=True,
    )
    trainer.train_gbdt(
        task="regression", directory=str(model_dir), plot_metrics=False, hp_tune=True
    )

    # Build and load surrogate models, build optimization model, and solve.
    params = _build_surrogate_parameters(load_surrogates=True, model_dir=model_dir)
    params.regression_scaling["offset_outputs"] = np.array([0], dtype="float32")
    params.regression_scaling["factor_outputs"] = np.array([1], dtype="float32")
    spfd = SurrogatesProcessFamily(params)
    # add in the output scaling
    spfd.build_model(params.labels_for_common_unit_module_designs)
    spfd.solve_model(solver_name="cbc")

    # Did the optimization model solve successfully and return a valid solution?
    assert spfd.results.solver.termination_condition in (
        pyo.TerminationCondition.optimal,
    )
    assert spfd.results.solver.status == pyo.SolverStatus.ok

    best_feasible_objective = getattr(spfd.results.problem, "upper_bound", None)
    best_objective_bound = getattr(spfd.results.problem, "lower_bound", None)

    assert pytest.approx(best_feasible_objective, rel=1e-8) == BEST_FEASIBLE_OBJECTIVE
    assert pytest.approx(best_objective_bound, rel=1e-8) == BEST_OBJECTIVE_BOUND
    assert isinstance(pyo.value(spfd.model.obj), float)
    assert pytest.approx(pyo.value(spfd.model.obj), rel=1e-6) == OBJECTIVE_VALUE

    # Did the optimization model return a solution dictionary with the expected keys and values?
    solution = spfd.get_results_dict(round_accuracy=3)
    assert set(solution.keys()) == SET_DESIGN_VARIABLE

    for v, designs in solution.items():
        assert designs[spfd.C.index("Evaporator Area")] in [66.667, 105.0]
        assert pytest.approx(designs[spfd.C.index("Condenser Area")]) == 50.0
        assert designs[spfd.C.index("Compressor Design Flow")] in [144.167, 62.5]

    results_file = tmp_path / "surrogates-results.txt"
    spfd.results_summary(show=False, directory=str(results_file))
    assert results_file.exists()
