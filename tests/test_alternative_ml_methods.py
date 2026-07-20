from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from process_family.utils.parameters.surrogates import SurrogateParameters
from process_family.utils.trainer.nn import NNTrainer
from process_family.utils.trainer.tree import TreeTrainer

# ============================================================
# 1. Test Parameters for Initilaizing Model Parameters
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
CSV_PATH = TESTS_DIR / "data" / "transcritical-co2-data-small.csv"
# ============================================================
# 2. Test Parameters for Neural Network Methods
# ============================================================
CLASSIFICATION_MODEL_FILE_NN = "linear-classification-nn.keras"
REGRESSION_MODEL_FILE_NN = "regression-nn.keras"
TRAIN_RMSE_NN = 0.13
TEST_RMSE_NN = 0.048
MAE_NN = 0.031
ROC_AUC_NN = 1.0
AVERAGE_PRECISION_NN = 1.0
OFFSET_INPUTS_SCALING_NN = [80.0, 28.0, 50.0, 25.0, 45.0]
FACTOR_INPUTS_SCALING_NN = [120.0, 7.0, 60.0, 50.0, 105.0]
# ============================================================
# 3. Test Parameters for Linear Model Decsion Tree Methods
# ============================================================
CLASSIFICATION_MODEL_FILE_LMDT = "classification-lmdt"
REGRESSION_MODEL_FILE_LMDT = "regression-lmdt"
TRAIN_RMSE_LMDT = 3.891e-09
TEST_RMSE_LMDT = 0.344
MAE_LMDT = 0.141
ROC_AUC_LMDT = 1.0
AVERAGE_PRECISION_LMDT = 1.0
OFFSET_INPUTS_SCALING_LMDT = [80.0, 28.0, 50.0, 25.0, 45.0]
FACTOR_INPUTS_SCALING_LMDT = [120.0, 7.0, 60.0, 50.0, 105.0]


def _build_surrogate_parameters():
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
    return params


def test_nn_trainer_can_train_nn_surrogates(tmp_path):
    """Train and persist classification and regression GBDT surrogates on the reduced dataset."""
    model_dir = tmp_path / "surrogates_models"
    model_dir.mkdir()

    params = _build_surrogate_parameters()
    nn_trainer = NNTrainer(params=params, label="surrogate-test", fit_epoch=200)

    # ============================================================
    # 1. Nerual Network Regression Tests
    # ============================================================
    nn_trainer.train_nn(
        directory=str(model_dir), task="regression", plot_metrics=False, hp_tuning=False
    )
    regression_model = nn_trainer.model
    # Did the model get built and is it located in the correct directory?
    assert regression_model is not None
    assert (model_dir / REGRESSION_MODEL_FILE_NN).exists()

    # Did the class-reg GBDT model cosntruct the surrogate correctly and accuratly to what is expected?
    y_pred_train = regression_model.predict(nn_trainer.X_train_regression)
    y_pred_test = regression_model.predict(nn_trainer.X_test_regression)

    rmse_train = np.sqrt(
        mean_squared_error(nn_trainer.y_train_regression, y_pred_train)
    )
    rmse_test = np.sqrt(mean_squared_error(nn_trainer.y_test_regression, y_pred_test))
    assert TRAIN_RMSE_NN == pytest.approx(rmse_train, abs=0.01)
    assert TEST_RMSE_NN == pytest.approx(rmse_test, abs=0.001)
    assert MAE_NN == pytest.approx(
        mean_absolute_error(nn_trainer.y_test_regression, y_pred_test), abs=0.001
    )

    # ============================================================
    # 2. Neural Network Classification Tests
    # ============================================================
    nn_trainer.train_nn(
        directory=str(model_dir),
        task="linear-classification",
        plot_metrics=True,
        hp_tuning=True,
    )
    classification_model = nn_trainer.model
    # Did the model get built and is it located in the correct directory?
    assert classification_model is not None
    assert (model_dir / CLASSIFICATION_MODEL_FILE_NN).exists()
    # Did the class-reg GBDT model cosntruct the surrogate correctly and accuratly to what is expected?
    scores = classification_model.predict(nn_trainer.X_test_classification)
    assert (
        pytest.approx(roc_auc_score(nn_trainer.y_test_classification, scores))
        == ROC_AUC_NN
    )
    assert (
        pytest.approx(average_precision_score(nn_trainer.y_test_classification, scores))
        == AVERAGE_PRECISION_NN
    )

    # Did the model pass the correct number of scaling factors and their expected values?
    assert len(nn_trainer.scale_x.data_min_) == 5
    assert len(nn_trainer.scale_x.data_range_) == 5
    assert pytest.approx(nn_trainer.scale_x.data_min_) == OFFSET_INPUTS_SCALING_NN
    assert pytest.approx(nn_trainer.scale_x.data_range_) == FACTOR_INPUTS_SCALING_NN


def test_lmdt_trainer_can_train_lmdt_surrogates(tmp_path):
    """Train and persist classification and regression GBDT surrogates on the reduced dataset."""
    model_dir = tmp_path / "surrogates_models"
    model_dir.mkdir()

    params = _build_surrogate_parameters()
    tree_trainer = TreeTrainer(params=params, label="surrogate-test")

    # ============================================================
    # 1. Nerual Network Regression Tests
    # ============================================================
    regression_model = tree_trainer.train_lmdt(
        directory=str(model_dir), task="regression", hp_tune=True
    )

    # Did the model get built and is it located in the correct directory?
    assert regression_model is not None
    assert (model_dir / REGRESSION_MODEL_FILE_LMDT).exists()

    # Did the class-reg GBDT model cosntruct the surrogate correctly and accuratly to what is expected?
    y_pred_train = regression_model.predict(tree_trainer.X_train_regression)
    y_pred_test = regression_model.predict(tree_trainer.X_test_regression)

    rmse_train = np.sqrt(
        mean_squared_error(tree_trainer.y_train_regression, y_pred_train)
    )
    rmse_test = np.sqrt(mean_squared_error(tree_trainer.y_test_regression, y_pred_test))
    assert TRAIN_RMSE_LMDT == pytest.approx(rmse_train, abs=0.01)
    assert TEST_RMSE_LMDT == pytest.approx(rmse_test, abs=0.001)
    assert MAE_LMDT == pytest.approx(
        mean_absolute_error(tree_trainer.y_test_regression, y_pred_test), abs=0.001
    )

    # ============================================================
    # 2. Neural Network Classification Tests
    # ============================================================
    classification_model = tree_trainer.train_lmdt(
        directory=str(model_dir),
        task="classification",
        hp_tune=True,
    )
    # Did the model get built and is it located in the correct directory?
    assert classification_model is not None
    assert (model_dir / CLASSIFICATION_MODEL_FILE_LMDT).exists()
    # Did the class-reg GBDT model cosntruct the surrogate correctly and accuratly to what is expected?
    scores = classification_model.predict(tree_trainer.X_test_classification)
    assert (
        pytest.approx(roc_auc_score(tree_trainer.y_test_classification, scores))
        == ROC_AUC_LMDT
    )
    assert (
        pytest.approx(
            average_precision_score(tree_trainer.y_test_classification, scores)
        )
        == AVERAGE_PRECISION_LMDT
    )

    # Did the model pass the correct number of scaling factors and their expected values?
    assert len(tree_trainer.scale_x.data_min_) == 5
    assert len(tree_trainer.scale_x.data_range_) == 5
    assert pytest.approx(tree_trainer.scale_x.data_min_) == OFFSET_INPUTS_SCALING_LMDT
    assert pytest.approx(tree_trainer.scale_x.data_range_) == FACTOR_INPUTS_SCALING_LMDT
