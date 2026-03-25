"""INCOMO 3 model modules."""

from .metrics import compute_rmse, compute_hbc, compute_hbc_monthly
from .targets import prepare_stationary
from .ensemble import REGIMES, HOUR_TO_REGIME, optimize_regime_weights, apply_regime_weights
from .tree_models import train_tree, retrain_tree, predict_tree, TreeResult
from .elastic_net import train_elastic_net, retrain_elastic_net, predict_elastic_net
try:
    from .lear import (
        LEARResult, train_lear, retrain_lear, predict_lear_test,
        LEAR_DEFAULT_CALIBRATION_WINDOW,
        LEAR_FEATURES_FR, LEAR_FEATURES_UK,
    )
except ModuleNotFoundError:
    pass  # lear.py not yet present
from .dnn import ElecDNN, DNN_DEVICE, train_dnn, predict_dnn
try:
    from .varx_pca import PCARegularizedVARX, VARXPCAResult
except ModuleNotFoundError:
    pass  # varx_pca.py not yet present
from .cnn_lstm import ElecCNNLSTM, CNNLSTMRegressor, CNNLSTM_DEVICE
