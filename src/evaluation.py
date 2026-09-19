import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class ModelEvaluator:

    def regression_metrics(self, y_true, y_pred):

        mae = mean_absolute_error(y_true, y_pred)

        rmse = np.sqrt(
            mean_squared_error(y_true, y_pred)
        )

        r2 = r2_score(y_true, y_pred)

        mape = np.mean(
            np.abs((y_true - y_pred) / y_true)
        ) * 100

        within_15 = np.mean(
            np.abs((y_true - y_pred) / y_true) < 0.15
        ) * 100

        metrics = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE": mape,
            "Within_15_Percent": within_15
        }

        return metrics