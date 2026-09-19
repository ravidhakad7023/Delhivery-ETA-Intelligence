from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

FEATURES = [
    "osrm_time",
    "osrm_distance",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_peak_hour",
    "cutoff_factor",
    "route_type",

    # GRAPH FEATURES
    "source_betweenness",
    "source_pagerank",
    "source_in_degree",
    "source_out_degree"
]


TARGET = "delay_ratio"

class BaselineModels:

    def train_linear_regression(self, train_df, test_df):

        model = LinearRegression()

        model.fit(
            train_df[FEATURES],
            train_df[TARGET]
        )

        predictions = model.predict(test_df[FEATURES])

        mae = mean_absolute_error(
            test_df[TARGET],
            predictions
        )

        print(f"Linear Regression MAE: {mae:.4f}")

        return model, predictions

    def train_random_forest(self, train_df, test_df):

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            train_df[FEATURES],
            train_df[TARGET]
        )

        predictions = model.predict(test_df[FEATURES])

        mae = mean_absolute_error(
            test_df[TARGET],
            predictions
        )

        print(f"Random Forest MAE: {mae:.4f}")

        return model, predictions

    def train_xgboost(self, train_df, test_df):

        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        model.fit(
            train_df[FEATURES],
            train_df[TARGET]
        )

        predictions = model.predict(test_df[FEATURES])

        mae = mean_absolute_error(
            test_df[TARGET],
            predictions
        )

        print(f"XGBoost MAE: {mae:.4f}")

        return model, predictions