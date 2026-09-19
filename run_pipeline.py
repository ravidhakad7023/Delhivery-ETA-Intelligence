from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.graph_builder import GraphBuilder
from src.baseline_models import BaselineModels
from src.bottleneck_analysis import BottleneckAnalyzer
from src.evaluation import ModelEvaluator
from src.route_strategy import RouteStrategy
from src.business_impact import BusinessImpact
from src.network_visualization import NetworkVisualizer
from src.strategy_memo import StrategyMemo
from src.route_analysis import RouteAnalysis

DATA_PATH = "data/raw/delivery_data.csv"


def main():

    # =========================================
    # PREPROCESSING
    # =========================================

    preprocessor = DataPreprocessor(DATA_PATH)

    df = preprocessor.load_data()

    df = preprocessor.clean_data(df)

    # =========================================
    # FEATURE ENGINEERING
    # =========================================

    engineer = FeatureEngineer()

    df = engineer.create_temporal_features(df)

    df = engineer.create_targets(df)

    # =========================================
    # SAVE CLEANED DATA
    # =========================================

    df.to_parquet(
        "data/processed/cleaned_data.parquet",
        index=False
    )

    print("Saved cleaned dataset")

    # =========================================
    # ENCODE FEATURES
    # =========================================

    df = preprocessor.encode_features(df)

    # =========================================
    # BUILD CORRIDOR FEATURES
    # =========================================

    corridor_df = engineer.build_corridor_features(df)

    # =========================================
    # SAVE CORRIDOR FEATURES
    # =========================================

    corridor_df.to_parquet(
        "data/processed/corridor_stats.parquet",
        index=False
    )

    corridor_df.to_csv(
        "data/processed/graph_edges.csv",
        index=False
    )

    print("Saved corridor features")

    


    # =========================================
    # GRAPH CONSTRUCTION
    # =========================================

    graph_builder = GraphBuilder()

    G = graph_builder.build_graph(corridor_df)

    graph_metrics_df = (
        graph_builder.compute_graph_metrics(G)
    )

    df = df.fillna(0)
 
    # =========================================
    # MERGE GRAPH FEATURES
    # =========================================

    df = graph_builder.merge_graph_features(df,
    graph_metrics_df
    )

    train_df, val_df, test_df = (
        preprocessor.split_data(df)
    )


    df["source_name"] = (
    df["source_name"]
    .astype(str))

    df["destination_name"] = (
        df["destination_name"]
        .astype(str))
    
    df.to_parquet(
    "data/processed/final_model_data.parquet",
    index=False
    )
    # =========================================
    # FIX OBJECT COLUMNS
    # =========================================

    text_cols = [
        "source_name",
        "destination_name",
        "source_center",
        "destination_center"
    ]

    for col in text_cols:

        if col in train_df.columns:
            train_df[col] = train_df[col].astype(str)

        if col in val_df.columns:
            val_df[col] = val_df[col].astype(str)

        if col in test_df.columns:
            test_df[col] = test_df[col].astype(str)

    # =========================================
    # SAVE SPLITS
    # =========================================

    train_df.to_parquet(
        "data/processed/train.parquet",
        index=False
    )

    val_df.to_parquet(
        "data/processed/val.parquet",
        index=False
    )

    test_df.to_parquet(
        "data/processed/test.parquet",
        index=False
    )

    print("Saved train/val/test splits")
    
    # =========================================
    # SAVE GRAPH NODE FEATURES
    # =========================================

    graph_metrics_df.to_csv(
        "data/processed/graph_nodes.csv",
        index=False
    )

    print("Saved graph node metrics")

    # =========================================
    # ROUTE STRATEGY
    # =========================================

    strategy = RouteStrategy()

    df = strategy.compute_route_risk_score(df)

    df = strategy.recommend_route_type(df)

    strategy_summary = (
        strategy.route_strategy_summary(df)
    )

    print(strategy_summary)

    strategy_summary.to_csv(
        "outputs/reports/route_strategy_summary.csv",
        index=False
    )

    # =========================================
    # ROUTE ANALYSIS
    # =========================================

    route_analysis = RouteAnalysis()

    route_summary_analysis = (
        route_analysis.compare_route_types(df)
    )

    corridor_route_analysis = (
        route_analysis.corridor_level_analysis(df)
    )

    print("\nROUTE TYPE ANALYSIS\n")

    print(route_summary_analysis)

    route_summary_analysis.to_csv(
        "outputs/reports/route_type_analysis.csv"
    )

    corridor_route_analysis.to_csv(
        "outputs/reports/corridor_route_analysis.csv",
        index=False
    )


    # =========================================
    # BASELINE MODELS
    # =========================================

    baseline = BaselineModels()

    xgb_model, predictions = (
        baseline.train_xgboost(
            train_df,
            test_df
        )
    )

    # =========================================
    # EVALUATION
    # =========================================

    evaluator = ModelEvaluator()

    metrics = evaluator.regression_metrics(
        test_df["delay_ratio"],
        predictions
    )

    print(metrics)

    # =========================================
    # BOTTLENECK ANALYSIS
    # =========================================

    analyzer = BottleneckAnalyzer()

    top_hubs = analyzer.rank_hubs(
        graph_metrics_df
    )

    top_corridors = analyzer.rank_corridors(
        corridor_df
    )

    print(top_hubs.head())

    print(top_corridors.head())

    print("Pipeline Completed Successfully")

    # =========================================
    # NETWORK VISUALIZATION
    # =========================================

    visualizer = NetworkVisualizer()

    visualizer.visualize_network(
        G,
        graph_metrics_df,
        top_corridors
    )

    # =========================================
    # BUSINESS IMPACT
    # =========================================

    impact = BusinessImpact()

    top_corridors = (
        impact.estimate_revenue_risk(
            top_corridors
        )
    )

    top_hubs = (
        impact.estimate_hub_impact(
            top_hubs
        )
    )

    top_corridors.to_csv(
        "outputs/reports/top_corridors_business.csv",
        index=False
    )

    top_hubs.to_csv(
        "outputs/reports/top_hubs_business.csv",
        index=False
    )

    print("Business impact analysis completed")

    # =========================================
    # STRATEGY MEMO
    # =========================================

    memo_generator = StrategyMemo()

    memo = memo_generator.generate_memo(
        top_hubs,
        top_corridors
    )

    with open(
        "outputs/reports/strategy_memo.txt",
        "w"
    ) as f:

        f.write(memo)

    print("Strategy memo generated")


if __name__ == "__main__":
    main()