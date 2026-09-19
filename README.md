# Delhivery ETA Intelligence - Graph-Based Network Analytics

## 1. Overview
This project addresses the operational challenge of unpredictable delivery ETAs and network bottlenecks in logistics operations. By analyzing 144,867 historical delivery records, the project constructs a transportation network graph to extract structural metrics (like hub centrality) and feeds them into machine learning models to predict delay ratios. The analytical pipeline successfully identifies critical bottlenecks and high-risk corridors, translating complex graph and model outputs into actionable business strategies for routing and capacity planning.

## 2. Business Problem
Unpredictable Estimated Time of Arrival (ETA) leads to Service Level Agreement (SLA) breaches, poor customer experience, and increased operational costs. In a complex logistics network, delays often propagate from specific bottleneck hubs or structurally inefficient corridors rather than isolated incidents. This project aims to:
- Accurately predict the `delay_ratio` of trips.
- Identify specific hubs and corridors causing network-wide delays.
- Support operational decisions regarding route types (e.g., Carting vs. FTL) and capacity upgrades at specific fulfillment centers.

## 3. Dataset
The analysis is based on historical delivery data located in `data/raw/delivery_data.csv`.
- **Size:** 144,867 records expanded to 31 features during processing.
- **Target Variable:** `delay_ratio` (predicted by the baseline ML models) and node `betweenness` (predicted by the GNN prototype).
- **Key Fields:** Source and destination center IDs, route types (Carting, FTL), actual time taken, OSRM distances, and SLA breach indicators.
- **Network Scope:** The dataset encompasses 1,500 unique centers (nodes) and 2,767 unique corridors (edges).

## 4. Analytical Workflow
The project implements an end-to-end pipeline (`run_pipeline.py`) structured as follows:
1. **Data Preprocessing:** Cleaning and encoding categorical variables.
2. **Feature Engineering:** Creating temporal features and target variables.
3. **Graph Construction:** Building a directed graph of the network and computing node metrics.
4. **Feature Integration:** Merging graph metrics (centrality, degree) back into the main dataset.
5. **Baseline Modeling:** Training Tree-based models (Linear Regression, Random Forest, XGBoost) to predict `delay_ratio`.
6. **Bottleneck Analysis:** Identifying the most strained hubs and corridors.
7. **Business Impact:** Estimating the financial delay costs and revenue risks.
8. **Route Strategy:** Generating a strategic operational memo.

## 5. Exploratory Data Analysis
EDA focused on uncovering operational patterns that impact ETAs:
- **Route Type Disparity:** Carting is the dominant transport mode (143,912 trips) with a mean delay ratio of 2.03. FTL routing represents a tiny fraction (298 trips) but operates over much longer distances (avg. 651 units vs 285 for Carting) with a higher mean delay ratio of 4.23.
- **SLA Breaches:** SLA breaches are highly concentrated in specific corridors (e.g., corridor 1479 -> 1437 has a 100% SLA breach rate across its trips).
- **Network Sparsity:** The graph consists of 1,500 nodes but only 2,767 edges, indicating a highly hub-and-spoke oriented network rather than a fully connected mesh.

## 6. Feature Engineering
The project engineers features across multiple dimensions:
- **Temporal Features:** Derived from timestamp columns to capture peak hours or seasonal effects.
- **Corridor Statistics:** Aggregations such as `mean_delay_ratio`, `delay_variance`, `avg_distance`, and `trip_count` per route.
- **Graph/Network Features:** Node-level structural metrics computed via NetworkX, including `in_degree`, `out_degree`, `betweenness` centrality, and `pagerank`.

## 7. Modeling Approach

### Baseline Models (ETA Prediction)
To predict the `delay_ratio`, the project merges the graph metrics with the tabular data and evaluates standard regression models:
- **Linear Regression:** Used as a simple linear baseline.
- **Random Forest & XGBoost:** Used to capture non-linear interactions between network features, distance, and time. 

### Graph Construction
- **Nodes:** Fulfillment centers (1,500).
- **Edges:** Active delivery corridors (2,767).
- **Flow:** Corridors are weighted by traffic volume and delay statistics to map how delays propagate.

### GNN Model (Network Prototyping)
A Graph Neural Network (GraphSAGE) was prototyped in `gnn_model.ipynb` to analyze network topology. 
- **Architecture:** Two `SAGEConv` layers utilizing PyTorch Geometric.
- **Input:** 4 node features (`in_degree`, `out_degree`, `betweenness`, `pagerank`).
- **Target:** Predicts the node's `betweenness` centrality.
- **Training:** Optimized using MSE loss over 100 epochs, reducing loss from 0.6163 to ~0.0014. 
*(Note: The GNN is used for network representation learning, while the primary ETA prediction relies on the Graph-informed Random Forest/XGBoost models).*

## 8. Model Evaluation
The predictive models for `delay_ratio` were evaluated on a hold-out test set (21,632 records). 

| Model | MAE | RMSE | MAPE |
|------|-----|------|----|
| Linear Regression | 0.5154 | 0.8493 | 26.08% |
| **Random Forest** | **0.4332** | **0.7587** | **20.97%** |
| XGBoost | 0.4744 | 0.7897 | 23.08% |

**Interpretation:** Random Forest outperformed XGBoost and Linear Regression across all metrics, achieving the lowest Mean Absolute Error and an average percentage error of ~21%. 

## 9. Key Findings
1. **Random Forest achieved the best performance** among the tested models for predicting delay ratio based on the MAE, RMSE, and MAPE results (Note: The automated pipeline currently utilizes XGBoost for final prediction generation).
2. **Hub 8.0 is the most severe bottleneck** in the network, with an estimated model-based operational delay cost of $147,324 (used for prioritization, not an observed financial loss).
3. **High-Risk Corridors are highly localized:** For example, corridor 1479 -> 1437 suffers from an extreme mean delay ratio of 6.73 and a 100% SLA breach rate.
4. **FTL routes experience higher proportional delays** (mean ratio 4.23) compared to Carting (mean ratio 2.03), though this is heavily influenced by the longer distances FTL covers.
5. **Network structural metrics (pagerank, betweenness)** were incorporated as network-level features for modeling and bottleneck analysis.

## 10. Bottleneck Analysis
Bottlenecks were identified by computing a composite `bottleneck_score` relying on graph centrality and throughput. 
- **Top Bottleneck Hubs:** Nodes 8.0, 916.0, and 134.0 exhibited the highest betweenness and out-degrees relative to their capacity, marking them as the primary choke points for network traffic.

## 11. Route & Corridor Analysis
The pipeline aggregates trip data to the corridor level (`corridor_route_analysis.csv`), identifying paths with chronic delays.
- **SLA Analysis:** Revealed that certain corridors consistently breach SLAs regardless of the specific trip, pointing to systemic routing or distance underestimation issues (e.g., OSRM distance vs actual traversed distance).

## 12. Business Impact
Using heuristic calculations in `business_impact.py`, the analysis translated delay ratios and trip volumes into estimated financial impacts:
- **Hub Impact:** Top hubs carry estimated delay costs exceeding $100,000 based on the volume of delayed trips flowing through them.
- **Corridor Risk:** Specific high-risk corridors pose estimated revenue risks of up to $5,000 per route due to extreme delay variance and SLA breach rates.
*(Note: The dollar values are model-based heuristic estimates used strictly for prioritizing operational interventions, not observed financial losses).*

## 13. Route Strategy / Recommendations
Based on the automated `strategy_memo.txt`, the following operational actions are recommended:
1. **Upgrade processing capacity** at top bottleneck hubs (specifically Node 8.0 and 916.0).
2. **Evaluate shifting selected high-risk corridors toward FTL routing** as an option where volume permits, to test if bypassing intermediate failing hubs reduces delays.
3. **Add network redundancy** around high-centrality hubs to prevent single points of failure.
4. **Prioritize SLA-sensitive corridors** (like 1479 -> 1437) during peak hours.
5. **Monitor corridors** with chronic delay propagation and high variance.

## 14. Project Outputs
The pipeline generates actionable artifacts in the `outputs/reports/` directory:
- `strategy_memo.txt`: A generated text memo outlining the strategic recommendations.
- `top_corridors_business.csv` & `top_hubs_business.csv`: Ranked lists of hubs/corridors with estimated financial risks.
- `route_type_analysis.csv`: Comparative performance between Carting and FTL.
- `xgb_predictions.csv`: The hold-out predictions generated by the baseline pipeline.

## 15. Project Structure
```text
GNN Based ETA Optimisation/
├── data/
│   ├── raw/
│   │   └── delivery_data.csv
│   └── processed/
├── notebooks/
│   ├── baseline_model.ipynb
│   ├── bottleneck_analysis.ipynb
│   ├── eda.ipynb
│   └── gnn_model.ipynb
├── outputs/
│   ├── figures/
│   ├── models/
│   ├── predictions/
│   └── reports/
│       ├── corridor_route_analysis.csv
│       ├── route_strategy_summary.csv
│       ├── route_type_analysis.csv
│       ├── strategy_memo.txt
│       ├── top_corridors.csv
│       ├── top_corridors_business.csv
│       ├── top_hubs.csv
│       └── top_hubs_business.csv
├── src/
│   ├── baseline_models.py
│   ├── bottleneck_analysis.py
│   ├── business_impact.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── gnn_models.py
│   ├── graph_builder.py
│   ├── network_visualization.py
│   ├── preprocessing.py
│   ├── route_analysis.py
│   ├── route_strategy.py
│   ├── strategy_memo.py
│   ├── utils.py
│   └── visualization.py
└── run_pipeline.py
```

## 16. Reproducibility / How to Run
The entire analytical workflow is orchestrated through a single script. Ensure all dependencies are installed, then execute:

```bash
python run_pipeline.py
```
This script handles data cleaning, graph metric extraction, baseline model training (XGBoost), business impact estimation, and memo generation automatically. 
To explore the GNN prototype or baseline model comparisons, run the Jupyter notebooks in the `notebooks/` directory.

## 17. Technologies Used
- **Data Analysis & Processing:** Python, Pandas, Scikit-learn
- **Graph Construction:** NetworkX
- **Machine Learning:** XGBoost, Random Forest
- **Deep Learning / GNN:** PyTorch, PyTorch Geometric (GraphSAGE)

## 18. Limitations
- **GNN Integration:** The GraphSAGE GNN was prototyped successfully for network representation (predicting centrality) but is not yet functioning as the primary end-to-end ETA predictor; instead, graph features are fed into tree-based models.
- **Estimated Business Metrics:** The financial delay costs and revenue risks are heuristic estimates used for relative ranking, rather than actual localized financial data.
- **Sample Size Imbalance:** The route comparison is limited by the massive imbalance between Carting (143k+ trips) and FTL (298 trips), meaning FTL insights may lack statistical power.

## 19. Future Improvements
- **End-to-End GNN ETA Prediction:** Modify the PyTorch Geometric architecture to directly predict `delay_ratio` rather than intermediate node metrics, and integrate it into `run_pipeline.py`.
- **Financial Data Integration:** Replace the heuristic risk estimates with actual operational cost data to calculate true ROI on capacity upgrades.
- **Dynamic Graph Routing:** Implement a real-time routing algorithm that dynamically updates edge weights based on real-time delays.
