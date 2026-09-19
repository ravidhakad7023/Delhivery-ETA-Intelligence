class StrategyMemo:

    def __init__(self):
        pass

    def generate_memo(
        self,
        top_hubs,
        top_corridors
    ):

        memo = f"""
NETWORK OPERATIONS STRATEGY MEMO
================================

TOP BOTTLENECK HUBS
-------------------

{top_hubs.head(5).to_string(index=False)}

TOP RISK CORRIDORS
------------------

{top_corridors.head(5).to_string(index=False)}

KEY RECOMMENDATIONS
-------------------

1. Upgrade processing capacity at top bottleneck hubs

2. Shift high-risk corridors toward FTL routing

3. Add redundancy around high-centrality hubs

4. Prioritize SLA-sensitive corridors during peak hours

5. Monitor corridors with chronic delay propagation

EXPECTED IMPACT
---------------

- Reduced SLA breach rates
- Improved ETA accuracy
- Lower operational delay costs
- Better corridor-level planning
"""

        return memo