# Six profile exemplars

**Generated synthetic examples**, not customer records.
Dataset: generate_profiles(1000, seed=1000). One actual row per use case.
Selection: nearest featurewise category median by squared distance after log-transforming queries, SLA, budget, token lengths and concurrency, then median/IQR scaling. Zero IQR uses scale 1. Accuracy and peak multiplier remain linear; sensitivity is categorical and excluded from distance. Ties use the original profile index.

| profile_index | use_case | monthly_queries | latency_sla_ms | acc_req | monthly_budget_usd | data_sensitivity | peak_multiplier | avg_input_tokens | avg_output_tokens | concurrent_users |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 159 | chatbot | 1542395.5036567794 | 888.4533985125589 | 0.7281263213080076 | 4004.9986549964137 | high | 3.755973337843907 | 99.447977323389 | 291.37530644042795 | 188.46676063062898 |
| 379 | code_generation | 940518.5216239856 | 1694.0341261009141 | 0.8323016578393108 | 4763.180320177551 | high | 3.2811698025612115 | 588.5596248464417 | 312.47792114696284 | 70.39169275554262 |
| 700 | batch_inference | 626196.0842916068 | 10773.706249686646 | 0.717504950771853 | 4721.371151246881 | low | 2.8699868870827796 | 303.0356697344515 | 207.81044689401986 | 204.80127229831308 |
| 685 | rag | 17863903.247442313 | 1245.4369073865014 | 0.7353072855669256 | 13120.223773867088 | high | 2.688963467954143 | 1776.2584626797152 | 203.3208102647368 | 152.97895272397272 |
| 839 | agentic | 1901902.1973722365 | 5699.177900596294 | 0.8234111995819001 | 6309.894190537303 | low | 3.5806005687622813 | 1416.9046163633386 | 515.5471580593294 | 34.39186356462909 |
| 892 | structured_extraction | 1515334.9333793102 | 694.3564652682086 | 0.8117961404939809 | 7429.720579300112 | low | 2.6800559093462515 | 516.919029480634 | 54.84555315919389 | 1120.0153642128203 |
