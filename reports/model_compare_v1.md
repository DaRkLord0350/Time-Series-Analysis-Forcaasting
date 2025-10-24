# 📊 Model Comparison Report: Prophet vs SARIMA

**Lower values indicate better performance.**

| series_id     |   sarima_mae |   prophet_mae |   sarima_mape |   prophet_mape |   sarima_smape |   prophet_smape | winner   |
|:--------------|-------------:|--------------:|--------------:|---------------:|---------------:|----------------:|:---------|
| airline_total |      5.44138 |       72.8345 |       5.44138 |        23.4759 |        5.44138 |         23.3681 | SARIMA   |

## 🏆 Winners per Series
- **airline_total** → 🏅 **SARIMA** → SARIMA fit short-term autocorrelations better.
