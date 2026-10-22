# api_00c3307101e34c3c841f7f861a72b99a forecasting result

The system predicted **water_level** 144 time step(s) ahead.
It regularized 302665 timestamps and detected 18084 missing timestamps.
Inputs were forward-filled only from earlier observations; missing future labels were excluded.

## Selected model

- Model: **ridge**
- Walk-forward validation MASE: **0.7847**
- Test MAE: **0.0955**
- Test RMSE: **0.1342**
- Test MASE: **0.7172**

Lower values are better. A test MASE below 1 means its average error was smaller than the average seasonal change measured in the training period.
