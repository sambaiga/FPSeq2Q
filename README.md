
# FPSeq2Q: Full Distributional  Net-Load Forecasting with Parameterized Quantile Regression.

This repository is the official implementation of [ FPSeq2Q: Full Distributional  Net-Load Forecasting with Parameterized Quantile Regression.](). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```



## Training

To train the model(s) for experiment 1 and 2 in the paper, run this command:

```train
python experiment_objective.py 
```


## Evaluation

1. To re-produce results for experiment one use the Final-results-analysis.ipynb notebook in the notebook floder
2. To re-produce results for experiment two use the Real-time-experiment.ipynb notebook in the notebook floder 


## Results

Our model achieves the following performance on :


| Model     | MAE | NRMSE | CWE | CRSP | PICP |
|-----------|-----|-------|-----|------|------|
| SVR       |5.50 ± 0.86     |  0.12 ± 0.01     |     |      |      |
| RF        | 5.03 ± 0.56    |     0.11 ± 0.01   |     |      |      |
| AR-NET    |   5.12 ± 0.79  |   0.11 ± 0.02    |     |      |      |
| SSM       |     |       |     |      |      |
| RNN-Gauss |     |       |     |      |      |
| FPSeq2Q-MLP |     |       |     |      |      |
