# TSCB: Section 5.2 experiments

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Dataset download links:

- [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
- [Bank Account Fraud (BAF)](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)
- [Credit Card Fraud Detection (CCFD)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

For the Section 5.2 experiments, place the IEEE-CIS files `train_transaction.csv` and `train_identity.csv` in `data/ieee/`.

Run from the repository root:

```bash
python main.py --data-dir data --seed 42 --cpu
python summarize.py results --seed 42
```

Omit `--cpu` to use CUDA when available. Each run writes five window records to `results/seed-42/`.
