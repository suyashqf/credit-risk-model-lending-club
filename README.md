# Credit Risk Model — Lending Club

End-to-end credit risk modelling project built on the Lending Club dataset (2007–2018).
Implements PD, LGD, and EAD models fully aligned with Basel II/III IRBA guidelines.

---

## Project Overview

Banks are required to hold capital against the risk of borrowers defaulting on loans.
This project builds the three models that feed into that capital calculation:

**Expected Loss (EL) = PD × LGD × EAD**

| Component | What it answers | Model used |
|---|---|---|
| PD — Probability of Default | Will this borrower default? | Logistic Regression + WoE Scorecard |
| LGD — Loss Given Default | If they default, how much do we lose? | Two-Stage (Logistic + Linear) |
| EAD — Exposure at Default | How much is owed at default? | Amortization Schedule |
| EL — Expected Loss | Average loss per loan | PD × LGD × EAD |
| RWA — Risk Weighted Assets | Regulatory capital required | Basel IRBA Vasicek Formula |

---

## Dataset

**Source:** [Lending Club Loan Data — Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

- Loan origination years: 2007–2018
- Covers the 2008–2010 financial crisis — provides downturn data required by Basel
- ~1.8M resolved loans (Fully Paid or Charged Off)
- Overall default rate: ~20–25%

> Data files are not included in this repo (too large for GitHub).
> Download from Kaggle and place the CSV in the root folder before running.

---

## Notebook Structure

Single notebook: `credit_risk_model_lending_club.ipynb`

| Part | What it does |
|---|---|
| **Part 1 — Data Loading** | Load raw CSV, define binary default target, parse dates, create vintage year, check missing values |
| **Part 2 — EDA** | Default rates by grade, FICO, DTI, income, purpose, term. Vintage × Grade heatmap. Correlation matrix. |
| **Part 3 — Feature Engineering** | Temporal train/OOT split, missing value treatment, WoE encoding, IV-based feature selection, multicollinearity check |
| **Part 4 — PD Model** | Logistic regression on WoE features, scorecard scaling (PDO=20), Platt calibration, Margin of Conservatism, Traffic Light backtesting, Master Rating Scale |
| **Part 5 — LGD, EAD, EL** | Two-stage LGD model, downturn adjustment, Basel floor, EAD from outstanding balance, EL = PD × LGD × EAD, RWA via Basel IRBA formula |

---

## Key Results

| Metric | Value |
|---|---|
| OOT Gini | ~0.50–0.55 |
| OOT KS Statistic | ~0.38–0.42 |
| OOT AUROC | ~0.75–0.78 |
| Mean Downturn LGD | ~65–75% |
| Portfolio EL Rate | ~8–12% |
| Basel PD Floor Applied | 0.03% |
| Basel LGD Floor Applied | 30% (unsecured retail) |

---

## Methodology

### PD Model

- **Target variable:** Binary default (1 = Charged Off / Default, 0 = Fully Paid)
- **Preprocessing:** Weight of Evidence (WoE) transformation on all features
- **Feature selection:** Information Value (IV) — features with IV < 0.02 dropped
- **Model:** Logistic regression (all coefficients positive — monotonicity enforced by WoE)
- **Scorecard:** Log-odds converted to integer score using PDO = 20, base score = 600
- **Calibration:** Platt scaling to align predicted PD with observed default rates
- **Margin of Conservatism:** MoC = Z(95%) × √(DR × (1−DR) / N) added per rating grade
- **Validation:** Gini, KS, AUROC, Brier Score, Basel Traffic Light backtesting

### Train / Validation / OOT Split

| Split | Period | Purpose |
|---|---|---|
| Train | 2007–2015 | Fit model parameters and WoE encoder |
| Validation | 2016 | Tune thresholds |
| Out-of-Time (OOT) | 2017-2018 | Primary Basel performance benchmark |

> Random splitting is never used — temporal splitting is mandatory for credit risk models
> to avoid data leakage and simulate real production performance.

### LGD Model

- **Population:** Defaulted loans only
- **Target:** LGD = 1 − (Recoveries / EAD), bounded [0, 1]
- **Stage 1:** Logistic regression — predicts probability of complete loss (LGD = 1)
- **Stage 2:** Linear regression — predicts LGD magnitude for partial losses
- **Combined:** E[LGD] = P(complete loss) × 1 + P(partial loss) × E[LGD | partial]
- **Downturn adjustment:** Add-on based on worst observed vintage year (2008–2010)
- **Basel floor:** 30% minimum for unsecured retail exposures

### EAD

- Term loans fully disbursed at origination — no revolving facility
- EAD = outstanding principal balance at time of default
- Basel floor: EAD ≥ current outstanding balance

### Expected Loss & RWA

- Loan-level: EL = PD × LGD × EAD
- Portfolio: Sum of all loan-level ELs
- RWA: Basel IRBA Vasicek formula — K × 12.5 × EAD
- Capital required: 8% × RWA (Pillar 1 minimum)

---

## Basel II/III Regulatory Alignment

| Requirement | Implementation |
|---|---|
| Minimum data history | 11 years (2007–2018 including GFC) |
| Default definition | Charged Off = 90+ DPD per Basel |
| Through-the-cycle PD | Long-run average DR used for calibration |
| Margin of Conservatism | Applied per rating grade at 95% confidence |
| PD floor | 0.03% applied to all grades |
| Downturn LGD | Worst vintage year used as downturn anchor |
| LGD floor | 30% for unsecured retail |
| Out-of-time validation | 2013 held out — never seen during development |
| Traffic Light backtesting | Binomial test per grade — Green / Yellow / Red |
| Rating grade granularity | 10 score bands mapped to master scale |

---

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Download data**

Go to [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club),
download `accepted_2007_to_2018Q4.csv` and place it in the root folder.

**3. Run the notebook**

Open `credit_risk_model_lending_club.ipynb` in Google Colab or Jupyter.
Run cells sequentially from Part 1 through Part 5.

---

## Tech Stack

| Library | Use |
|---|---|
| pandas | Data manipulation and aggregation |
| numpy | Numerical computations |
| scikit-learn | Logistic regression, metrics, calibration |
| scipy | Statistical tests — binomial, normal distribution |
| statsmodels | Beta regression, OLS |
| matplotlib / seaborn | All visualisations |
| pyarrow | Parquet file I/O for checkpoints |

---

## Concepts Demonstrated

- Weight of Evidence (WoE) and Information Value (IV)
- Logistic regression scorecard with PDO scaling
- Platt scaling and probability calibration
- Margin of Conservatism (MoC)
- Two-stage LGD modelling
- Downturn LGD and Basel regulatory floors
- Basel Traffic Light backtesting
- Population Stability Index (PSI)
- Basel IRBA Vasicek RWA formula
- Expected Loss vs Unexpected Loss vs Economic Capital

---

## Author

**Suyash Bajpayee**
[GitHub](https://github.com/suyashqf)

---

*Built as a credit risk modelling portfolio project aligned with Basel II/III IRBA standards.*
