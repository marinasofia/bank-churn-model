# Bank Churn Model with Fairness Audit

Predicting which credit card customers are about to leave, and checking whether the model treats customer groups equitably. Built with Python, pandas, scikit-learn, scipy, and matplotlib.

## Problem

Churn is expensive: acquiring a new credit card customer costs far more than retaining an existing one. The goal is to flag likely churners early enough for a retention team to intervene, and to size that outreach list against team capacity. Because a missed churner (lost revenue) costs more than a wasted retention call, the model is tuned for recall first.

## Data

10,127 credit card customers from the BankChurners dataset (Kaggle), 23 raw columns covering demographics, product relationship, and 12 months of transaction behavior. Target: whether the customer attrited. 1,627 of 10,127 customers churned, a 16.1% positive class.

Cleaning steps (`bankchurners_cleaning.ipynb`):

- **Leakage removal.** The raw file ships with two `Naive_Bayes_Classifier_*` columns, which are predictions from another model trained on this same target. Keeping them would leak the answer into the features, so they are dropped programmatically.
- **Encoded missing values.** Three categorical columns encode missingness as the string "Unknown" (Education_Level: 1,519; Income_Category: 1,112; Marital_Status: 749). These are recoded to NaN so missingness is visible instead of silently becoming a category.
- **Identifier handling.** `CLIENTNUM` becomes the index: rows stay traceable, but the ID can never enter a model as a feature.
- **Ordered categoricals.** Income and card tier are stored as ordered categories so sorts and group-bys respect real-world order.

## Method

**Feature selection.** EDA (`bankchurners_eda.ipynb`) ranked all numeric features by correlation with churn and verified the top candidates with KDE distribution overlays of churned vs retained customers. The model uses the top 5 behavioral features: `Total_Trans_Ct`, `Total_Ct_Chng_Q4_Q1`, `Total_Revolving_Bal`, `Contacts_Count_12_mon`, `Months_Inactive_12_mon`. Demographics (gender, income) are deliberately excluded from the features.

**Model** (`bankchurners_model.ipynb`): logistic regression on a stratified 80/20 split (`random_state=42`), features standardized with a scaler fit on the training set only.

**Class imbalance: class weighting, not SMOTE.** With 16% positives, an unweighted model maximizes accuracy by ignoring churners. `class_weight='balanced'` reweights errors inversely to class frequency (roughly 5.2x weight on churners) inside the loss function. I chose it over SMOTE because it involves no synthetic data, adds no hyperparameters, and keeps the pipeline simple and auditable; I did not run a SMOTE comparison, which is listed under limitations.

**Threshold analysis.** The model outputs probabilities, and the flagging threshold is a business decision about retention team capacity. Sweeping it on the test set:

| Threshold | Recall | Precision | Customers flagged |
|-----------|--------|-----------|-------------------|
| 0.3 | 0.889 | 0.319 | 906 |
| 0.4 | 0.822 | 0.364 | 733 |
| 0.5 | 0.766 | 0.415 | 600 |
| 0.6 | 0.717 | 0.511 | 456 |
| 0.7 | 0.603 | 0.589 | 333 |

A team that can contact ~900 customers catches 89% of churners at threshold 0.3; a smaller team gets better precision at a higher threshold.

## Results

- **ROC-AUC 0.870** on the held-out test set (2,026 customers).
- **Up to 89% churner recall** (threshold 0.3); at the default 0.5 threshold, recall 0.766 and precision 0.415.
- Standardized coefficients agree with the EDA: transaction count is the dominant signal (-1.355), followed by revolving balance (-0.634), quarter-over-quarter transaction change (-0.584), months inactive (+0.500), and contact count (+0.498). Customers who transact less, carry no revolving balance, go inactive, and contact the bank repeatedly are the ones leaving.

## Fairness audit

Gender and income are not model inputs, but excluding an attribute does not guarantee the model treats those groups the same: behavioral features can act as proxies. That is the logic of fair lending review (ECOA/Reg B), so `audit/fairness_audit.py` rebuilds the exact model and audits its test-set predictions across Gender and Income_Category at threshold 0.5. Full numbers in [`audit/fairness_report.json`](audit/fairness_report.json).

![Per-group recall and precision](audit/fairness_audit.png)

What the audit measures and what it found:

- **Per-group churn rates** (base rates): women churn more than men (17.4% vs 14.6%, chi-squared p = 0.0002); income slices range 13.5% to 17.3%, highest at both extremes (chi-squared p = 0.015).
- **Equal opportunity difference** (recall gap; are actual churners equally likely to be caught?): 0.079 by gender (F 0.80 vs M 0.72). By income it is 0.233, but that gap is driven by the $120K+ slice, which has only 21 actual churners in the test set, so its recall estimate is noisy.
- **Demographic parity difference** (selection rate gap; are groups flagged at equal rates?): 0.087 by gender, with men flagged more often (34.3% vs 25.6%), and 0.103 by income. Chi-squared tests on group vs model flag are significant for both (gender p < 0.001, income p = 0.012).
- **Precision by group**: notably lower for men (0.32 vs 0.52), meaning a larger share of flagged men are false alarms.

**Reading of the results:** the model over-flags men relative to their lower actual churn rate while catching a slightly smaller share of male churners, and the gender gaps run in the opposite direction of the base rates, so this is model behavior, not just a reflection of the data. The gaps are moderate rather than severe, and the intervention is a retention offer, not a credit denial, so the main cost of the disparity is wasted outreach and unevenly missed churners. In production I would monitor these gaps over time, review per-group thresholds, and re-audit at whatever threshold the business actually deploys, since the audit is threshold-dependent.

## Repository guide

| File | Contents |
|------|----------|
| `bankchurners_cleaning.ipynb` | Raw Excel to clean CSV: leakage removal, missing values, types |
| `bankchurners_eda.ipynb` | Churn rates by segment, correlation ranking, top-5 distributions |
| `bankchurners_model.ipynb` | Split, scaling, weighted logistic regression, threshold sweep |
| `audit/fairness_audit.py` | Fairness metrics and figure; writes `fairness_report.json` |

Run order: cleaning notebook, then EDA, then model (each top to bottom), then `python audit/fairness_audit.py` from the project root.

## Limitations

- **Feature selection is univariate.** Ranking by correlation with the target misses interactions and non-linear effects; a tree-based model or L1 selection could surface features this approach ignores.
- **No SMOTE comparison was run.** Class weighting was chosen on reasoning (no synthetic data, simpler pipeline), not on a measured head-to-head.
- **The threshold sweep is evaluated on the test set.** Picking an operating threshold from the same data that reports final metrics is mildly optimistic; a validation split or cross-validation would be cleaner.
- **Thin slices limit the fairness audit.** The $120K+ group has 21 test-set churners, so its recall estimate carries wide uncertainty; the audit reports group sizes alongside every metric for this reason.
- **Single snapshot, single model.** One dataset vintage, one logistic regression. No temporal validation, and churn behavior drifts.
