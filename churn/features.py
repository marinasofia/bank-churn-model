"""The single place where the model's inputs are defined.

Everything that touches the feature matrix imports from here: training, the
fairness audit, prediction, the drift check, and the tests. If a feature is
added or removed it changes in exactly one file.
"""

FEATURES = [
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Total_Revolving_Bal",
    "Contacts_Count_12_mon",
    "Months_Inactive_12_mon",
]

TARGET = "Churned"

# The customer identifier. It is the index of the clean CSV and must never be a
# feature: a model will happily fit to it, and it carries no signal.
ID_COLUMN = "CLIENTNUM"

# Columns whose name starts with one of these are predictions from another
# model trained on the same target. They leak the answer and are rejected on
# sight rather than by exact name, so a reissued file with a slightly different
# column name still fails loudly.
FORBIDDEN_PREFIXES = ("Naive_Bayes",)

# Demographic columns. Not features, but the fairness audit slices by them.
PROTECTED_ATTRIBUTES = ["Gender", "Income_Category"]

INCOME_ORDER = [
    "Less than $40K",
    "$40K - $60K",
    "$60K - $80K",
    "$80K - $120K",
    "$120K +",
]

CARD_ORDER = ["Blue", "Silver", "Gold", "Platinum"]

# Plain-language reason shown to a retention agent when a feature pushes a
# customer's churn probability up. The sign says which direction of the
# feature is the risky one: negative coefficient means low values are risky.
REASON_TEXT = {
    "Total_Trans_Ct": "low transaction count",
    "Total_Ct_Chng_Q4_Q1": "transaction count fell quarter over quarter",
    "Total_Revolving_Bal": "low revolving balance",
    "Contacts_Count_12_mon": "repeated contacts with the bank",
    "Months_Inactive_12_mon": "months inactive",
}
