import pandas as pd

url = r"./data/default of credit card clients.xls"
df = pd.read_excel(url, header=1, engine='xlrd')

df.rename(columns=lambda x: x.strip(), inplace=True)
df = df.drop(columns=["ID"])

df["income"] = df["LIMIT_BAL"]

debt_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
df["debts"] = df[debt_cols].mean(axis=1)

pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
df["punctual_count"] = df[pay_cols].apply(lambda row: sum(v <= 0 for v in row), axis=1)
df["punctual_rate"] = df["punctual_count"] / len(pay_cols)

def compute_score(row):
    if row["punctual_rate"] >= 0.55 and row["debts"] / row["income"] <= 0.5:
        return 1
    elif row["punctual_rate"] >= 0.92 and row["income"] < row["debts"]:
        return 1
    else:
        return 0

df["credit_score"] = df.apply(compute_score, axis=1)

print("Extrait des données traitées :")
print(df[["income", "debts", "punctual_rate", "credit_score"]].head())

output_path = "./data/custom_credit_data.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Données sauvegardées dans : {output_path}")
