import pandas as pd
import numpy as np

# Chemin unique pour le CSV final
# Adapte ce chemin pour votre projet (par exemple "data/newdata-final.csv")
DATA_PATH = "data/newdata-final.csv"


def load_final_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Charge le dataset final utilisé partout (Overview, Cohortes, Segments, Scénarios).

    Colonnes attendues:
      - InvoiceDate (sera converti en datetime si présent)
      - Revenue OU (Quantity et Price)

    On ajoute Revenue si besoin.
    """
    df = pd.read_csv(path)

    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    if "Revenue" not in df.columns and {"Quantity", "Price"}.issubset(df.columns):
        df["Revenue"] = df["Quantity"] * df["Price"]

    return df


def _get_revenue_series(df, prefer_neutralized=True):
    """
    Retourne la série de revenu à utiliser.
    Si Revenue_neutralized existe et prefer_neutralized est True,
    on l'utilise, sinon on utilise Revenue.
    """
    if prefer_neutralized and "Revenue_neutralized" in df.columns:
        return df["Revenue_neutralized"]
    return df["Revenue"]


def compute_global_kpis(df, prefer_neutralized=True):
    """
    Calcule quelques KPIs simples sur un dataframe déjà filtré.

    Retourne un dictionnaire:
      nb_clients
      nb_factures
      total_revenue
      avg_order_value
      avg_revenue_per_customer
    """
    rev = _get_revenue_series(df, prefer_neutralized=prefer_neutralized)

    nb_clients = df["Customer ID"].nunique()
    nb_factures = df["Invoice"].nunique()
    total_revenue = rev.sum()

    avg_order_value = total_revenue / nb_factures if nb_factures > 0 else 0.0
    avg_rev_per_customer = total_revenue / nb_clients if nb_clients > 0 else 0.0

    return {
        "nb_clients": int(nb_clients),
        "nb_factures": int(nb_factures),
        "total_revenue": float(total_revenue),
        "avg_order_value": float(avg_order_value),
        "avg_revenue_per_customer": float(avg_rev_per_customer),
    }


def compute_rfm(df, snapshot_date=None, prefer_neutralized=True, n_bins=5):
    """
    Calcule la table RFM pour un dataframe déjà filtré.

    Colonnes nécessaires dans df:
      InvoiceDate (datetime)
      Invoice
      Customer ID
      Revenue

    Retourne un dataframe rfm avec:
      Recency, Frequency, Monetary, R_score, F_score, M_score, RFM_sum, Segment
    """
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rev = _get_revenue_series(df, prefer_neutralized=prefer_neutralized)

    temp = df.copy()
    temp["RevenueUsed"] = rev

    rfm = temp.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("RevenueUsed", "sum"),
    )

    rfm["R_score"] = pd.qcut(
        rfm["Recency"],
        n_bins,
        labels=list(range(n_bins, 0, -1))
    )
    rfm["F_score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"),
        n_bins,
        labels=list(range(1, n_bins + 1))
    )
    rfm["M_score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"),
        n_bins,
        labels=list(range(1, n_bins + 1))
    )

    rfm[["R_score", "F_score", "M_score"]] = rfm[["R_score", "F_score", "M_score"]].astype(int)
    rfm["RFM_sum"] = rfm[["R_score", "F_score", "M_score"]].sum(axis=1)

    def _segment(row):
        if row["RFM_sum"] >= 12:
            return "High_value"
        elif row["RFM_sum"] >= 8:
            return "Medium_value"
        else:
            return "Low_value"

    rfm["Segment"] = rfm.apply(_segment, axis=1)

    return rfm


def compute_monthly_cohort_retention(df):
    """
    Construit une table de rétention par cohorte mensuelle.

    Colonnes nécessaires:
      InvoiceDate (datetime)
      Customer ID

    Retourne:
      retention_table: DataFrame index = CohortMonth, colonnes = âge (1,2,3,...) en mois, valeurs = taux de rétention
      cohort_sizes:    Série avec la taille initiale de chaque cohorte
    """
    data = df.copy()

    data["InvoiceMonth"] = data["InvoiceDate"].values.astype("datetime64[M]")
    first_purchase = data.groupby("Customer ID")["InvoiceMonth"].min()
    data = data.join(first_purchase.rename("CohortMonth"), on="Customer ID")

    data["InvoiceMonth"] = data["InvoiceMonth"].dt.to_period("M")
    data["CohortMonth"] = data["CohortMonth"].dt.to_period("M")

    year_diff = data["InvoiceMonth"].dt.year - data["CohortMonth"].dt.year
    month_diff = data["InvoiceMonth"].dt.month - data["CohortMonth"].dt.month
    data["CohortIndex"] = year_diff * 12 + month_diff + 1

    cohort_data = (
        data.groupby(["CohortMonth", "CohortIndex"])["Customer ID"]
        .nunique()
        .reset_index()
    )

    cohort_pivot = cohort_data.pivot_table(
        index="CohortMonth",
        columns="CohortIndex",
        values="Customer ID"
    )

    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_sizes, axis=0)

    return retention, cohort_sizes


# Fonctions CLV / scénarios (ancien utils_scenario)


def estimate_avg_revenue_per_customer(df: pd.DataFrame):
    """
    Revenu moyen par client + nombre de clients.
    """
    if "Customer ID" not in df.columns or "Revenue" not in df.columns:
        raise ValueError("Les colonnes 'Customer ID' et 'Revenue' sont requises.")

    revenue_per_customer = df.groupby("Customer ID")["Revenue"].sum()

    avg_rev = revenue_per_customer.mean()
    n_customers = revenue_per_customer.shape[0]

    return float(avg_rev), int(n_customers)


def clv_closed_form(
    avg_rev_per_period: float,
    margin_rate: float,
    r: float,
    d: float,
) -> float:
    """
    Formule fermée du CLV:

    CLV = marge_par_periode * r / (1 + d - r)
    """
    margin_per_period = avg_rev_per_period * margin_rate

    r = min(max(r, 1e-4), 0.999)
    d = max(min(d, 0.999), 0.0)

    clv = margin_per_period * (r / (1 + d - r))
    return float(clv)


def compute_baseline_and_scenario(
    df: pd.DataFrame,
    base_margin: float,
    base_r: float,
    d: float,
    delta_margin: float,
    delta_r: float,
) -> dict:
    """
    Calcule CLV et CA baseline + scénario fermé.
    """
    avg_rev, n_customers = estimate_avg_revenue_per_customer(df)

    r_base = base_r
    margin_base = base_margin

    r_scenario = base_r + delta_r
    margin_scenario = base_margin + delta_margin

    clv_base = clv_closed_form(avg_rev, margin_base, r_base, d)
    clv_scenario = clv_closed_form(avg_rev, margin_scenario, r_scenario, d)

    ca_base = clv_base * n_customers
    ca_scenario = clv_scenario * n_customers

    return {
        "avg_rev": avg_rev,
        "n_customers": n_customers,
        "clv_base": clv_base,
        "clv_scenario": clv_scenario,
        "ca_base": ca_base,
        "ca_scenario": ca_scenario,
    }
