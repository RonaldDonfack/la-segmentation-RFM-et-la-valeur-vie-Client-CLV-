# app/utils.py
import pandas as pd

# ⚠️ adapte ce chemin selon l'endroit du CSV
# Si ton fichier est à la racine du projet : ../newdata-final.csv
DATA_PATH = "C:/Users/etche/Downloads/Projet_data_vis/newdata-final/newdata-final.csv"


def load_final_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Charge le dataset final utilisé pour les scénarios.

    Le fichier doit contenir au moins :
    - 'Customer ID'
    - 'Quantity' et 'Price'  OU bien 'Revenue'
    """
    df = pd.read_csv(path)

    # On ajoute Revenue si besoin
    if "Revenue" not in df.columns and {"Quantity", "Price"}.issubset(df.columns):
        df["Revenue"] = df["Quantity"] * df["Price"]
     # ➕ pour pouvoir filtrer par année
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


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
    Formule fermée du CLV :

    CLV = marge_par_période * r / (1 + d - r)
    """
    # Marge moyenne par période
    margin_per_period = avg_rev_per_period * margin_rate

    # Sécurisation des paramètres
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
    Calcule CLV & CA baseline + scénario.
    """
    avg_rev, n_customers = estimate_avg_revenue_per_customer(df)

    # baseline
    r_base = base_r
    margin_base = base_margin

    # scénario
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
