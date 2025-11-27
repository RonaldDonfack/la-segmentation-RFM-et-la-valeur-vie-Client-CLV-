import streamlit as st
from filters import load_data, apply_filters, compute_rfm

st.set_page_config(page_title="Overview", layout="wide")

def main():
    st.title("📊 KPIs — Overview")

    # Chargement
    raw_df = load_data()
    df = apply_filters(raw_df)
    rfm = compute_rfm(raw_df)

    # --- Clients actifs ---
    active_customers = df["CustomerID"].nunique()

    # --- CA total ---
    total_revenue = df["Revenue"].sum()

    # --- Taille Segments RFM ---
    # Nombre de clients filtrés qui ont un RFM calculé
    rfm_in_period = rfm.loc[rfm.index.intersection(df["CustomerID"].unique())]
    nb_rfm_clients = rfm_in_period.shape[0]

    # --- CLV baseline (90 jours) ---
    df["days_since_cohort"] = (df["InvoiceDate"] - df["CohortDate"]).dt.days
    df_90 = df[df["days_since_cohort"] <= 90]
    clv_90 = df_90.groupby("CustomerID")["Revenue"].sum().mean()

    # --- North Star ---
    north_star = total_revenue / active_customers if active_customers else 0

    # --- CA / âge de cohorte ---
    df_age1 = df[df["CohortIndex"] == 1]
    cohort_sizes = df_age1.groupby("CohortMonth")["CustomerID"].nunique()
    cohort_revenue_age1 = df_age1.groupby("CohortMonth")["Revenue"].sum()

    if not cohort_sizes.empty:
        ca_age1 = (cohort_revenue_age1 / cohort_sizes).mean()
    else:
        ca_age1 = 0

    # --- DISPLAY METRICS ---
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Clients actifs", active_customers)
    col2.metric("CA total", f"{total_revenue:,.0f}")
    col3.metric("Taille segments RFM", nb_rfm_clients)
    col4.metric("CLV baseline (90j)", f"{clv_90:,.0f}")
    col5.metric("CA / âge cohorte (âge 1)", f"{ca_age1:,.0f}")

    st.markdown("---")
    st.caption("KPIs calculés après filtrage : période, pays, seuil CA, retours.")


if __name__ == "__main__":
    main()
