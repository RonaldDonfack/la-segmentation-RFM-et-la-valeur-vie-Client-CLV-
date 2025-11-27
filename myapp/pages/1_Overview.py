import streamlit as st
import pandas as pd

from filters import load_data, apply_filters
from utils import compute_global_kpis, compute_rfm

st.set_page_config(page_title="KPIs — Overview", layout="wide")


def compute_clv_90_days(df: pd.DataFrame) -> float:
    """
    CLV baseline empirique :
    CA moyen par client sur ses 90 premiers jours après première facture.
    """
    if df.empty:
        return 0.0

    # Date de première facture par client
    first_purchase = df.groupby("Customer ID")["InvoiceDate"].min().rename("CohortDate")
    tmp = df.merge(first_purchase, on="Customer ID", how="left")

    tmp["days_since_cohort"] = (tmp["InvoiceDate"] - tmp["CohortDate"]).dt.days
    tmp_90 = tmp[tmp["days_since_cohort"].between(0, 90)]

    if tmp_90.empty:
        return 0.0

    revenue_90 = tmp_90.groupby("Customer ID")["Revenue"].sum()
    return float(revenue_90.mean())


def compute_ca_age1(df: pd.DataFrame) -> float:
    """
    CA/âge de cohorte :
    on calcule le CA moyen par client à l'âge 1 de cohorte (mois d'acquisition),
    puis on fait la moyenne sur toutes les cohortes.
    """
    if df.empty:
        return 0.0

    data = df.copy()

    # Mois de facture et mois de cohorte
    data["InvoiceMonth"] = data["InvoiceDate"].values.astype("datetime64[M]")
    first_purchase_month = data.groupby("Customer ID")["InvoiceMonth"].min()
    data = data.join(first_purchase_month.rename("CohortMonth"), on="Customer ID")

    data["InvoiceMonth"] = data["InvoiceMonth"].dt.to_period("M")
    data["CohortMonth"] = data["CohortMonth"].dt.to_period("M")

    year_diff = data["InvoiceMonth"].dt.year - data["CohortMonth"].dt.year
    month_diff = data["InvoiceMonth"].dt.month - data["CohortMonth"].dt.month
    data["CohortIndex"] = year_diff * 12 + month_diff + 1

    # On se limite à l'âge 1
    age1 = data[data["CohortIndex"] == 1]
    if age1.empty:
        return 0.0

    cohort_sizes = age1.groupby("CohortMonth")["Customer ID"].nunique()
    cohort_revenue = age1.groupby("CohortMonth")["Revenue"].sum()

    ca_par_client_par_cohorte = cohort_revenue / cohort_sizes
    return float(ca_par_client_par_cohorte.mean())


def main():
    st.title("📊 KPIs — Vue d’ensemble")

    # ===== 1. Chargement global & filtres =====
    df_all = load_data()
    df_filtered, filters_summary, badge = apply_filters(df_all)

    # Affichage des filtres actifs
    st.caption(
        f"**Filtres actifs** — Période : {filters_summary['Période']} | "
        f"Unité de temps : {filters_summary['Unité de temps']} | "
        f"Pays : {filters_summary['Pays'] or 'Tous'} | "
        f"Type client : {filters_summary['Type client'] or 'Tous'} | "
        f"Seuil commande ≥ {filters_summary['Seuil commande']:.2f} | "
        f"Retours : {filters_summary['Retours']}"
    )
    if badge:
        st.info(f"🛈 Mode retours : **{badge}**")

    if df_filtered.empty:
        st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")
        return

    # ===== 2. KPIs globaux (basés sur utils.compute_global_kpis) =====
    global_kpis = compute_global_kpis(df_filtered, prefer_neutralized=True)
    nb_clients = global_kpis["nb_clients"]
    total_revenue = global_kpis["total_revenue"]
    north_star = global_kpis["avg_revenue_per_customer"]  # CA moyen / client actif

    # ===== 3. RFM sur le périmètre filtré =====
    rfm = compute_rfm(df_filtered, prefer_neutralized=True)
    nb_rfm_clients = rfm.shape[0]

    # ===== 4. CLV baseline (90 jours) =====
    clv_90 = compute_clv_90_days(df_filtered)

    # ===== 5. CA / âge de cohorte (âge 1) =====
    ca_age1 = compute_ca_age1(df_filtered)

    # ===== 6. Affichage des cartes KPIs =====
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        label="Clients actifs",
        value=f"{nb_clients:,}",
        help="Nombre de clients uniques (Customer ID) dans le périmètre filtré."
    )

    col2.metric(
        label="CA total (période filtrée)",
        value=f"{total_revenue:,.0f}",
        help=(
            "Somme du chiffre d'affaires sur le périmètre filtré.\n"
            "Le traitement des retours dépend du mode choisi (inclure / exclure / neutraliser)."
        )
    )

    col3.metric(
        label="Clients avec RFM calculé",
        value=f"{nb_rfm_clients:,}",
        help=(
            "Nombre de clients pour lesquels un score RFM (Recency–Frequency–Monetary) "
            "a été calculé sur le périmètre filtré."
        )
    )

    col4.metric(
        label="CLV baseline (90 jours)",
        value=f"{clv_90:,.0f}",
        help=(
            "CLV empirique : CA moyen par client sur ses 90 premiers jours après sa première facture.\n"
            "Exemple : un client qui dépense 50€, 100€ et 150€ dans les 90 premiers jours "
            "a une CLV(90j) = 300€."
        )
    )

    col5.metric(
        label="CA moyen à l’âge 1 de cohorte",
        value=f"{ca_age1:,.0f}",
        help=(
            "CA moyen par client au mois d’acquisition (âge 1 de cohorte), "
            "moyenné sur toutes les cohortes.\n"
            "Permet de comparer la qualité des cohortes dès le premier mois."
        )
    )

    st.markdown("---")

    st.subheader("⭐ North Star Metric")
    st.metric(
        label="CA moyen par client actif",
        value=f"{north_star:,.0f}",
        help=(
            "North Star : CA total sur le périmètre filtré / nombre de clients actifs.\n"
            "C'est l'indicateur central de valeur moyenne créée par client."
        )
    )

    st.markdown("### Notes méthodologiques")
    st.caption(
        "- Les clients sans 'Customer ID' sont exclus dans les fonctions utilitaires.\n"
        "- Les retours produits sont traités selon le mode choisi (inclure, exclure, neutraliser).\n"
        "- Les cohortes sont basées sur la première date de facture du client (mensuelle).\n"
        "- CLV baseline et CA/âge de cohorte sont des approximations empiriques pour le diagnostic."
    )


if __name__ == "__main__":
    main()
