import streamlit as st
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

from utils import load_final_data


@st.cache_data
def load_data():
    """
    Chargement unique du dataset pour toute l appli.

    On part de utils.load_final_data, puis on ajoute les colonnes
    nécessaires aux filtres globaux:
      - is_return
      - ClientType
      - InvoiceRevenue
    """
    df = load_final_data()

    # Sécurité, au cas où
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    if "Revenue" not in df.columns and {"Quantity", "Price"}.issubset(df.columns):
        df["Revenue"] = df["Quantity"] * df["Price"]

    # Flag retours
    df["is_return"] = (df["Quantity"] < 0) | (df["Invoice"].astype(str).str.startswith("C"))

    # Typologie client
    client_stats = df.groupby("Customer ID").agg(
        total_qty=("Quantity", "sum"),
        total_rev=("Revenue", "sum")
    )
    seuil_90 = client_stats["total_qty"].quantile(0.90)
    client_stats["ClientType"] = np.where(
        client_stats["total_qty"] >= seuil_90,
        "Grossiste_like",
        "Detaillant_like"
    )
    df = df.merge(
        client_stats["ClientType"],
        left_on="Customer ID",
        right_index=True,
        how="left"
    )

    # Revenu par facture
    df["InvoiceRevenue"] = df.groupby("Invoice")["Revenue"].transform("sum")

    return df


def apply_filters(df):
    """
    Panneau de filtres globaux utilisé par Overview, Cohortes, etc.

    Retourne:
      df_filtered, filters_summary, badge
    """
    st.sidebar.title("Filtres")

    # Période min et max
    min_date = df["InvoiceDate"].min().date()
    max_date = df["InvoiceDate"].max().date()

    mode_periode = st.sidebar.radio(
        "Mode de sélection de la période",
        ["Fenêtre glissante", "Période personnalisée"]
    )

    if mode_periode == "Fenêtre glissante":
        nb_mois = st.sidebar.slider("Fenêtre (mois)", 1, 24, 6)
        end_date = max_date
        start_date = (pd.to_datetime(end_date) - relativedelta(months=nb_mois)).date()
    else:
        start_date, end_date = st.sidebar.date_input(
            "Période personnalisée",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

    mask_date = (
        (df["InvoiceDate"].dt.date >= start_date)
        & (df["InvoiceDate"].dt.date <= end_date)
    )
    df_filtered = df[mask_date].copy()

    # Unité de temps (utile pour les pages qui en ont besoin)
    time_unit = st.sidebar.selectbox("Unité de temps", ["Mois", "Trimestre"])

    # Filtre pays
    available_countries = sorted(df["Country"].dropna().unique())
    selected_countries = st.sidebar.multiselect(
        "Pays",
        options=available_countries,
        default=available_countries
    )
    if selected_countries:
        df_filtered = df_filtered[df_filtered["Country"].isin(selected_countries)]
    else:
        df_filtered = df_filtered.iloc[0:0]

    # Type client
    types_client = sorted(df["ClientType"].dropna().unique())
    selected_types = st.sidebar.multiselect(
        "Type client",
        options=types_client,
        default=types_client
    )
    if selected_types:
        df_filtered = df_filtered[df_filtered["ClientType"].isin(selected_types)]
    else:
        df_filtered = df_filtered.iloc[0:0]

    # Seuil de commande
    seuil_commande = st.sidebar.slider(
        "Seuil minimum par facture (GBP)",
        float(df["InvoiceRevenue"].min()),
        float(df["InvoiceRevenue"].max()),
        float(df["InvoiceRevenue"].quantile(0.25))
    )
    df_filtered = df_filtered[df_filtered["InvoiceRevenue"] >= seuil_commande]

    # Gestion des retours
    retour_mode = st.sidebar.radio(
        "Mode retours",
        ["Inclure", "Exclure", "Neutraliser"]
    )

    badge = ""

    if retour_mode == "Exclure":
        df_filtered = df_filtered[~df_filtered["is_return"]]
        badge = "retours exclus"
    elif retour_mode == "Neutraliser":
        df_filtered = df_filtered.copy()
        df_filtered["Revenue"] = np.where(
            df_filtered["is_return"],
            0,
            df_filtered["Revenue"]
        )
        badge = "retours neutralisés"

    filters_summary = {
        "Période": f"{start_date} → {end_date}",
        "Unité de temps": time_unit,
        "Pays": ", ".join(selected_countries),
        "Type client": ", ".join(selected_types),
        "Seuil commande": seuil_commande,
        "Retours": retour_mode,
    }

    return df_filtered, filters_summary, badge
