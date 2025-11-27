

import streamlit as st
import pandas as pd

# Import des nouveaux filtres + utils
from filters import load_data, apply_filters
from utils import (
    compute_global_kpis,
    compute_rfm,
    compute_baseline_and_scenario,
)

 
# CONFIG
 
st.set_page_config(
    page_title="RFM / CLV Diagnostic & Simulator",
    layout="wide"
)

st.title("RFM & CLV Diagnostic — Streamlit App")
st.markdown(
    "Une application pour diagnostiquer, prioriser et simuler l'impact CRM "
    "(rétention, CLV, CA)."
)

 
# 1) Chargement DATA CENTRALISÉ
 
df_raw = load_data()

if df_raw is None or df_raw.empty:
    st.error("❌ Impossible de charger les données.")
    st.stop()

 
# 2) Application des filtres globaux (sidebar)
 
df, filters_summary, badge = apply_filters(df_raw)

if df.empty:
    st.warning("⚠️ Aucun enregistrement après application des filtres.")
    st.stop()

 
# 3) RÉCAP DES FILTRES ACTIVÉS
 
st.subheader("🎛️ Filtres actifs")

with st.expander("Voir les filtres appliqués", expanded=True):

    # Affiche un résumé propre
    for key, value in filters_summary.items():
        st.markdown(f"**{key}** : {value}")

    # Badge
    if badge:
        st.success(f"**Badge actif : {badge}**")

 
# 4) KPIs GLOBAUX (après filtres)
 
st.subheader("📌 KPIs globaux")

kpis = compute_global_kpis(df, prefer_neutralized=True)

col1, col2, col3 = st.columns(3)
col1.metric("Nombre de clients", f"{kpis['nb_clients']:,}")
col2.metric("Nombre de factures", f"{kpis['nb_factures']:,}")
col3.metric("CA total", f"{kpis['total_revenue']:,.2f} GBP")

col4, col5 = st.columns(2)
col4.metric("AOV (panier moyen)", f"{kpis['avg_order_value']:,.2f} GBP")
col5.metric("CA / Client", f"{kpis['avg_revenue_per_customer']:,.2f} GBP")

 
# 5) RFM ANALYSIS

# ------------------- RFM ANALYSIS -------------------
st.header("📊 Analyse RFM")

# snapshot_date : on prend la fin de la fenêtre filtrée comme référence
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

# compute_rfm renvoie un DataFrame indexed by Customer ID (selon ton utils)
rfm_df = compute_rfm(df, snapshot_date=snapshot_date, prefer_neutralized=True, n_bins=5)

# Assurer que Customer ID est une colonne
if rfm_df.index.name == "Customer ID":
    rfm_df = rfm_df.reset_index()

# Créer un RFMCode s'il n'existe pas (ex: '543')
if "RFMCode" not in rfm_df.columns:
    rfm_df["RFMCode"] = (
        rfm_df["R_score"].astype(str)
        + rfm_df["F_score"].astype(str)
        + rfm_df["M_score"].astype(str)
    )

# Agrégation pour la table de segments (par code RFM et label Segment)
seg_table = (
    rfm_df.groupby(["RFMCode", "Segment"], dropna=False)
    .agg(
        n_customers=("Customer ID", "count"),
        CA=("Monetary", "sum"),
        avg_basket=("Monetary", lambda x: x.sum() / max(1, x.count())),
    )
    .reset_index()
    .rename(columns={"Segment": "SegmentLabel"})
)

# Marge estimée : utiliser margin_pct si déjà défini dans le script sinon défaut 30%
try:
    margin_pct_local = float(margin_pct)
except Exception:
    margin_pct_local = 30.0

seg_table["marge_estimee"] = seg_table["CA"] * (margin_pct_local / 100.0)
seg_table["panier_moyen"] = seg_table["avg_basket"]

# Priorités d'activation : mappe tes labels produits par compute_rfm
priority_map = {"High_value": 1, "Medium_value": 2, "Low_value": 3}
seg_table["priority_activation"] = seg_table["SegmentLabel"].map(priority_map).fillna(4).astype(int)

# Tri pour affichage (priorité asc, puis taille desc)
seg_table_display = seg_table.sort_values(["priority_activation", "n_customers"], ascending=[True, False])

# Affichage propre
st.subheader("Table des segments RFM")
st.dataframe(
    seg_table_display.rename(
        columns={
            "RFMCode": "Code",
            "SegmentLabel": "Label",
            "n_customers": "N",
            "CA": "CA",
            "marge_estimee": "Marge_estimee",
            "panier_moyen": "Panier_moyen",
            "priority_activation": "Priorite",
        }
    ).style.format({"CA": "{:,.0f}", "Marge_estimee": "{:,.0f}", "Panier_moyen": "{:,.2f}"})
)

# Export CSV des segments
csv_seg = seg_table_display.to_csv(index=False).encode("utf-8")
fn_snapshot = snapshot_date.date().isoformat()
st.download_button(
    "Exporter table Segments (CSV)",
    data=csv_seg,
    file_name=f"segments_rfm_{fn_snapshot}.csv",
    mime="text/csv",
)

# Affichage / export liste clients par RFM code sélectionné
selected_code = st.selectbox(
    "Voir clients d'un segment (RFM Code)",
    options=[None] + seg_table_display["RFMCode"].tolist(),
)
if selected_code:
    subset = rfm_df[rfm_df["RFMCode"] == selected_code].copy()
    st.write(f"Clients dans {selected_code}: {subset.shape[0]}")
    st.dataframe(subset.head(200))

    csv_cust = subset.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Exporter liste clients (CSV)",
        data=csv_cust,
        file_name=f"clients_{selected_code}_{fn_snapshot}.csv",
        mime="text/csv",
    )

# Optionnel : montrer un extrait du DataFrame RFM complet (utile pour debug)
with st.expander("Aperçu table RFM (extrait)", expanded=False):
    st.dataframe(rfm_df.head(200))
