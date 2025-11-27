
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from filters import load_data, apply_filters
from utils import compute_monthly_cohort_retention

st.set_page_config(page_title="Cohortes — Diagnostiquer", layout="wide")


def _add_cohort_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduit la logique de compute_monthly_cohort_retention pour annoter
    chaque ligne avec InvoiceMonth, CohortMonth et CohortIndex.
    Utile pour calculer les densités de CA par âge.
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

    return data


def main():
    st.title("📈 Cohortes — Diagnostiquer la rétention")

    # ===== 1. Chargement & filtres =====
    df_all = load_data()
    df_filtered, filters_summary, badge = apply_filters(df_all)

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

    # ===== 2. Heatmap de rétention =====
    st.subheader("Heatmap de rétention par cohorte")

    retention_table, cohort_sizes = compute_monthly_cohort_retention(df_filtered)

    # On convertit en % pour affichage
    retention_pct = retention_table * 100.0

    # Pour un affichage plus propre, on convertit l'index (Period) en str
    retention_pct_display = retention_pct.copy()
    retention_pct_display.index = retention_pct_display.index.astype(str)

    fig_heatmap = px.imshow(
        retention_pct_display,
        labels=dict(
            x="Âge de cohorte (mois)",
            y="Mois de cohorte",
            color="Rétention (%)"
        ),
        text_auto=".1f",
        aspect="auto"
    )
    fig_heatmap.update_layout(
        xaxis_title="Âge de cohorte (mois depuis l’acquisition)",
        yaxis_title="Mois de cohorte (cohorte d’acquisition)",
        coloraxis_colorbar=dict(title="Rétention (%)")
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.caption(
        "Chaque cellule représente le **pourcentage de clients de la cohorte initiale** "
        "encore actifs à l’âge de cohorte donné. Entre parenthèses : taille initiale de la cohorte (n)."
    )

    # Affichage des effectifs n par cohorte
    effectifs_df = cohort_sizes.reset_index()
    effectifs_df.columns = ["CohortMonth", "n_clients"]
    effectifs_df["CohortMonth"] = effectifs_df["CohortMonth"].astype(str)
    st.markdown("#### Effectifs par cohorte (n)")
    st.dataframe(effectifs_df, use_container_width=True)

    st.markdown("---")

    # ===== 3. Courbes de densité de CA par âge de cohorte =====
    st.subheader("Densité du CA par âge de cohorte")

    data_cohort = _add_cohort_index(df_filtered)

    # CA par client et par âge de cohorte
    revenue_age_client = (
        data_cohort.groupby(["Customer ID", "CohortIndex"])["Revenue"]
        .sum()
        .reset_index()
    )

    if revenue_age_client.empty:
        st.warning("Pas assez de données pour calculer les densités de CA par âge de cohorte.")
    else:
        min_age = int(revenue_age_client["CohortIndex"].min())
        max_age = int(revenue_age_client["CohortIndex"].max())

        age_selected = st.slider(
            "Âge de cohorte (en mois) à analyser",
            min_value=min_age,
            max_value=max_age,
            value=min_age
        )

        data_age = revenue_age_client[revenue_age_client["CohortIndex"] == age_selected]

        n_clients_age = data_age["Customer ID"].nunique()
        st.caption(
            f"Densité du CA **par client** pour l’âge de cohorte **{age_selected} mois** "
            f"(n = {n_clients_age:,} clients)."
        )

        fig_density = px.histogram(
            data_age,
            x="Revenue",
            nbins=30,
            histnorm="probability density",
            marginal="box",
            labels={"Revenue": "CA par client"},
        )
        fig_density.update_layout(
            xaxis_title="CA par client à cet âge de cohorte",
            yaxis_title="Densité"
        )
        st.plotly_chart(fig_density, use_container_width=True)

    st.markdown("---")

    # ===== 4. Focus sur une cohorte : CA moyen par âge =====
    st.subheader("Focus sur une cohorte : CA moyen par âge de cohorte")

    # CA total par cohorte & âge
    cohort_revenue = (
        data_cohort.groupby(["CohortMonth", "CohortIndex"])["Revenue"]
        .sum()
        .reset_index()
    )
    cohort_revenue["CohortMonth"] = cohort_revenue["CohortMonth"].astype("period[M]")

    # On réutilise cohort_sizes (taille initiale de cohorte)
    cohort_sizes_series = cohort_sizes.copy()
    cohort_sizes_series.index = cohort_sizes_series.index.astype("period[M]")
    cohort_sizes_series = cohort_sizes_series.rename("CohortSize")

    cohort_revenue = cohort_revenue.merge(
        cohort_sizes_series,
        left_on="CohortMonth",
        right_index=True,
        how="left"
    )

    cohort_revenue["Revenue_per_customer"] = (
        cohort_revenue["Revenue"] / cohort_revenue["CohortSize"]
    )

    cohort_revenue["CohortLabel"] = cohort_revenue["CohortMonth"].astype(str)

    cohort_options = sorted(cohort_revenue["CohortLabel"].unique().tolist())
    if not cohort_options:
        st.warning("Impossible de calculer le focus cohorte (pas de cohorte valide).")
        return

    selected_cohort = st.selectbox(
        "Sélectionner une cohorte (mois d’acquisition)",
        options=cohort_options
    )

    focus_data = cohort_revenue[cohort_revenue["CohortLabel"] == selected_cohort]
    n_clients_focus = int(focus_data["CohortSize"].iloc[0]) if not focus_data.empty else 0

    st.caption(
        f"Évolution du **CA moyen par client** pour la cohorte **{selected_cohort}** "
        f"(n = {n_clients_focus:,} clients initiaux)."
    )

    fig_line = px.line(
        focus_data,
        x="CohortIndex",
        y="Revenue_per_customer",
        markers=True,
        labels={
            "CohortIndex": "Âge de cohorte (mois)",
            "Revenue_per_customer": "CA moyen par client"
        }
    )
    st.plotly_chart(fig_line, use_container_width=True)


if __name__ == "__main__":
    main()
