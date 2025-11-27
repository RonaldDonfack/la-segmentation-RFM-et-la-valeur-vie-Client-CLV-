import streamlit as st
import plotly.express as px
import pandas as pd

from filters import load_data, apply_filters, build_retention_matrix

st.set_page_config(page_title="Cohortes", layout="wide")

def main():
    st.title("📈 Cohortes — Diagnostiquer")

    # 1) Chargement + filtres
    raw_df = load_data()
    df = apply_filters(raw_df)

    # =====================================================
    # 1. HEATMAP DE RÉTENTION PAR COHORTE
    # =====================================================
    st.subheader("Heatmap de rétention par cohorte")

    retention = build_retention_matrix(df) * 100  # en %

    st.caption(
        "Chaque cellule = **% de clients de la cohorte encore actifs** "
        "à un âge de cohorte donné. (n = effectif de la cohorte)"
    )

    fig_heatmap = px.imshow(
        retention,
        labels={
            "x": "Âge de cohorte (mois)",
            "y": "Mois de cohorte",
            "color": "Rétention (%)"
        },
        text_auto=".1f",
        aspect="auto"
    )
    fig_heatmap.update_layout(
        xaxis_title="Âge de cohorte (mois depuis l’acquisition)",
        yaxis_title="Mois d’acquisition (cohorte)",
        coloraxis_colorbar=dict(title="Rétention (%)")
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("---")

    # =====================================================
    # 2. COURBES DE DENSITÉ DE CA PAR ÂGE DE COHORTE
    # =====================================================
    st.subheader("Densité de CA par âge de cohorte")

    # CA par client et par âge de cohorte
    revenue_age_client = (
        df.groupby(["CustomerID", "CohortIndex"])["Revenue"]
        .sum()
        .reset_index()
    )

    if revenue_age_client.empty:
        st.warning("Pas assez de données pour calculer les densités de CA.")
    else:
        min_age = int(revenue_age_client["CohortIndex"].min())
        max_age = int(revenue_age_client["CohortIndex"].max())

        age_selected = st.slider(
            "Âge de cohorte à analyser (en mois)",
            min_value=min_age,
            max_value=max_age,
            value=min_age
        )

        data_age = revenue_age_client[revenue_age_client["CohortIndex"] == age_selected]

        st.caption(
            f"Distribution du **CA par client** pour l’âge de cohorte "
            f"**{age_selected} mois** (n = {data_age['CustomerID'].nunique():,} clients)."
        )

        fig_density = px.histogram(
            data_age,
            x="Revenue",
            nbins=30,
            histnorm="probability density",
            marginal="box",
            labels={"Revenue": "CA par client à cet âge"},
        )
        fig_density.update_layout(
            xaxis_title="CA par client",
            yaxis_title="Densité"
        )
        st.plotly_chart(fig_density, use_container_width=True)

    st.markdown("---")

    # =====================================================
    # 3. FOCUS SUR UNE COHORTE : CA MOYEN PAR ÂGE
    # =====================================================
    st.subheader("Focus sur une cohorte : CA moyen par âge de cohorte")

    # CA total par cohorte & âge
    cohort_revenue = (
        df.groupby(["CohortMonth", "CohortIndex"])["Revenue"]
        .sum()
        .reset_index()
    )

    # Taille des cohortes
    cohort_sizes = (
        df.groupby("CohortMonth")["CustomerID"]
        .nunique()
        .rename("CohortSize")
    )

    cohort_revenue = cohort_revenue.merge(
        cohort_sizes, on="CohortMonth", how="left"
    )
    cohort_revenue["Revenue_per_customer"] = (
        cohort_revenue["Revenue"] / cohort_revenue["CohortSize"]
    )
    cohort_revenue["CohortLabel"] = cohort_revenue["CohortMonth"].dt.strftime("%Y-%m")

    if cohort_revenue.empty:
        st.warning("Pas assez de données pour afficher le focus par cohorte.")
        return

    cohort_options = sorted(cohort_revenue["CohortLabel"].unique().tolist())
    selected_cohort = st.selectbox(
        "Sélectionner une cohorte à analyser",
        options=cohort_options
    )

    focus_data = cohort_revenue[cohort_revenue["CohortLabel"] == selected_cohort]
    n_clients = int(
        focus_data["CohortSize"].iloc[0]
    )

    st.caption(
        f"Évolution du **CA moyen par client** pour la cohorte **{selected_cohort}** "
        f"(n = {n_clients:,} clients)."
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

    # Tableau récap des effectifs
    st.markdown("### Effectifs par cohorte")
    effectifs = cohort_sizes.reset_index()
    effectifs["CohortLabel"] = effectifs["CohortMonth"].dt.strftime("%Y-%m")
    st.dataframe(
        effectifs[["CohortLabel", "CohortSize"]]
        .rename(columns={"CohortSize": "n_clients"})
        .sort_values("CohortLabel"),
        use_container_width=True
    )


if __name__ == "__main__":
    main()

