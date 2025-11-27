# app/app_scenario.py

import streamlit as st
import pandas as pd
import numpy as np

from filters import load_data, apply_filters
from utils import compute_baseline_and_scenario, clv_closed_form


def page_scenarios():
    st.title("📌 Scénarios : marge, rétention, CLV & CA")

    st.write(
        """
Cette page permet de **tester des scénarios marketing** :

- modifier la marge (remises),
- améliorer ou dégrader la rétention,
- ajuster le taux d'actualisation,

et voir l'impact sur :
- le **CLV moyen** (Customer Lifetime Value),
- le **CA total** du portefeuille clients (en GBP).
"""
    )

    # ===========================
    # 0. Chargement + filtres globaux
    # ===========================
    # -> load_data() vient de filters.py
    df_raw = load_data()

    # -> apply_filters() applique les filtres de la sidebar (période, pays,
    #    type client, seuil commande, retours…)
    df, filters_summary, badge_retour = apply_filters(df_raw)

    # Petit rappel des filtres appliqués
    st.caption(
        f"**Filtres actifs** — Période : {filters_summary['Période']} | "
        f"Pays : {filters_summary['Pays']} | Type client : {filters_summary['Type client']} | "
        f"Seuil commande : {filters_summary['Seuil commande']} | Retours : {badge_retour}"
    )

    if df.empty:
        st.warning("Aucun client ne correspond aux filtres sélectionnés.")
        return

    # ===========================
    # 1. Paramètres de base
    # ===========================
    st.markdown("## 1. Paramètres de base")

    col1, col2 = st.columns(2)

    with col1:
        base_margin_pct = st.number_input(
            "Marge brute de référence (%)",
            min_value=0.0,
            max_value=100.0,
            value=30.0,
            step=1.0,
        )
        base_r = st.slider(
            "Taux de rétention de base r",
            min_value=0.10,
            max_value=0.95,
            value=0.60,
            step=0.01,
        )

    with col2:
        d = st.slider(
            "Taux d'actualisation d",
            min_value=0.0,
            max_value=0.50,
            value=0.10,
            step=0.01,
        )

    # ===========================
    # 2. Paramètres du scénario
    # ===========================
    st.markdown("## 2. Paramètres du scénario (variation)")

    col3, col4 = st.columns(2)

    with col3:
        delta_margin_pct = st.slider(
            "Variation de marge (points de %)",
            min_value=-30.0,
            max_value=30.0,
            value=-5.0,
            step=1.0,
            help="Exemple : -5 = promotion qui réduit la marge de 5 points.",
        )

    with col4:
        delta_r_pts = st.slider(
            "Variation de rétention (points de %)",
            min_value=-30.0,
            max_value=30.0,
            value=5.0,
            step=1.0,
            help="Exemple : +5 = amélioration de la fidélité de 5 points.",
        )

    # Conversion en [0,1]
    base_margin = base_margin_pct / 100.0
    delta_margin = delta_margin_pct / 100.0
    delta_r = delta_r_pts / 100.0

    st.markdown("---")

    # ===========================
    # 3. Calculs + affichage
    # ===========================
    if st.button("Calculer le scénario"):
        # compute_baseline_and_scenario vient de utils.py
        results = compute_baseline_and_scenario(
            df,
            base_margin=base_margin,
            base_r=base_r,
            d=d,
            delta_margin=delta_margin,
            delta_r=delta_r,
        )

        st.markdown("## 3. Résultats numériques")

        col_a, col_b = st.columns(2)

        with col_a:
            st.metric("CLV moyen baseline", f"{results['clv_base']:,.0f} £")
            st.metric(
                "CLV moyen scénario",
                f"{results['clv_scenario']:,.0f} £",
                delta=f"{results['clv_scenario'] - results['clv_base']:,.0f} £",
            )

        with col_b:
            st.metric("CA total baseline", f"{results['ca_base']:,.0f} £")
            st.metric(
                "CA total scénario",
                f"{results['ca_scenario']:,.0f} £",
                delta=f"{results['ca_scenario'] - results['ca_base']:,.0f} £",
            )

        # ---------- A. Marge en GBP ----------
        st.markdown("### 🔢 Marge moyenne (en £)")

        margin_base_gbp = results["clv_base"] * base_margin
        margin_scenario_gbp = results["clv_scenario"] * (base_margin + delta_margin)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Marge baseline", f"{margin_base_gbp:,.0f} £")
        with col_m2:
            st.metric(
                "Marge scénario",
                f"{margin_scenario_gbp:,.0f} £",
                delta=f"{margin_scenario_gbp - margin_base_gbp:,.0f} £",
            )

        # ---------- C. Résumé automatique + reco ----------
        st.markdown("### 📝 Résumé automatique")

        delta_clv = results["clv_scenario"] - results["clv_base"]
        delta_ca = results["ca_scenario"] - results["ca_base"]

        # Texte de recommandation "type conseiller"
        if delta_clv > 0 and delta_ca > 0:
            reco = (
                "✅ **Scénario favorable** : il améliore à la fois la valeur client "
                "et le chiffre d'affaires. Scénario à **prioriser** ou à tester à grande échelle."
            )
        elif delta_clv > 0 and delta_ca <= 0:
            reco = (
                "⚖️ **CLV en hausse mais CA global en baisse** : scénario intéressant "
                "pour renforcer la fidélité sur des segments précis, mais à déployer "
                "avec prudence pour ne pas dégrader le volume global."
            )
        elif delta_clv <= 0 and delta_ca > 0:
            reco = (
                "📈 **CA en hausse mais CLV en baisse** : bon pour des objectifs de "
                "volume à court terme (soldes, opérations flash), mais attention au "
                "**risque de dégrader la valeur long terme** des clients."
            )
        else:
            reco = (
                "❌ **Scénario défavorable** : il dégrade à la fois le CLV et le CA. "
                "À éviter ou à fortement ajuster (moins de remise, meilleure cible, etc.)."
            )

        resume = f"""
📌 **Résumé du scénario appliqué**

- Variation de marge : **{delta_margin_pct:+.1f} points**
- Variation de rétention : **{delta_r_pts:+.1f} points**
- Impact sur le CLV : **{delta_clv:+,.0f} £**
- Impact sur le CA total : **{delta_ca:+,.0f} £**

👉 Cela signifie que votre action marketing entraîne un changement  
de **{delta_clv:+,.0f} £ par client**, soit un effet global de  
**{delta_ca:+,.0f} £** sur l’ensemble du portefeuille filtré.

**Recommandation :** {reco}
"""

        st.info(resume)

        st.caption(
            f"Nombre de clients pris en compte : {results['n_customers']:,} – "
            f"Revenu moyen par client observé : {results['avg_rev']:,.0f} £."
        )

        # ===========================
        # 4. Comparaison baseline vs scénario (barres)
        # ===========================
        st.markdown("## 4. Comparaison baseline vs scénario (barres)")

        clv_df = (
            pd.DataFrame(
                {
                    "Scénario": ["Baseline", "Scénario"],
                    "CLV": [results["clv_base"], results["clv_scenario"]],
                }
            )
            .set_index("Scénario")
        )

        ca_df = (
            pd.DataFrame(
                {
                    "Scénario": ["Baseline", "Scénario"],
                    "CA": [results["ca_base"], results["ca_scenario"]],
                }
            )
            .set_index("Scénario")
        )

        col_c, col_d = st.columns(2)
        with col_c:
            st.subheader("CLV moyen")
            st.bar_chart(clv_df)
        with col_d:
            st.subheader("CA total")
            st.bar_chart(ca_df)

        # ===========================
        # 5. Sensibilité : CLV en fonction de r
        # ===========================
        st.markdown("## 5. Sensibilité : CLV en fonction de la rétention r")

        r_min = max(0.05, base_r - 0.20)
        r_max = min(0.95, base_r + 0.20)
        r_values = np.linspace(r_min, r_max, 25)

        clv_values_r = [
            clv_closed_form(results["avg_rev"], base_margin, r, d)
            for r in r_values
        ]

        sens_r_df = (
            pd.DataFrame({"r": r_values, "CLV": clv_values_r})
            .set_index("r")
        )
        st.line_chart(sens_r_df)

        # ===========================
        # 6. Sensibilité : CLV en fonction de la marge
        # ===========================
        st.markdown("## 6. Sensibilité : CLV en fonction de la marge (%)")

        m_min = max(0.01, base_margin - 0.15)
        m_max = min(0.90, base_margin + 0.15)
        m_values = np.linspace(m_min, m_max, 25)

        clv_values_m = [
            clv_closed_form(results["avg_rev"], m, base_r, d)
            for m in m_values
        ]

        sens_m_df = (
            pd.DataFrame({"Marge": m_values, "CLV": clv_values_m})
            .set_index("Marge")
        )
        st.line_chart(sens_m_df)


def main():
    page_scenarios()


if __name__ == "__main__":
    main()
