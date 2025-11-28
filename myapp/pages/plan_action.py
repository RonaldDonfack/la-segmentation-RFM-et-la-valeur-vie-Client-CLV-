import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io
import base64


from utils import compute_rfm, compute_global_kpis
from filters import apply_filters


def create_actionable_list(df, rfm_df, filters_summary):
    """
    Crée une liste activable avec CustomerID, segment RFM et métriques clés
    
    Args:
        df: DataFrame filtré des transactions
        rfm_df: DataFrame RFM avec scores et segments
        filters_summary: Dict des filtres appliqués
        
    Returns:
        DataFrame prêt à l'export avec les colonnes actionnables
    """
    
    # Enrichissement des données client
    client_metrics = df.groupby("Customer ID").agg({
        "Invoice": "nunique",
        "Revenue": "sum",
        "InvoiceDate": ["min", "max"],
        "Country": "first",
        "ClientType": "first"
    }).reset_index()
    
    # Aplatissement des colonnes multi-index
    client_metrics.columns = [
        "Customer ID", 
        "Nb_Commandes", 
        "CA_Total",
        "Date_Premier_Achat",
        "Date_Dernier_Achat", 
        "Pays",
        "Type_Client"
    ]
    
    # Calcul de métriques additionnelles
    client_metrics["Panier_Moyen"] = (
        client_metrics["CA_Total"] / client_metrics["Nb_Commandes"]
    )
    client_metrics["Anciennete_Jours"] = (
        client_metrics["Date_Dernier_Achat"] - client_metrics["Date_Premier_Achat"]
    ).dt.days
    
    # Fusion avec RFM
    actionable_list = client_metrics.merge(
        rfm_df[["Recency", "Frequency", "Monetary", "R_score", "F_score", "M_score", "Segment"]],
        left_on="Customer ID",
        right_index=True,
        how="left"
    )
    
    # Définition des priorités d'activation
    def define_priority(row):
        segment = row["Segment"]
        r_score = row["R_score"]
        f_score = row["F_score"]
        m_score = row["M_score"]
        
        # Champions et High Value
        if segment == "High_value" and r_score >= 4 and f_score >= 4:
            return "P1_RETENTION_VIP"
        
        # At Risk (haute valeur qui décroche)
        elif m_score >= 4 and r_score <= 2:
            return "P1_WINBACK_URGENT"
        
        # Promising (récents mais faible fréquence)
        elif r_score >= 4 and f_score <= 2:
            return "P2_ACTIVATION"
        
        # Medium value stable
        elif segment == "Medium_value" and r_score >= 3:
            return "P2_UPSELL"
        
        # Low value
        elif segment == "Low_value":
            return "P3_LOW_PRIORITY"
        
        else:
            return "P3_MONITORING"
    
    actionable_list["Priorite_Action"] = actionable_list.apply(define_priority, axis=1)
    
    # Recommandations d'action par priorité
    def define_action_recommandee(priorite):
        actions = {
            "P1_RETENTION_VIP": "Programme fidélité premium, offres exclusives, contact direct",
            "P1_WINBACK_URGENT": "Campagne réactivation, remise ciblée, enquête satisfaction",
            "P2_ACTIVATION": "Incitation 2ème achat, cross-sell, communication régulière",
            "P2_UPSELL": "Recommandations produits premium, bundles, promotions paliers",
            "P3_LOW_PRIORITY": "Communication automatisée low-cost, remarketing passif",
            "P3_MONITORING": "Observation comportement, actions génériques"
        }
        return actions.get(priorite, "À définir")
    
    actionable_list["Action_Recommandee"] = actionable_list["Priorite_Action"].apply(
        define_action_recommandee
    )
    
    # Tri par priorité puis par valeur monétaire
    priority_order = {
        "P1_RETENTION_VIP": 1,
        "P1_WINBACK_URGENT": 2,
        "P2_ACTIVATION": 3,
        "P2_UPSELL": 4,
        "P3_LOW_PRIORITY": 5,
        "P3_MONITORING": 6
    }
    actionable_list["_priority_rank"] = actionable_list["Priorite_Action"].map(priority_order)
    actionable_list = actionable_list.sort_values(
        ["_priority_rank", "CA_Total"], 
        ascending=[True, False]
    ).drop("_priority_rank", axis=1)
    
    # Ajout métadonnées export
    actionable_list["Date_Export"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    actionable_list["Periode_Analyse"] = filters_summary.get("Période", "N/A")
    actionable_list["Filtres_Appliques"] = str(filters_summary)
    
    # Réorganisation des colonnes pour lisibilité
    cols_order = [
        "Customer ID",
        "Priorite_Action",
        "Action_Recommandee",
        "Segment",
        "R_score", "F_score", "M_score",
        "CA_Total",
        "Nb_Commandes",
        "Panier_Moyen",
        "Recency",
        "Frequency",
        "Monetary",
        "Date_Premier_Achat",
        "Date_Dernier_Achat",
        "Anciennete_Jours",
        "Pays",
        "Type_Client",
        "Date_Export",
        "Periode_Analyse"
    ]
    
    actionable_list = actionable_list[cols_order]
    
    return actionable_list


def create_summary_charts(df, rfm_df, actionable_list, filters_summary):
    """
    Crée les graphiques de synthèse pour export PNG
    
    Returns:
        Dict de figures Plotly prêtes à l'export
    """
    charts = {}
    
    # === GRAPHIQUE 1: Distribution des priorités ===
    priority_counts = actionable_list["Priorite_Action"].value_counts().sort_index()
    
    fig_priorities = go.Figure(data=[
        go.Bar(
            x=priority_counts.index,
            y=priority_counts.values,
            text=priority_counts.values,
            textposition="auto",
            marker=dict(
                color=priority_counts.values,
                colorscale="RdYlGn_r",
                showscale=False
            )
        )
    ])
    
    fig_priorities.update_layout(
        title=f"Distribution des Priorités d'Action<br><sub>Période: {filters_summary.get('Période', 'N/A')}</sub>",
        xaxis_title="Priorité",
        yaxis_title="Nombre de Clients",
        template="plotly_white",
        height=400,
        showlegend=False
    )
    
    charts["priorities"] = fig_priorities
    
    # === GRAPHIQUE 2: CA par segment RFM ===
    segment_ca = actionable_list.groupby("Segment").agg({
        "CA_Total": "sum",
        "Customer ID": "count"
    }).reset_index()
    segment_ca.columns = ["Segment", "CA_Total", "Nb_Clients"]
    
    fig_segments = go.Figure()
    
    fig_segments.add_trace(go.Bar(
        name="CA Total",
        x=segment_ca["Segment"],
        y=segment_ca["CA_Total"],
        text=[f"£{val:,.0f}" for val in segment_ca["CA_Total"]],
        textposition="auto",
        marker_color="steelblue"
    ))
    
    fig_segments.update_layout(
        title=f"Chiffre d'Affaires par Segment RFM<br><sub>Période: {filters_summary.get('Période', 'N/A')}</sub>",
        xaxis_title="Segment RFM",
        yaxis_title="CA Total (£)",
        template="plotly_white",
        height=400
    )
    
    charts["segments_ca"] = fig_segments
    
    # === GRAPHIQUE 3: Top 20 clients par valeur ===
    top_clients = actionable_list.nlargest(20, "CA_Total")
    
    fig_top_clients = go.Figure(data=[
        go.Bar(
            x=top_clients["Customer ID"].astype(str),
            y=top_clients["CA_Total"],
            text=[f"£{val:,.0f}" for val in top_clients["CA_Total"]],
            textposition="auto",
            marker=dict(
                color=top_clients["CA_Total"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="CA (£)")
            ),
            hovertemplate=(
                "<b>Client:</b> %{x}<br>"
                "<b>CA:</b> £%{y:,.0f}<br>"
                "<b>Segment:</b> %{customdata[0]}<br>"
                "<b>Priorité:</b> %{customdata[1]}<extra></extra>"
            ),
            customdata=top_clients[["Segment", "Priorite_Action"]].values
        )
    ])
    
    fig_top_clients.update_layout(
        title=f"Top 20 Clients par Chiffre d'Affaires<br><sub>Période: {filters_summary.get('Période', 'N/A')}</sub>",
        xaxis_title="Customer ID",
        yaxis_title="CA Total (£)",
        template="plotly_white",
        height=500,
        xaxis=dict(tickangle=-45)
    )
    
    charts["top_clients"] = fig_top_clients
    
    # === GRAPHIQUE 4: Matrice Priorité × Segment ===
    matrix_data = actionable_list.groupby(
        ["Priorite_Action", "Segment"]
    ).size().reset_index(name="Count")
    
    matrix_pivot = matrix_data.pivot(
        index="Priorite_Action",
        columns="Segment",
        values="Count"
    ).fillna(0)
    
    fig_matrix = go.Figure(data=go.Heatmap(
        z=matrix_pivot.values,
        x=matrix_pivot.columns,
        y=matrix_pivot.index,
        colorscale="Blues",
        text=matrix_pivot.values,
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate="Priorité: %{y}<br>Segment: %{x}<br>Clients: %{z}<extra></extra>"
    ))
    
    fig_matrix.update_layout(
        title=f"Matrice Priorité × Segment RFM<br><sub>Période: {filters_summary.get('Période', 'N/A')}</sub>",
        xaxis_title="Segment RFM",
        yaxis_title="Priorité d'Action",
        template="plotly_white",
        height=400
    )
    
    charts["priority_segment_matrix"] = fig_matrix
    
    # === GRAPHIQUE 5: Distribution Recency vs Monetary (scatter) ===
    fig_scatter = px.scatter(
        actionable_list,
        x="Recency",
        y="Monetary",
        color="Priorite_Action",
        size="Nb_Commandes",
        hover_data=["Customer ID", "Segment", "Panier_Moyen"],
        title=f"Recency vs Monetary par Priorité<br><sub>Période: {filters_summary.get('Période', 'N/A')}</sub>",
        labels={
            "Recency": "Recency (jours depuis dernier achat)",
            "Monetary": "Monetary (CA total £)",
            "Priorite_Action": "Priorité"
        },
        template="plotly_white",
        height=500
    )
    
    fig_scatter.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey')))
    
    charts["recency_monetary"] = fig_scatter
    
    return charts


def export_to_csv(df, filename="export.csv"):
    """
    Convertit un DataFrame en CSV téléchargeable
    
    Returns:
        bytes du CSV encodé
    """
    return df.to_csv(index=False).encode('utf-8')


def export_chart_to_png(fig, filename="chart.png", width=1200, height=600):
    """
    Convertit une figure Plotly en PNG
    
    Returns:
        bytes du PNG
    """
    img_bytes = fig.to_image(format="png", width=width, height=height, engine="kaleido")
    return img_bytes


def generate_executive_summary(actionable_list, filters_summary):
    """
    Génère un résumé exécutif textuel pour accompagner les exports
    
    Returns:
        String formaté en Markdown
    """
    total_clients = len(actionable_list)
    
    priority_counts = actionable_list["Priorite_Action"].value_counts()
    p1_clients = priority_counts.get("P1_RETENTION_VIP", 0) + priority_counts.get("P1_WINBACK_URGENT", 0)
    
    total_ca = actionable_list["CA_Total"].sum()
    ca_p1 = actionable_list[
        actionable_list["Priorite_Action"].isin(["P1_RETENTION_VIP", "P1_WINBACK_URGENT"])
    ]["CA_Total"].sum()
    
    pct_p1_clients = (p1_clients / total_clients * 100) if total_clients > 0 else 0
    pct_p1_ca = (ca_p1 / total_ca * 100) if total_ca > 0 else 0
    
    summary = f"""
# 📊 RÉSUMÉ EXÉCUTIF - PLAN D'ACTION

**Date d'export:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Période analysée:** {filters_summary.get('Période', 'N/A')}

---

## 🎯 VUE D'ENSEMBLE

- **Total clients analysés:** {total_clients:,}
- **Chiffre d'affaires total:** £{total_ca:,.2f}

---

## 🚨 PRIORITÉS CRITIQUES (P1)

- **Clients prioritaires (P1):** {p1_clients} ({pct_p1_clients:.1f}% de la base)
- **CA généré par P1:** £{ca_p1:,.2f} ({pct_p1_ca:.1f}% du CA total)

### Répartition P1:
- **Rétention VIP:** {priority_counts.get('P1_RETENTION_VIP', 0)} clients
- **Winback Urgent:** {priority_counts.get('P1_WINBACK_URGENT', 0)} clients

**→ Action requise immédiate sur ces segments**

---

## 📈 DISTRIBUTION DES PRIORITÉS

| Priorité | Nb Clients | % Base | Action Recommandée |
|----------|------------|--------|-------------------|
| P1 - Rétention VIP | {priority_counts.get('P1_RETENTION_VIP', 0)} | {priority_counts.get('P1_RETENTION_VIP', 0)/total_clients*100:.1f}% | Programme fidélité premium |
| P1 - Winback Urgent | {priority_counts.get('P1_WINBACK_URGENT', 0)} | {priority_counts.get('P1_WINBACK_URGENT', 0)/total_clients*100:.1f}% | Campagne réactivation |
| P2 - Activation | {priority_counts.get('P2_ACTIVATION', 0)} | {priority_counts.get('P2_ACTIVATION', 0)/total_clients*100:.1f}% | Incitation 2ème achat |
| P2 - Upsell | {priority_counts.get('P2_UPSELL', 0)} | {priority_counts.get('P2_UPSELL', 0)/total_clients*100:.1f}% | Cross-sell/Upsell |
| P3 - Low Priority | {priority_counts.get('P3_LOW_PRIORITY', 0)} | {priority_counts.get('P3_LOW_PRIORITY', 0)/total_clients*100:.1f}% | Communication automatisée |
| P3 - Monitoring | {priority_counts.get('P3_MONITORING', 0)} | {priority_counts.get('P3_MONITORING', 0)/total_clients*100:.1f}% | Observation passive |

---

## 💡 RECOMMANDATIONS IMMÉDIATES

### 1. Focus Rétention VIP
Clients à haute valeur avec engagement fort. Investissement relationnel prioritaire.

**Actions:**
- Contact direct personnalisé
- Offres exclusives early access
- Programme ambassadeurs

### 2. Campagne Winback Urgent
Clients à forte valeur historique en décrochage. ROI potentiel élevé.

**Actions:**
- Enquête satisfaction
- Remise ciblée conditionnée
- Communication réactivation multi-canal

### 3. Activation Nouveaux Clients
Récents mais faible fréquence. Potentiel conversion en récurrents.

**Actions:**
- Incitation 2ème achat (remise limitée)
- Onboarding renforcé
- Communication pédagogique produits

---

## 📁 FICHIERS EXPORTÉS

1. **liste_activable_complete.csv** - Liste complète avec toutes les métriques
2. **liste_p1_urgent.csv** - Focus sur priorités P1 uniquement
3. **synthese_segments_rfm.csv** - Agrégations par segment
4. **graphiques/** - Visualisations PNG haute définition

---

## ⚙️ FILTRES APPLIQUÉS

{filters_summary}

---

**Note:** Cette analyse est basée sur les données disponibles à la date d'export.  
Les priorités doivent être revues régulièrement (recommandation: hebdomadaire pour P1, mensuel pour P2-P3).
"""
    
    return summary


def render_plan_action_page(df, rfm_df, filters_summary, badge=""):
    """
    Page Streamlit principale pour le Plan d'Action
    """
    st.title("📋 Plan d'Action - Exports & Exécution")
    
    if badge:
        st.info(f"ℹ️ Mode actif: **{badge}**")
    
    st.markdown("""
    Cette page vous permet de **passer du diagnostic à l'action** en exportant:
    - 📊 **Listes activables** (CSV) avec priorités et recommandations
    - 📈 **Visualisations** (PNG) haute définition pour présentations
    - 📄 **Résumé exécutif** pour alignement équipe
    """)
    
    st.markdown("---")
    
    # === GÉNÉRATION LISTE ACTIVABLE ===
    with st.spinner("🔄 Génération de la liste activable..."):
        actionable_list = create_actionable_list(df, rfm_df, filters_summary)
    
    st.success(f"✅ Liste activable générée: **{len(actionable_list)} clients**")
    
    # === APERÇU LISTE ACTIVABLE ===
    st.subheader("👀 Aperçu de la Liste Activable")
    
    col1, col2, col3, col4 = st.columns(4)
    
    priority_counts = actionable_list["Priorite_Action"].value_counts()
    
    with col1:
        p1_count = priority_counts.get("P1_RETENTION_VIP", 0) + priority_counts.get("P1_WINBACK_URGENT", 0)
        st.metric("Clients P1 (Urgent)", p1_count, 
                  delta=f"{p1_count/len(actionable_list)*100:.1f}% base")
    
    with col2:
        p2_count = priority_counts.get("P2_ACTIVATION", 0) + priority_counts.get("P2_UPSELL", 0)
        st.metric("Clients P2 (Important)", p2_count,
                  delta=f"{p2_count/len(actionable_list)*100:.1f}% base")
    
    with col3:
        ca_total = actionable_list["CA_Total"].sum()
        st.metric("CA Total", f"£{ca_total:,.0f}")
    
    with col4:
        panier_moy = actionable_list["Panier_Moyen"].mean()
        st.metric("Panier Moyen", f"£{panier_moy:.2f}")
    
    # Filtres d'affichage
    st.markdown("#### Filtrer l'aperçu")
    col_filter1, col_filter2 = st.columns(2)
    
    with col_filter1:
        priority_filter = st.multiselect(
            "Priorités à afficher",
            options=sorted(actionable_list["Priorite_Action"].unique()),
            default=sorted(actionable_list["Priorite_Action"].unique())
        )
    
    with col_filter2:
        segment_filter = st.multiselect(
            "Segments RFM à afficher",
            options=sorted(actionable_list["Segment"].unique()),
            default=sorted(actionable_list["Segment"].unique())
        )
    
    # Application des filtres d'affichage
    filtered_display = actionable_list[
        (actionable_list["Priorite_Action"].isin(priority_filter)) &
        (actionable_list["Segment"].isin(segment_filter))
    ]
    
    # Affichage tableau
    st.dataframe(
        filtered_display[[
            "Customer ID", "Priorite_Action", "Action_Recommandee",
            "Segment", "CA_Total", "Nb_Commandes", "Panier_Moyen",
            "Recency", "R_score", "F_score", "M_score"
        ]],
        use_container_width=True,
        height=400
    )
    
    st.markdown("---")
    
    # === EXPORTS CSV ===
    st.subheader("📥 Exports CSV")
    
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        st.markdown("**Liste Complète**")
        st.markdown("Tous les clients avec toutes les métriques")
        csv_complete = export_to_csv(actionable_list, "liste_activable_complete.csv")
        st.download_button(
            label="⬇️ Télécharger Liste Complète",
            data=csv_complete,
            file_name=f"liste_activable_complete_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_export2:
        st.markdown("**Priorités P1 Uniquement**")
        st.markdown("Focus sur clients urgents (Rétention VIP + Winback)")
        p1_list = actionable_list[
            actionable_list["Priorite_Action"].isin(["P1_RETENTION_VIP", "P1_WINBACK_URGENT"])
        ]
        csv_p1 = export_to_csv(p1_list, "liste_p1_urgent.csv")
        st.download_button(
            label="⬇️ Télécharger P1 Urgent",
            data=csv_p1,
            file_name=f"liste_p1_urgent_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_export3:
        st.markdown("**Synthèse par Segment**")
        st.markdown("Agrégations pour analyse macro")
        segment_summary = actionable_list.groupby(["Segment", "Priorite_Action"]).agg({
            "Customer ID": "count",
            "CA_Total": "sum",
            "Panier_Moyen": "mean",
            "Nb_Commandes": "sum"
        }).reset_index()
        segment_summary.columns = [
            "Segment", "Priorite", "Nb_Clients", "CA_Total", "Panier_Moyen", "Total_Commandes"
        ]
        csv_summary = export_to_csv(segment_summary, "synthese_segments.csv")
        st.download_button(
            label="⬇️ Télécharger Synthèse",
            data=csv_summary,
            file_name=f"synthese_segments_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # === VISUALISATIONS ===
    st.subheader("📊 Visualisations")
    
    with st.spinner("🎨 Génération des graphiques..."):
        charts = create_summary_charts(df, rfm_df, actionable_list, filters_summary)
    
    # Affichage des graphiques avec export PNG
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Priorités", "Segments CA", "Top Clients", "Matrice", "Recency vs Monetary"
    ])
    
    with tab1:
        st.plotly_chart(charts["priorities"], use_container_width=True)
        png_priorities = export_chart_to_png(charts["priorities"])
        st.download_button(
            label="⬇️ Télécharger PNG",
            data=png_priorities,
            file_name=f"priorities_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
            mime="image/png"
        )
    
    with tab2:
        st.plotly_chart(charts["segments_ca"], use_container_width=True)
        png_segments = export_chart_to_png(charts["segments_ca"])
        st.download_button(
            label="⬇️ Télécharger PNG",
            data=png_segments,
            file_name=f"segments_ca_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
            mime="image/png"
        )
    
    with tab3:
        st.plotly_chart(charts["top_clients"], use_container_width=True)
        png_top = export_chart_to_png(charts["top_clients"])
        st.download_button(
            label="⬇️ Télécharger PNG",
            data=png_top,
            file_name=f"top_clients_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
            mime="image/png"
        )
    
    with tab4:
        st.plotly_chart(charts["priority_segment_matrix"], use_container_width=True)
        png_matrix = export_chart_to_png(charts["priority_segment_matrix"])
        st.download_button(
            label="⬇️ Télécharger PNG",
            data=png_matrix,
            file_name=f"matrice_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
            mime="image/png"
        )
    
    with tab5:
        st.plotly_chart(charts["recency_monetary"], use_container_width=True)
        png_scatter = export_chart_to_png(charts["recency_monetary"])
        st.download_button(
            label="⬇️ Télécharger PNG",
            data=png_scatter,
            file_name=f"recency_monetary_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
            mime="image/png"
        )
    
    st.markdown("---")
    
    # === RÉSUMÉ EXÉCUTIF ===
    st.subheader("📄 Résumé Exécutif")
    
    executive_summary = generate_executive_summary(actionable_list, filters_summary)
    
    with st.expander("📖 Voir le résumé exécutif complet", expanded=False):
        st.markdown(executive_summary)
    
    # Export du résumé en Markdown
    st.download_button(
        label="⬇️ Télécharger Résumé Exécutif (Markdown)",
        data=executive_summary.encode('utf-8'),
        file_name=f"resume_executif_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown"
    )
    
    st.markdown("---")
    
    # === MÉTADONNÉES & TRAÇABILITÉ ===
    with st.expander("🔍 Métadonnées & Traçabilité", expanded=False):
        st.json({
            "date_export": datetime.now().isoformat(),
            "periode_analyse": filters_summary.get("Période", "N/A"),
            "filtres_appliques": filters_summary,
            "nb_clients_total": len(actionable_list),
            "nb_clients_p1": int(priority_counts.get("P1_RETENTION_VIP", 0) + priority_counts.get("P1_WINBACK_URGENT", 0)),
            "ca_total": float(actionable_list["CA_Total"].sum()),
            "mode_retours": badge if badge else "inclus",
            "version_app": "1.0"
        })


# === EXEMPLE D'INTÉGRATION DANS APP PRINCIPALE ===
if __name__ == "__main__":
    st.set_page_config(page_title="Plan d'Action", layout="wide")
    
    # Simulation de données pour test standalone
    st.warning("⚠️ Mode test - Chargez les vraies données depuis l'app principale")
    
    # Exemple d'appel (à adapter selon votre structure)
    # from filters import load_data, apply_filters
    # from utils import compute_rfm
    # 
    # df = load_data()
    # df_filtered, filters_summary, badge = apply_filters(df)
    # rfm_df = compute_rfm(df_filtered)
    # 
    # render_plan_action_page(df_filtered, rfm_df, filters_summary, badge)
