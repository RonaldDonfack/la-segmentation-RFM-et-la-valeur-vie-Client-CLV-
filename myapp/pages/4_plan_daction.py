"""
Page Streamlit : Plan d'Action
Génération de listes activables et exports pour passage à l'exécution
"""

import streamlit as st
import sys
import os

# Ajouter le dossier parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filters import load_data, apply_filters
from utils import compute_rfm
from plan_action import render_plan_action_page

# Configuration de la page
st.set_page_config(
    page_title="Plan d'Action",
    page_icon="📋",
    layout="wide"
)

# Chargement des données (avec cache)
@st.cache_data
def load_cached_data():
    return load_data()

# Chargement
df = load_cached_data()

# Application des filtres (sidebar)
df_filtered, filters_summary, badge = apply_filters(df)

# Calcul RFM
rfm_df = compute_rfm(df_filtered, prefer_neutralized=True)

# Rendu de la page Plan d'Action
render_plan_action_page(df_filtered, rfm_df, filters_summary, badge)