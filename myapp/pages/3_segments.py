import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from datetime import datetime, timedelta

# ------------------------- Helpers -------------------------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    df = pd.read_csv(uploaded_file, parse_dates=["InvoiceDate"], dayfirst=True, low_memory=False)
    expected = ["Invoice","StockCode","Description","Quantity","InvoiceDate","Price","Customer ID","Country","Year","Month","DayOfWeek"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        st.error(f"Fichier manquant colonnes attendues: {missing}")
    df.rename(columns={c:c.strip() for c in df.columns}, inplace=True)
    df["Revenue"] = df["Quantity"] * df["Price"]
    return df

@st.cache_data
def preprocess(df, start_date, end_date, returns_mode):
    d = df.copy()
    d = d[(d["InvoiceDate"] >= start_date) & (d["InvoiceDate"] <= end_date)].copy()
    if returns_mode == "exclude":
        d = d[d["Quantity"] >= 0]
    elif returns_mode == "neutralize":
        d.loc[d["Quantity"] < 0, "Revenue"] = 0
    d["Customer ID"] = d["Customer ID"].fillna("UNKNOWN")
    d["InvoiceMonth"] = d["InvoiceDate"].dt.to_period('M').dt.to_timestamp()
    d["InvoiceQuarter"] = d["InvoiceDate"].dt.to_period('Q').dt.to_timestamp()
    d["InvoiceDay"] = d["InvoiceDate"].dt.date
    return d

@st.cache_data
def compute_kpis(df, analysis_end):
    total_rev = df["Revenue"].sum()
    unique_customers = df["Customer ID"].nunique()
    orders = df["Invoice"].nunique()
    aov = total_rev / orders if orders>0 else 0
    # Retention M+3 (approx): proportion of customers who repurchase within 90 days after their first purchase
    first_purchase = df.groupby("Customer ID")["InvoiceDate"].min().reset_index().rename(columns={"InvoiceDate":"FirstPurchase"})
    merged = first_purchase.merge(df[["Customer ID","InvoiceDate"]], on="Customer ID", how="left")
    merged["months_after_first"] = ((merged["InvoiceDate"] - merged["FirstPurchase"]).dt.days / 30.0)
    cohorts = df.groupby("Customer ID").agg(FirstPurchase=("InvoiceDate","min"), NOrders=("Invoice","nunique"))
    cohorts["RetainedM3"] = cohorts.apply(lambda row: ((df[(df["Customer ID"]==row.name) & (df["InvoiceDate"] > row["FirstPurchase"]) & (df["InvoiceDate"] <= (row["FirstPurchase"] + pd.Timedelta(days=90)))].shape[0])>0), axis=1)
    retention_m3_rate = cohorts["RetainedM3"].mean() if len(cohorts)>0 else 0
    return {
        "total_rev": total_rev,
        "unique_customers": unique_customers,
        "aov": aov,
        "retention_m3": retention_m3_rate
    }

@st.cache_data
def compute_rfm(df, analysis_end, r_bins=5, f_bins=5, m_bins=5):
    d = df[df["Customer ID"] != "UNKNOWN"].copy()
    rfm = d.groupby("Customer ID").agg(recency_days=("InvoiceDate", lambda x: (analysis_end - x.max()).days),
                                         frequency=("Invoice", "nunique"),
                                         monetary=("Revenue", "sum"))
    rfm["monetary"] = rfm["monetary"].clip(lower=0.0)
    try:
        rfm["R"] = pd.qcut(rfm["recency_days"], q=r_bins, labels=[5,4,3,2,1]).astype(int)
    except Exception:
        rfm["R"] = pd.cut(rfm["recency_days"], bins=r_bins, labels=[5,4,3,2,1]).astype(int)
    try:
        rfm["F"] = pd.qcut(rfm["frequency"].rank(method='first'), q=f_bins, labels=[1,2,3,4,5]).astype(int)
    except Exception:
        rfm["F"] = pd.cut(rfm["frequency"].rank(method='first'), bins=f_bins, labels=[1,2,3,4,5]).astype(int)
    try:
        rfm["M"] = pd.qcut(rfm["monetary"], q=m_bins, labels=[1,2,3,4,5]).astype(int)
    except Exception:
        rfm["M"] = pd.cut(rfm["monetary"], bins=m_bins, labels=[1,2,3,4,5]).astype(int)
    rfm["RFMCode"] = rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)

    def label_from_scores(row):
        r,f,m = row["R"], row["F"], row["M"]
        if r==5 and f>=4 and m>=4:
            return "Champions"
        if f>=4 and m>=3:
            return "Loyal"
        if r>=4 and f<=2:
            return "New / Promising"
        if r<=2 and f>=3:
            return "At Risk"
        if r<=2 and f<=2:
            return "Can't Lose Them"
        return "Others"

    rfm["SegmentLabel"] = rfm.apply(label_from_scores, axis=1)
    seg = rfm.groupby(["RFMCode","SegmentLabel"]).agg(n_customers=("recency_days","count"),
                                                         CA=("monetary","sum"),
                                                         avg_basket=("monetary",lambda x: x.sum()/max(x.count(),1))).reset_index()
    return rfm.reset_index(), seg

@st.cache_data
def cohort_retention(df, cohort_period='M'):
    d = df.copy()
    d = d[d["Customer ID"] != "UNKNOWN"]
    d["CohortMonth"] = d.groupby("Customer ID")["InvoiceDate"].transform("min").dt.to_period('M').dt.to_timestamp()
    d["InvoiceMonth"] = d["InvoiceDate"].dt.to_period('M').dt.to_timestamp()
    cohort_counts = d.groupby(["CohortMonth","InvoiceMonth"]).agg(n_customers=("Customer ID","nunique")).reset_index()
    cohort_counts["PeriodIndex"] = ((cohort_counts["InvoiceMonth"] - cohort_counts["CohortMonth"]).dt.days / 30.0).round().astype(int)
    cohort_pivot = cohort_counts.pivot_table(index="CohortMonth", columns="PeriodIndex", values="n_customers").fillna(0)
    cohort_sizes = cohort_pivot.iloc[:,0] if cohort_pivot.shape[1]>0 else pd.Series(dtype=float)
    retention = cohort_pivot.divide(cohort_sizes, axis=0) if cohort_pivot.shape[1]>0 else pd.DataFrame()
    return retention

@st.cache_data
def empirical_clv(df, discount_rate_annual=0.1):
    d = df.copy()
    d = d[d["Customer ID"] != "UNKNOWN"]
    # d["MonthIndex"] = (d["InvoiceDate"].dt.to_period('M').astype(int) - d["InvoiceDate"].dt.to_period('M').min().astype(int))
    d["MonthIndex"] = (
        (d["InvoiceDate"].dt.to_period('M') - d["InvoiceDate"].dt.to_period('M').min())
        .apply(lambda x: x.n)  # convert PeriodIndex difference to integer
    )
    cust_month = d.groupby(["Customer ID","MonthIndex"]).agg(month_rev=("Revenue","sum")).reset_index()
    monthly_discount = (1 + discount_rate_annual) ** (cust_month["MonthIndex"] / 12.0)
    cust_month["NPV"] = cust_month["month_rev"] / monthly_discount
    clv_per_customer = cust_month.groupby("Customer ID")["NPV"].sum()
    return clv_per_customer

def simulate_scenario(rfm_df, base_clv_per_cust, margin_pct, retention_lift_pct, discount_pct, apply_by_segment=None):
    base_margin = 0.30
    effective_margin = max(0.0, margin_pct/100.0 - discount_pct/100.0)
    factor = (effective_margin / base_margin) * (1 + retention_lift_pct)
    new_clv = base_clv_per_cust * factor
    delta_clv = new_clv.sum() - base_clv_per_cust.sum()
    delta_ca = (new_clv.sum() / effective_margin) - (base_clv_per_cust.sum() / base_margin) if effective_margin>0 else np.nan
    return {
        "new_clv_total": new_clv.sum(),
        "delta_clv": delta_clv,
        "delta_ca": delta_ca,
        "new_clv_per_customer": new_clv
    }



st.set_page_config(page_title="RFM / CLV Diagnostic & Simulator", layout="wide")


st.title("RFM & CLV Diagnostic — Streamlit App")
st.markdown("Une application pour diagnostiquer, prioriser et simuler l'impact CRM (rétention, CLV, CA).")

with st.sidebar:
    st.header("Chargement & Périmètres")
    uploaded_file = st.file_uploader("Charger le fichier CSV (colonnes attendues: Invoice,Quantity,InvoiceDate,Price,Customer ID,Country)", type=["csv","txt"])
    st.caption("Fichier exemple: dataset transactions. Dates au format ISO ou dd/mm/yyyy.")
    df_raw = load_data(uploaded_file)

    df_raw["InvoiceDate"] = pd.to_datetime(df_raw["InvoiceDate"])

    if df_raw is not None:
        min_date = df_raw["InvoiceDate"].min().date()
        max_date = df_raw["InvoiceDate"].max().date()
    else:
        min_date = datetime.today().date() - timedelta(days=365)
        max_date = datetime.today().date()

    st.subheader("Fenêtre d'analyse")
    start_date = st.date_input("Date de début", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input("Date de fin", min_value=min_date, max_value=max_date, value=max_date)
    time_unit = st.selectbox("Unité de temps", ["Mois","Trimestre","Annee"], index=0)
    country_filter = st.multiselect("Pays", options=sorted(df_raw["Country"].unique().tolist()) if df_raw is not None else [], default=None)
    mode_returns = st.selectbox("Mode retours", ["include","exclude","neutralize"], index=0)
    min_order_threshold = st.number_input("Seuil de commande (quantité min)", min_value=0, value=1)

    st.markdown("---")
    st.header("Scénario: paramètres globaux")
    margin_pct = st.slider("Marge attendue (%)", 0.0, 100.0, 30.0, step=1.0)
    retention_lift = st.slider("Delta rétention (fraction, ex: 0.05 = +5%)", -0.5, 2.0, 0.0, step=0.01)
    discount_pct = st.slider("Remise moyenne appliquée (%)", 0.0, 100.0, 0.0, step=1.0)
    discount_apply_scope = st.selectbox("Appliquer remise à", ["Global","Par segment RFM"], index=0)
    discount_include_returns = st.checkbox("Inclure retours dans simulation", value=(mode_returns=="include"))
    discount_rate = st.slider("Taux d'actualisation annuel (d)", 0.0, 0.5, 0.10, step=0.01)

if 'df_raw' not in globals() or df_raw is None:
    st.info("Chargez le fichier CSV pour activer l'application. J'ai créé un modèle d'app qui portera sur vos données.")
    st.stop()

analysis_end = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
df = preprocess(df_raw, pd.to_datetime(start_date), analysis_end, mode_returns)

if country_filter:
    df = df[df["Country"].isin(country_filter)]

if min_order_threshold > 0:
    df = df[df["Quantity"].abs() >= min_order_threshold]

with st.expander("Filtres actifs", expanded=True):
    cols = st.columns(3)
    cols[0].write(f"Période: {start_date} → {end_date}")
    cols[1].write(f"Unité: {time_unit}")
    cols[2].write(f"Retours: {mode_returns} {'(retours exclus)' if mode_returns=='exclude' else ''}")
    st.write(f"Pays sélectionnés: {country_filter or 'Tous'}")
    st.write(f"Seuil commande: {min_order_threshold}")

kpis = compute_kpis(df, analysis_end)
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric(label="CA total", value=f"{kpis['total_rev']:.0f}")
with k2:
    st.metric(label="Clients uniques", value=f"{kpis['unique_customers']}")
with k3:
    st.metric(label="AOV (ordre moyen)", value=f"{kpis['aov']:.2f}")
with k4:
    st.metric(label="Rétention M+3", value=f"{kpis['retention_m3']*100:.1f}%")

st.markdown("---")
st.header("Cohortes & Rétention")
retention = cohort_retention(df)
if retention.empty:
    st.info("Pas assez d'observations pour calculer les cohortes avec les filtres appliqués.")
else:
    st.write("Rétention par cohorte (lignes = cohorte d'acquisition, colonnes = mois depuis acquisition)")
    st.dataframe(retention.round(3).style.format("{:.2%}"))
    fig, ax = plt.subplots(figsize=(10,4))
    for i, idx in enumerate(retention.index[:2]):
        ax.plot(retention.columns, retention.loc[idx].fillna(0), marker='o', label=idx.strftime('%Y-%m'))
    ax.set_xlabel('Mois depuis acquisition')
    ax.set_ylabel('Taux de rétention')
    ax.legend(title='Cohorte')
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    st.download_button("Télécharger graphique cohorte (PNG)", data=buf, file_name=f"cohort_retention_{start_date}_{end_date}.png", mime="image/png")

st.markdown("---")
st.header("Segments RFM — Prioriser")
rfm_df, seg_table = compute_rfm(df, analysis_end)
seg_table_display = seg_table.copy()
seg_table_display["marge_estimee"] = seg_table_display["CA"] * (margin_pct/100.0)
seg_table_display["panier_moyen"] = seg_table_display["avg_basket"]
priority_map = {"Champions":1, "Loyal":2, "New / Promising":3, "Can't Lose Them":2, "At Risk":1, "Others":4}
seg_table_display["priority_activation"] = seg_table_display["SegmentLabel"].map(priority_map).fillna(4).astype(int)
seg_table_display = seg_table_display.sort_values(["priority_activation","n_customers"])
st.dataframe(seg_table_display.rename(columns={"RFMCode":"Code","SegmentLabel":"Label","n_customers":"N","CA":"CA","marge_estimee":"Marge_estimee","panier_moyen":"Panier_moyen","priority_activation":"Priorite"}).style.format({"CA":"{:.0f}","Marge_estimee":"{:.0f}","Panier_moyen":"{:.2f}"}))
csv_seg = seg_table_display.to_csv(index=False).encode('utf-8')
st.download_button("Exporter table Segments (CSV)", data=csv_seg, file_name=f"segments_rfm_{start_date}_{end_date}.csv", mime='text/csv')

selected_seg = st.selectbox("Voir clients d'un segment (RFM Code)", options=[None]+seg_table_display["RFMCode"].tolist())
if selected_seg:
    subset = rfm_df[rfm_df["RFMCode"]==selected_seg]
    st.write(f"Clients dans {selected_seg}: {subset.shape[0]}")
    st.dataframe(subset.head(200))
    csv_cust = subset.to_csv(index=False).encode('utf-8')
    st.download_button("Exporter liste clients (CSV)", data=csv_cust, file_name=f"clients_{selected_seg}_{start_date}_{end_date}.csv", mime='text/csv')

st.markdown("---")
st.header("CLV — Empirique et Formel (approx.)")
with st.expander("Définitions & exemples"):
    st.markdown("- CLV empirique: somme actualisée des revenus historiques par client.\n- CLV (formule fermée) exemple: CLV = margin * AOV * purchase_frequency / (1 - retention + discount).\n  Exemple numérique: marge 30%, AOV 50€, fréquence 2/an, retention 0.6, discount 0.10 -> CLV ≈ 0.3*50*2/(1-0.6+0.1)=30/0.5=60€.")

base_clv_series = empirical_clv(df, discount_rate)
st.write(f"CLV empirique moyen (par client): {base_clv_series.mean():.2f} — n clients calculés: {base_clv_series.shape[0]}")

clv_df = base_clv_series.reset_index()
clv_df.columns = ["Customer ID","CLV"]
clv_df = clv_df.merge(rfm_df[["Customer ID","RFMCode","SegmentLabel"]], on="Customer ID", how='left')
clv_by_seg = clv_df.groupby(["RFMCode","SegmentLabel"]).agg(n_customers=("Customer ID","count"), avg_clv=("CLV","mean"), total_clv=("CLV","sum")).reset_index()
st.dataframe(clv_by_seg.style.format({"avg_clv":"{:.2f}","total_clv":"{:.0f}"}))

st.header("Scénarios — Simulateur d'impact")
if st.button("Lancer la simulation"):
    res = simulate_scenario(rfm_df, base_clv_series, margin_pct, retention_lift, discount_pct, apply_by_segment=None)
    st.metric("Δ CLV total (approx.)", f"{res['delta_clv']:.0f}")
    st.metric("Δ CA (approx.)", f"{res['delta_ca']:.0f}")
    st.write("Note: simulation approximative basée sur simplifications — utiliser résultats comme indicateur relatif.")

st.markdown("---")
st.header("Exports & Traçabilité")
csv_filtered = df.to_csv(index=False).encode('utf-8')
st.download_button("Exporter données filtrées (CSV)", data=csv_filtered, file_name=f"transactions_filtered_{start_date}_{end_date}.csv", mime='text/csv')

fig2, ax2 = plt.subplots(figsize=(12,6))
monthly = df.groupby(df["InvoiceDate"].dt.to_period('M').dt.to_timestamp())["Revenue"].sum()
ax2.plot(monthly.index, monthly.values, marker='o')
ax2.set_title('CA mensuel')
ax2.set_ylabel('CA')
st.pyplot(fig2)
buf2 = io.BytesIO()
fig2.savefig(buf2, format='png', bbox_inches='tight')
buf2.seek(0)
st.download_button("Télécharger CA mensuel (PNG)", data=buf2, file_name=f"ca_mensuel_{start_date}_{end_date}.png", mime='image/png')

# st.markdown("---")
# st.caption("Conseils d'accessibilité: tailles de police agrandies, labels clairs et contrastes. Toujours indiquer les effectifs (n) à côté des pourcentages.")

# st.markdown("---")
# st.write("**Pour exécuter localement:** `pip install streamlit pandas numpy matplotlib` puis `streamlit run streamlit_rfm_clv_app.py`.")
