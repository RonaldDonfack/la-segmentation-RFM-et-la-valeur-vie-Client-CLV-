# streamlit_rfm_clv_app.py
# Single-file Streamlit app implementing: filters, RFM segments table, cohort retention,
# CLV per segment and global, scenario simulator (margin/retention/discount), exports (CSV/PNG),
# KPIs with tooltips and accessibility concerns.



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
    d["MonthIndex"] = (d["InvoiceDate"].dt.to_period('M').astype(int) - d["InvoiceDate"].dt.to_period('M').min().astype(int))
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


