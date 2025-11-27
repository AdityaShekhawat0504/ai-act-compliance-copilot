# app.py
import streamlit as st
import pandas as pd

from monitoring_utils import log_audit_run, load_monitoring_log
from sklearn.model_selection import train_test_split


from data_utils import generate_synthetic_credit
from model_training import train_random_forest, evaluate_model, get_feature_importances
from risk_engine import compute_risk_score
from explanations import explain_model_shap_tree
from fairness_utils import statistical_parity_difference
from report_generator import generate_pdf_report

st.set_page_config(page_title="AI Act Compliance Copilot", layout="wide")

st.title("AI Act Compliance Copilot — Demo")

# ---------------- Sidebar navigation ----------------
st.sidebar.header("Demo Flow")
page = st.sidebar.radio(
    "Step",
    ["1. Risk Questionnaire", "2. Model Audit", "3. Report", "4. Monitoring Dashboard"]
)


# ====================================================
#                PAGE 1: RISK QUESTIONNAIRE
# ====================================================
# ====================================================
#                PAGE 1: RISK QUESTIONNAIRE
# ====================================================
if page == "1. Risk Questionnaire":
    st.header("📝 1. Advanced Risk Questionnaire")

    st.caption(
        "Describe your AI system across domain, data, model, deployment and governance. "
        "We compute a risk score inspired by the EU AI Act (Annex III + governance controls)."
    )

    with st.container():
        st.markdown("### 🧭 A. Domain / Use Case")
        domain_label = st.selectbox(
            "Which domain best describes this AI system?",
            [
                "Finance / credit / insurance",
                "Employment / HR",
                "Education / vocational training",
                "Healthcare / medical",
                "Biometrics",
                "Law enforcement / public safety",
                "Migration / border / asylum",
                "Justice / democratic processes",
                "Other / not listed",
            ],
        )

        domain_map = {
            "Finance / credit / insurance": "finance",
            "Employment / HR": "employment_hr",
            "Education / vocational training": "education",
            "Healthcare / medical": "healthcare_medical",
            "Biometrics": "biometrics",
            "Law enforcement / public safety": "law_enforcement_public_safety",
            "Migration / border / asylum": "migration_border_asylum",
            "Justice / democratic processes": "justice_democracy",
            "Other / not listed": "other",
        }
        domain = domain_map[domain_label]

    st.markdown("---")

    # --------------------- B. DATA PROFILE ---------------------
    st.markdown("### 💾 B. Data Profile")
    col1, col2 = st.columns(2)
    with col1:
        uses_personal_data = st.checkbox("Uses personal data (identifiable individuals)")
        uses_special_categories = st.checkbox("Uses special categories (health, ethnicity, religion, etc.)")

    with col2:
        large_scale_data = st.checkbox("Large-scale data processing (many users/records)")
        third_party_data = st.checkbox("Uses third-party / purchased data")

    st.markdown("---")

    # --------------------- C. MODEL CHARACTERISTICS ---------------------
    st.markdown("### 🤖 C. Model Characteristics")
    model_type_label = st.selectbox(
        "Model type (primary decision component)",
        [
            "Rules / expert system",
            "Simple linear model (e.g., logistic regression)",
            "Tree ensemble (e.g., random forest, gradient boosting)",
            "Neural network / deep learning",
            "External AI API / hosted model",
            "Other / not sure",
        ],
    )

    model_type_map = {
        "Rules / expert system": "rules",
        "Simple linear model (e.g., logistic regression)": "simple_linear",
        "Tree ensemble (e.g., random forest, gradient boosting)": "tree_ensemble",
        "Neural network / deep learning": "neural_network",
        "External AI API / hosted model": "external_api",
        "Other / not sure": "other",
    }
    model_type = model_type_map[model_type_label]

    col3, col4 = st.columns(2)
    with col3:
        uses_generative_ai = st.checkbox("Uses generative AI (LLMs, image generators, etc.)")
    with col4:
        online_learning = st.checkbox("Model updates itself continuously (online learning)")

    st.markdown("---")

    # --------------------- D. DEPLOYMENT ---------------------
    st.markdown("### 🚀 D. Deployment & Impact")
    cold1, cold2 = st.columns(2)
    with cold1:
        fully_automated = st.checkbox("Decisions are fully automated by default")
        affects_legal_rights = st.checkbox("Affects legal/economic rights (credit, jobs, housing, benefits)")
    with cold2:
        safety_critical = st.checkbox("Safety-critical (health, physical safety, critical infrastructure)")
        real_time = st.checkbox("Used in real-time / low-latency setting")

    eu_users = st.checkbox("Impacts users in the EU", value=True)

    st.markdown("---")

    # --------------------- E. GOVERNANCE ---------------------
    st.markdown("### 🛡️ E. Governance & Controls")
    colg1, colg2 = st.columns(2)
    with colg1:
        has_human_oversight = st.checkbox("Human oversight for important decisions (manual review / override)")
        has_logging = st.checkbox("Technical logging / audit trail implemented")
    with colg2:
        has_documentation = st.checkbox("Documentation of data, model, limitations exists")
        has_monitoring = st.checkbox("Monitoring in place (performance, drift, incidents)")

    # --------------------- COLLECT ANSWERS ---------------------
    answers = {
        "domain": domain,
        "uses_personal_data": uses_personal_data,
        "uses_special_categories": uses_special_categories,
        "large_scale_data": large_scale_data,
        "third_party_data": third_party_data,
        "model_type": model_type,
        "uses_generative_ai": uses_generative_ai,
        "online_learning": online_learning,
        "fully_automated": fully_automated,
        "affects_legal_rights": affects_legal_rights,
        "safety_critical": safety_critical,
        "real_time": real_time,
        "eu_users": eu_users,
        "has_human_oversight": has_human_oversight,
        "has_logging": has_logging,
        "has_documentation": has_documentation,
        "has_monitoring": has_monitoring,
    }

    st.markdown("---")

    # --------------------- COMPUTE RISK CLASSIFICATION ---------------------
    compute_btn = st.button("🔎 Compute Risk Classification")

    if compute_btn:
        res = compute_risk_score(answers)

        # --- Save in session_state for REPORT PAGE ---
        st.session_state.risk_level = res["level"]
        st.session_state.risk_score = res["score"]
        st.session_state.risk_breakdown = res["breakdown"]
        st.session_state.risk_reasons = res["reasons_by_pillar"]

        # Color-coded badge
        level = res["level"]
        if level == "Low Risk":
            color = "#00b894"  # green
        elif level == "Limited Risk":
            color = "#fdcb6e"  # yellow/orange
        elif level == "High Risk":
            color = "#e17055"  # orange/red
        else:  # Very High Risk
            color = "#d63031"  # red

        badge_html = f"""
        <div style='display:inline-block; padding:6px 14px; border-radius:999px;
                    background-color:{color}; color:white; font-weight:600;'>
            {level}
        </div>
        """

        # ---------- Summary card ----------
        st.markdown("### 📋 Risk Summary")
        colR1, colR2 = st.columns([2, 1])
        with colR1:
            st.markdown("**Overall Risk Level**")
            st.markdown(badge_html, unsafe_allow_html=True)
        with colR2:
            st.metric("Total Risk Score", res["score"])

        # ---------- Breakdown ----------
        st.markdown("### 📊 Breakdown by Pillar")
        breakdown_df = pd.DataFrame(
            {
                "pillar": ["Domain", "Data", "Model", "Deployment", "Governance (-)"],
                "score": [
                    res["breakdown"]["domain_score"],
                    res["breakdown"]["data_score"],
                    res["breakdown"]["model_score"],
                    res["breakdown"]["deployment_score"],
                    res["breakdown"]["governance_score"],
                ],
            }
        )
        st.table(breakdown_df)

        # ---------- Reasons ----------
        with st.expander("🔍 See detailed reasoning by pillar", expanded=True):
            for pillar, reasons in res["reasons_by_pillar"].items():
                if reasons:
                    st.markdown(f"**{pillar.capitalize()}**")
                    for r in reasons:
                        st.write("- ", r)

        st.caption(
            "This scoring follows EU AI Act-style thinking: Annex III sector risk, data sensitivity, "
            "model transparency, automation level, deployment impact, and mitigating governance controls."
        )

# ====================================================
#                PAGE 2: MODEL AUDIT
# ====================================================
elif page == "2. Model Audit":
    st.header("2. Model Audit")

    st.write(
        "Audit either the built-in synthetic credit model or upload your own tabular dataset "
        "to train & audit a model."
    )

    mode = st.radio(
        "Choose audit mode",
        ["Built-in synthetic credit demo", "Upload your own dataset"],
    )

    # --- initialise session_state keys if not present (for synthetic mode) ---
    if "audit_model" not in st.session_state:
        st.session_state.audit_model = None
        st.session_state.audit_numeric_cols = None
        st.session_state.audit_X_test = None
        st.session_state.audit_metrics = None
        st.session_state.audit_preds = None
        st.session_state.audit_spd = None

    # ----------------------------------------------------------------------
    # MODE 1: SYNTHETIC CREDIT DEMO (same as before, but inside this branch)
    # ----------------------------------------------------------------------
    if mode == "Built-in synthetic credit demo":
        st.subheader("Synthetic Credit Risk Demo")

        # Slider to choose synthetic dataset size
        n = st.slider("Number of samples for demo", 200, 5000, 1000)

        # Generate synthetic data on every run (cheap)
        X_train, X_test, y_train, y_test = generate_synthetic_credit(n=n)
        st.write("Synthetic dataset created with columns:", list(X_train.columns))

        # --- Train button: train model & store everything in session_state ---
        if st.button("Train & Evaluate Model (Synthetic)"):
            model, numeric_cols = train_random_forest(X_train, y_train)
            metrics, preds, probs = evaluate_model(model, numeric_cols, X_test, y_test)
            spd = statistical_parity_difference(preds, X_test["gender"].values)

            # log this audit run for monitoring
            risk_answers = {
                "domain": "finance",
                "uses_personal_data": True,
                "uses_special_categories": True,
                "large_scale_data": True,
                "third_party_data": False,
                "model_type": "tree_ensemble",
                "uses_generative_ai": False,
                "online_learning": False,
                "fully_automated": True,
                "affects_legal_rights": True,
                "safety_critical": False,
                "real_time": False,
                "eu_users": True,
                "has_human_oversight": True,
                "has_logging": True,
                "has_documentation": True,
                "has_monitoring": False,
            }
            risk_res = compute_risk_score(risk_answers)
            risk_level = risk_res["level"]

            log_audit_run(
                model_name="rf_credit_demo",
                domain="finance",
                accuracy=metrics["accuracy"],
                roc_auc=metrics["roc_auc"],
                spd=spd,
                risk_level=risk_level,
            )

            # store in session_state for display / SHAP
            st.session_state.audit_model = model
            st.session_state.audit_numeric_cols = numeric_cols
            st.session_state.audit_X_test = X_test
            st.session_state.audit_metrics = metrics
            st.session_state.audit_preds = preds
            st.session_state.audit_spd = spd

            st.success("Synthetic model trained, audited, and logged for monitoring.")

        # show results if we have a synthetic model
        if st.session_state.audit_model is not None:
            model = st.session_state.audit_model
            numeric_cols = st.session_state.audit_numeric_cols
            X_test = st.session_state.audit_X_test
            metrics = st.session_state.audit_metrics
            preds = st.session_state.audit_preds
            spd = st.session_state.audit_spd

            st.subheader("Performance Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", round(metrics["accuracy"], 3))
            with col2:
                st.metric("ROC AUC", round(metrics["roc_auc"], 3))

            st.write("Classification report (per class):")
            st.json(metrics["report"])

            st.subheader("Feature Importances")
            feat_imp = get_feature_importances(model, numeric_cols)
            st.table(pd.DataFrame(feat_imp, columns=["feature", "importance"]))

            st.subheader("Fairness Check (Statistical Parity Difference)")
            st.write("Statistical Parity Difference (male=1 - female=0):", round(spd, 3))
            st.caption("Values near 0 indicate similar approval rates; large positive/negative values may indicate bias.")

            # SHAP explanation (now safe to toggle without losing model)
            show_shap = st.checkbox("Show SHAP explanation (TreeExplainer)")
            if show_shap:
                st.write("Computing SHAP summary on a sample (this may take a few seconds)...")
                sample = X_test.sample(min(200, len(X_test)), random_state=42)
                shp_path, _ = explain_model_shap_tree(
                    model,
                    numeric_cols,
                    sample,
                    save_path="shap_summary.png"
                )
                st.image(shp_path, caption="SHAP summary — feature impact on default prediction")

    # ----------------------------------------------------------------------
    # MODE 2: UPLOAD YOUR OWN DATASET
    # ----------------------------------------------------------------------
    else:
        st.subheader("Upload Your Own Dataset")

        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())

                # Choose target column
                target_col = st.selectbox("Choose target (label) column", df.columns)
                # Choose protected attribute for fairness (optional but recommended)
                protected_col = st.selectbox(
                    "Choose protected attribute column for fairness (binary recommended)",
                    ["(None)"] + list(df.columns)
                )

                # Choose domain for logging
                domain_choice = st.selectbox(
                    "Domain for this model (for risk/monitoring)",
                    ["finance", "employment_hr", "healthcare_medical", "other"]
                )

                if st.button("Train & Audit Model (Uploaded Data)"):
                    # Basic preprocessing: drop rows with NA in target or protected
                    df_clean = df.dropna(subset=[target_col])
                    if protected_col != "(None)":
                        df_clean = df_clean.dropna(subset=[protected_col])

                    X = df_clean.drop(columns=[target_col])
                    y = df_clean[target_col]

                    # Keep only numeric features for this demo
                    X_numeric = X.select_dtypes(include=["number"])
                    if X_numeric.shape[1] == 0:
                        st.error("No numeric features found after preprocessing. Please upload numeric/tabular data.")
                    else:
                        X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
                            X_numeric, y, test_size=0.2, random_state=42
                        )

                        model_u, numeric_cols_u = train_random_forest(X_train_u, y_train_u)
                        metrics_u, preds_u, probs_u = evaluate_model(model_u, numeric_cols_u, X_test_u, y_test_u)

                        # Fairness if protected attribute chosen and numeric/binary
                        spd_u = None
                        if protected_col != "(None)":
                            if protected_col in df_clean.columns:
                                prot_test = df_clean.loc[X_test_u.index, protected_col]
                                try:
                                    spd_u = statistical_parity_difference(preds_u, prot_test.values)
                                except Exception as e:
                                    st.warning(f"Could not compute SPD for protected attribute: {e}")
                            else:
                                st.warning("Protected column not found in cleaned data; skipping fairness metric.")

                        st.subheader("Performance Metrics (Uploaded Data)")
                        colu1, colu2 = st.columns(2)
                        with colu1:
                            st.metric("Accuracy", round(metrics_u["accuracy"], 3))
                        with colu2:
                            st.metric("ROC AUC", round(metrics_u["roc_auc"], 3))

                        st.write("Classification report (per class):")
                        st.json(metrics_u["report"])

                        if spd_u is not None:
                            st.subheader("Fairness (Statistical Parity Difference)")
                            st.write("SPD (protected=1 - protected=0):", round(spd_u, 3))
                            st.caption("Values near 0 indicate similar positive prediction rates across groups.")

                        # Log this run into monitoring
                        log_audit_run(
                            model_name="rf_uploaded_demo",
                            domain=domain_choice,
                            accuracy=metrics_u["accuracy"],
                            roc_auc=metrics_u["roc_auc"],
                            spd=spd_u if spd_u is not None else 0.0,
                            risk_level="N/A",  # you could hook in compute_risk_score with more metadata
                        )

                        st.success("Uploaded-data model trained, audited, and logged to monitoring.")
                        st.info("Go to '4. Monitoring Dashboard' to see how this run appears over time.")

            except Exception as e:
                st.error(f"Failed to read or process uploaded CSV: {e}")
        else:
            st.info("Upload a CSV file to begin auditing your own dataset.")
# ====================================================
#                PAGE 3: REPORT GENERATOR
# ====================================================
elif page == "3. Report":
    st.header("3. Generate Compliance Report")

    st.write(
        "This generates a PDF report summarizing the risk classification and model audit. "
        "Make sure you have performed a Risk Classification or a Model Audit first."
    )

    # Collect what we can from session state
    risk_data_available = "risk_level" in st.session_state
    model_audit_available = st.session_state.get("audit_model") is not None

    if not (risk_data_available or model_audit_available):
        st.info("No risk or audit results found. Please complete steps 1 or 2 first.")
    else:
        # Ask user what to include in the final PDF
        include_risk = st.checkbox("Include Risk Assessment", value=risk_data_available)
        include_audit = st.checkbox("Include Model Audit Results", value=model_audit_available)

        if st.button("Generate PDF Report"):
            pdf_path = generate_pdf_report(
                include_risk=include_risk,
                include_audit=include_audit
            )

            if pdf_path:
                with open(pdf_path, "rb") as pdf:
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf,
                        file_name="ai_compliance_report.pdf",
                        mime="application/pdf"
                    )
                st.success("PDF report generated successfully.")
            else:
                st.error("Failed to generate PDF report.")

# ====================================================
#           PAGE 4: MONITORING DASHBOARD
# ====================================================
elif page == "4. Monitoring Dashboard":
    st.header("4. Monitoring Dashboard")

    df = load_monitoring_log()

    if df is None or df.empty:
        st.info("No audit runs logged yet. Go to '2. Model Audit' and run a few audits first.")
    else:
        st.subheader("Logged Audit Runs")
        st.dataframe(df)

        # Filter by model or domain (basic for now)
        models = ["(All)"] + sorted(df["model_name"].unique().tolist())
        domains = ["(All)"] + sorted(df["domain"].unique().tolist())

        colf1, colf2 = st.columns(2)
        with colf1:
            selected_model = st.selectbox("Filter by model name", models)
        with colf2:
            selected_domain = st.selectbox("Filter by domain", domains)

        filtered = df.copy()
        if selected_model != "(All)":
            filtered = filtered[filtered["model_name"] == selected_model]
        if selected_domain != "(All)":
            filtered = filtered[filtered["domain"] == selected_domain]

        if filtered.empty:
            st.warning("No records match the current filters.")
        else:
            st.subheader("Performance Over Time")
            st.line_chart(
                filtered.set_index("timestamp")[["accuracy", "roc_auc"]]
            )

            st.subheader("Fairness (SPD) Over Time")
            st.line_chart(
                filtered.set_index("timestamp")[["spd"]]
            )

            st.subheader("Risk Levels")
            st.write("Counts of runs by risk level:")
            st.bar_chart(
                filtered.groupby("risk_level")["timestamp"].count()
            )

            st.caption(
                "This dashboard shows real audit runs over time — "
                "in production, the same pattern would be fed by automated nightly or CI/CD checks."
            )
