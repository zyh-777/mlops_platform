"""Multi-project monitoring dashboard.

Streamlit app showing:
    - Project selector (dropdown of all active projects)
    - Run history table from platform.run_log
    - Monitoring checks from platform.monitoring_log
    - Prediction distribution plots
    - Drift trend charts (PSI over time)

Usage:
    streamlit run src/dashboard/streamlit_app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy import text

from src.database.db_connection import get_engine
from src.logger.logger import get_logger

logger = get_logger("dashboard")


# ═══════════════════════════════════════════════════════════
# DATA LOADING FUNCTIONS
# ═══════════════════════════════════════════════════════════


@st.cache_data(ttl=60)
def load_projects() -> pd.DataFrame:
    """Loads all registered projects from platform.project_registry."""
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT * FROM project_registry ORDER BY project_id"),
            conn,
        )
    return df


@st.cache_data(ttl=60)
def load_run_log(project_id: str, limit: int = 30) -> pd.DataFrame:
    """Loads recent run history for a project."""
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT run_date, status, row_count, duration_sec, error_message, created_at
                FROM run_log
                WHERE project_id = :project_id
                ORDER BY run_date DESC
                LIMIT :limit
            """),
            conn,
            params={"project_id": project_id, "limit": limit},
        )
    return df


@st.cache_data(ttl=60)
def load_monitoring_log(project_id: str, limit: int = 100) -> pd.DataFrame:
    """Loads recent monitoring results for a project."""
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT run_date, check_name, passed, value, threshold, created_at
                FROM monitoring_log
                WHERE project_id = :project_id
                ORDER BY run_date DESC, check_name
                LIMIT :limit
            """),
            conn,
            params={"project_id": project_id, "limit": limit},
        )
    return df


@st.cache_data(ttl=60)
def load_predictions(table_name: str, limit: int = 100) -> pd.DataFrame:
    """Loads recent predictions from a project's result table."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text(f"SELECT * FROM {table_name} ORDER BY run_date DESC LIMIT :limit"),
                conn,
                params={"limit": limit},
            )
        return df
    except Exception:
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════
# DASHBOARD LAYOUT
# ═══════════════════════════════════════════════════════════


def main() -> None:
    """Main dashboard entry point."""
    st.set_page_config(
        page_title="MLOps Platform",
        page_icon="||",
        layout="wide",
    )

    st.title("MLOps Inference Platform")
    st.caption("Multi-project monitoring dashboard")

    # --- Sidebar: Project selector ---
    try:
        projects_df = load_projects()
        project_ids = projects_df["project_id"].tolist()
    except Exception:
        st.warning("Could not connect to database. Is MySQL running?")
        project_ids = []

    if not project_ids:
        st.info("No projects registered yet. Add a YAML to project_registry/projects/.")
        return

    selected = st.sidebar.selectbox("Select Project", project_ids)

    if not selected:
        return

    # --- Project info ---
    project_row = projects_df[projects_df["project_id"] == selected].iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Project", project_row.get("display_name", selected))
    col2.metric("Owner", project_row.get("owner", "—"))
    col3.metric("Status", project_row.get("status", "—"))

    st.divider()

    # --- Run History ---
    st.subheader("Run History")
    run_log = load_run_log(selected)
    if run_log.empty:
        st.info("No runs recorded yet.")
    else:
        # Summary metrics
        latest = run_log.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last Run", str(latest["run_date"]))
        c2.metric("Status", latest["status"])
        c3.metric("Rows", int(latest["row_count"]))
        c4.metric("Duration", f"{latest['duration_sec']:.1f}s")

        # Color-code status
        st.dataframe(
            run_log,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # --- Monitoring Checks ---
    st.subheader("Monitoring Checks")
    mon_log = load_monitoring_log(selected)
    if mon_log.empty:
        st.info("No monitoring data yet.")
    else:
        # Show latest check results
        latest_date = mon_log["run_date"].max()
        latest_checks = mon_log[mon_log["run_date"] == latest_date]
        st.caption(f"Latest checks ({latest_date})")
        st.dataframe(latest_checks, use_container_width=True, hide_index=True)

        # Monitoring trend chart
        if "value" in mon_log.columns:
            check_names = mon_log["check_name"].unique()
            selected_check = st.selectbox("Check trend", check_names)
            trend = mon_log[mon_log["check_name"] == selected_check].sort_values("run_date")
            if not trend.empty and trend["value"].notna().any():
                st.line_chart(trend.set_index("run_date")["value"])

    st.divider()

    # --- Recent Predictions ---
    st.subheader("Recent Predictions")
    try:
        from src.core.project_config import PROJECT_ROOT
        yaml_path = PROJECT_ROOT / "project_registry" / "projects" / f"{selected}.yaml"
        from src.core.project_config import ProjectConfig
        config = ProjectConfig.from_yaml(yaml_path)
        preds = load_predictions(config.output.target_table)
        if preds.empty:
            st.info("No predictions found.")
        else:
            st.dataframe(preds.head(50), use_container_width=True, hide_index=True)

            # Distribution plot for output columns
            for col in config.schema.output_columns:
                if col in preds.columns and pd.api.types.is_numeric_dtype(preds[col]):
                    st.subheader(f"Distribution: {col}")
                    st.bar_chart(preds[col].dropna().value_counts(bins=30).sort_index())
    except Exception as e:
        st.warning(f"Could not load predictions: {e}")


if __name__ == "__main__":
    main()
