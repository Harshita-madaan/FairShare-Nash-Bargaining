import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from fairshare import allocate_nash, allocate_equal, allocate_priority, fairness_index

st.set_page_config(page_title="FairShare", page_icon="⚖️", layout="wide")

# Compact styling: less vertical clutter, consistent cards and headings.
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px;}
[data-testid="stMetric"] {background: rgba(128,128,128,0.08); padding: 14px 16px; border-radius: 12px;}
div[data-testid="stDataEditor"] {border-radius: 10px;}
h1 {padding-bottom: 0;}
</style>
""", unsafe_allow_html=True)

st.title("⚖️ FairShare")
st.caption("Fair CPU & RAM allocation using the Nash Bargaining Solution")

with st.sidebar:
    st.header("Server capacity")
    cpu_capacity = st.number_input("CPU units available", min_value=0, max_value=10000, value=100, step=10)
    ram_capacity = st.number_input("RAM available (GB)", min_value=0, max_value=10000, value=32, step=4)
    st.divider()
    st.caption("Simulation only — this app does not control your computer's actual processes.")

left, right = st.columns([1.15, 1], gap="large")
with left:
    st.subheader("1. Process demands")
    default = pd.DataFrame({
        "Process": ["Process A", "Process B", "Process C"],
        "CPU demand": [60, 50, 40],
        "RAM demand (GB)": [16, 20, 12],
        "Priority": [3.0, 2.0, 1.0],
    })
    edited = st.data_editor(
        default,
        hide_index=True,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Process": st.column_config.TextColumn("Process", required=True),
            "CPU demand": st.column_config.NumberColumn("CPU", min_value=0, max_value=10000, step=5, required=True),
            "RAM demand (GB)": st.column_config.NumberColumn("RAM (GB)", min_value=0, max_value=10000, step=2, required=True),
            "Priority": st.column_config.NumberColumn("Weight", min_value=0.1, max_value=100.0, step=0.5, required=True,
                                                       help="Higher weight increases utility from allocated resources."),
        },
    )
    run = st.button("🤝 Calculate fair allocation", type="primary", use_container_width=True)

with right:
    st.subheader("2. What this simulation does")
    st.markdown("""
    - Processes compete for limited CPU and RAM.
    - Nash bargaining balances utility gains.
    - Results are compared with equal and priority-first sharing.
    """)
    with st.expander("Game theory model"):
        st.latex(r"u_i(x_i)=\\ln(1+w_i x_i)")
        st.latex(r"\\max \\prod_i u_i(x_i)")
        st.markdown("""
        **Constraints**
        - Allocation cannot be negative.
        - A process cannot receive more than its demand.
        - Total allocation cannot exceed capacity.

        CPU and RAM are allocated independently. This is a simulation, not an operating-system scheduler.
        """)

if "results" not in st.session_state:
    st.session_state["results"] = None
if run:
    st.session_state["results"] = edited.copy()

if st.session_state["results"] is None:
    st.info("Edit the demands if needed, then click **Calculate fair allocation**.")
    st.stop()

data = st.session_state["results"].copy()
data["Process"] = data["Process"].astype(str).str.strip()
if len(data) == 0 or (data["Process"] == "").any() or data["Process"].duplicated().any():
    st.error("Add at least one process. Process names must be non-empty and unique.")
    st.stop()
for col in ["CPU demand", "RAM demand (GB)", "Priority"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")
if data[["CPU demand", "RAM demand (GB)", "Priority"]].isna().any().any():
    st.error("Please enter valid numeric demands and weights.")
    st.stop()
if (data[["CPU demand", "RAM demand (GB)"]] < 0).any().any() or (data["Priority"] <= 0).any():
    st.error("Demands must be non-negative and weights must be positive.")
    st.stop()

names = data["Process"].tolist()
cpu_demands = data["CPU demand"].to_numpy(dtype=float)
ram_demands = data["RAM demand (GB)"].to_numpy(dtype=float)
weights = data["Priority"].to_numpy(dtype=float)

try:
    cpu_nash = allocate_nash(cpu_capacity, cpu_demands, weights)
    ram_nash = allocate_nash(ram_capacity, ram_demands, weights)
    cpu_equal = allocate_equal(cpu_capacity, cpu_demands)
    ram_equal = allocate_equal(ram_capacity, ram_demands)
    cpu_priority = allocate_priority(cpu_capacity, cpu_demands, weights)
    ram_priority = allocate_priority(ram_capacity, ram_demands, weights)
except Exception as exc:
    st.error(f"Could not calculate allocation: {exc}")
    st.stop()

results = pd.DataFrame({
    "Process": names,
    "CPU demand": cpu_demands,
    "CPU allocated": cpu_nash,
    "RAM demand (GB)": ram_demands,
    "RAM allocated (GB)": ram_nash,
})
results["CPU allocated"] = results["CPU allocated"].round(2)
results["RAM allocated (GB)"] = results["RAM allocated (GB)"].round(2)

st.divider()
st.subheader("3. Results")
m1, m2, m3 = st.columns(3)
m1.metric("CPU allocated", f"{cpu_nash.sum():.1f} / {cpu_capacity}")
m2.metric("RAM allocated", f"{ram_nash.sum():.1f} / {ram_capacity} GB")
m3.metric("Capacity checks", "PASS ✓" if (
    cpu_nash.sum() <= cpu_capacity + 1e-5 and ram_nash.sum() <= ram_capacity + 1e-5
    and np.all(cpu_nash <= cpu_demands + 1e-5) and np.all(ram_nash <= ram_demands + 1e-5)
    and np.all(cpu_nash >= -1e-5) and np.all(ram_nash >= -1e-5)
) else "CHECK FAILED")

st.dataframe(results, hide_index=True, use_container_width=True)

with st.expander("Compare allocation methods"):
    policy = st.selectbox("Resource to compare", ["CPU", "RAM"])
    if policy == "CPU":
        comp = pd.DataFrame({"Process": names, "Nash bargaining": cpu_nash,
                             "Equal sharing": cpu_equal, "Priority-first": cpu_priority})
        comp = comp.melt(id_vars="Process", var_name="Method", value_name="CPU units")
        fig = px.bar(comp, x="Process", y="CPU units", color="Method", barmode="group",
                     title="CPU allocation by method")
    else:
        comp = pd.DataFrame({"Process": names, "Nash bargaining": ram_nash,
                             "Equal sharing": ram_equal, "Priority-first": ram_priority})
        comp = comp.melt(id_vars="Process", var_name="Method", value_name="RAM (GB)")
        fig = px.bar(comp, x="Process", y="RAM (GB)", color="Method", barmode="group",
                     title="RAM allocation by method")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Advanced checks"):
    a, b = st.columns(2)
    a.metric("CPU Jain fairness index", f"{fairness_index(cpu_nash):.3f}")
    b.metric("RAM Jain fairness index", f"{fairness_index(ram_nash):.3f}")
    st.caption("Jain's index measures equality of allocation amounts; it is not a proof of Nash optimality.")
    st.write("CPU total demand:", float(cpu_demands.sum()), "| CPU capacity:", float(cpu_capacity))
    st.write("RAM total demand:", float(ram_demands.sum()), "| RAM capacity:", float(ram_capacity))
