# FairShare — Nash Bargaining for Fair CPU & RAM Allocation

A classroom-ready Python simulation for the Game Theory topic **Bargaining Game for Fair Distribution of Computational Resources**.

## Features
- Interactive Streamlit dashboard
- Editable process names, CPU demand, RAM demand, and priority weights
- Nash bargaining allocation using numerical optimization
- Comparison against equal-sharing and priority-first baselines
- CPU/RAM charts, utilization metrics, fairness index, and capacity checks
- No database, API key, or real server required

## Requirements
- Python 3.10 or newer
- Internet only for the first installation of Python packages

## Run on macOS / Linux
Open Terminal and move into this project folder:

```bash
cd FairShare_Nash_Bargaining
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.

## Run on Windows
```powershell
cd FairShare_Nash_Bargaining
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Suggested live demo
1. Keep the default three processes.
2. Keep CPU capacity at 100 and RAM capacity at 32 GB.
3. Click **Run bargaining and compare policies**.
4. Point out total allocation never exceeds capacity.
5. Change Process A's priority weight from 3 to 1, click the button again, and observe the results.
6. Compare the Nash policy with equal-sharing and priority-first baselines.
7. Try changing CPU capacity to 60 to demonstrate scarcity.

## Mathematical model
For process i, allocation is `x_i`, demand is `d_i`, and positive priority weight is `w_i`.

Utility:
`u_i(x_i) = ln(1 + w_i * x_i)`

Disagreement allocation is zero, so `u_i(0) = 0`.

The Nash bargaining solution maximizes the product of utility gains:
`maximize ∏ [u_i(x_i) - u_i(0)]`

Equivalently, the implementation minimizes:
`-Σ ln(u_i(x_i))`

subject to:
- `0 ≤ x_i ≤ d_i`
- `Σ x_i ≤ capacity`

CPU and RAM are allocated independently with the same method. This is a simulation, not an operating-system scheduler.

## Important interpretation
Nash bargaining is not synonymous with equal sharing. Results depend on the utility function, disagreement point, priority weights, demand caps, and total capacity. The priority-first method is only a comparison baseline.
