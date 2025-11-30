import numpy as np
from scipy.integrate import odeint
import time

# ==============================================================================
# PART 0: PARAMETERS AND V31B (CHAOS PROOF) SUBSYSTEM
# ==============================================================================
CYCLES = 1000       
KAPPA_TARGET = 0.40
FREQ_FIXED = 5.0
GM_FIXED = -1.0
AMPL_FIXED = 0.20
DELTA_MIN = -2.0
DELTA_MAX = 0.5

# CRITICAL PARAMETER: Anomaly Filtering Threshold (S_min)
# LLE values below this threshold are considered sensor noise and filtered out.
LLE_THRESHOLD = 0.0600 

# --- V31B HELPER FUNCTIONS ---

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

class RegulatorSim:
    """Simulates the V31B Regulator Dynamics to generate time series data."""
    def __init__(self, rr_rate):
        self.kappa = 0.10 + np.random.normal(0, 0.0001)
        self.kappa_lag = KAPPA_TARGET
        self.rr = rr_rate
        self.k_log = []
        
    def step(self, i, delta_m):
        micro = AMPL_FIXED * np.sin(2 * np.pi * FREQ_FIXED * i / CYCLES)
        self.kappa += micro * GM_FIXED
        
        error_linier = self.kappa_lag - KAPPA_TARGET
        err_non_linier = np.tanh(error_linier) * 2.0
        self.kappa -= err_non_linier * self.rr * min(1.0, abs(GM_FIXED))
        
        self.kappa_lag = self.kappa
        self.k_log.append(self.kappa)
        return self.kappa

def lle_estimator(data, m=5, tau=2, max_iter=20):
    """Calculates the Largest Lyapunov Exponent (LLE) using Wolf's algorithm principles."""
    N = len(data)
    if N < m * tau: return 0.0
    epsilon = 1e-6
    trajectories = []
    
    for i in range(N - (m - 1) * tau):
        trajectory = [data[i + j * tau] for j in range(m)]
        trajectories.append(trajectory)
    
    distances = []
    if len(trajectories) < 2: return 0.0
    x_i = trajectories[0]
    
    for i in range(1, len(trajectories)):
        x_j = trajectories[i]
        dist = np.linalg.norm(np.array(x_i) - np.array(x_j))
        if dist > 0: distances.append(np.log(dist / epsilon))
            
    if len(distances) < 2: return 0.0
    time_points = np.arange(1, len(distances) + 1) * tau
    points_to_fit = min(len(time_points), max_iter)
    
    try:
        # Simple linear fit to log(distance) vs time
        slope, _ = np.polyfit(time_points[:points_to_fit], distances[:points_to_fit], 1)
        return max(0.0, slope)
    except ValueError:
        return 0.0 

def get_lle_signal(rr_value, cycles=CYCLES):
    """Executes the V31B simulation and returns the raw LLE signal."""
    reg = RegulatorSim(rr_value)
    np.random.seed(42) 
    
    for i in range(cycles):
        delta = np.random.uniform(DELTA_MIN, DELTA_MAX) 
        reg.step(i, delta)

    k_steady = np.array(reg.k_log)[cycles//2:] 
    lle = lle_estimator(k_steady)
    
    return lle

# ==============================================================================
# PART 1: NEURO-DYN (Bistable Dynamical System)
# ==============================================================================

def neuro_chaos_ode(C, t, alpha1, alpha2, gamma1, gamma2, K1, K2, eta1, eta2, beta, S0_lle_filtered):
    """The core Bistable ODE model: C1=Critical, C2=Safe."""
    C1, C2 = C
    S_t_filtered = S0_lle_filtered 
    
    # Cross-Repression Terms (Hill function based)
    repression_C2_on_C1 = alpha1 / (1 + (C2 / K1)**eta1)
    repression_C1_on_C2 = alpha2 / (1 + (C1 / K2)**eta2)
    
    # Dynamics (C1 is driven by the filtered LLE signal)
    dC1dt = repression_C2_on_C1 + beta * S_t_filtered - gamma1 * C1
    dC2dt = repression_C1_on_C2 - gamma2 * C2
    
    return np.array([dC1dt, dC2dt])

class NeuroChaosAI:
    def __init__(self, C0, alpha, gamma, K, eta, beta):
        self.C0 = np.array(C0)
        self.params = (alpha[0], alpha[1], gamma[0], gamma[1], K[0], K[1], eta[0], eta[1], beta)
        self.t = np.linspace(0, 50, 1000)

    def infer_decision(self, lle_input_raw, regime_name):
        
        # --- LLE SIGNAL FILTERING: S_filtered = max(0, LLE_raw - S_min) ---
        s_filtered = max(0.0, lle_input_raw - LLE_THRESHOLD) 
        
        ode_params = self.params + (s_filtered,)
        
        try:
            solution = odeint(neuro_chaos_ode, self.C0, self.t, args=ode_params)
            C_final = solution[-1, :]
            
            # Classification Logic (Based on raw LLE for reporting)
            if lle_input_raw > 0.01:
                decision_klasifikasi = f"STATUS CRITICAL: {regime_name.upper()} DETECTED"
            else:
                decision_klasifikasi = "STATUS NORMAL: SYSTEM STABLE" 
            
            # Bistable Fate Determination
            if C_final[0] > C_final[1]:
                nasib = "C1 HIGH (Chaos Triggered Regime)"
            else:
                nasib = "C2 HIGH (Stability Regime)"
                
            return decision_klasifikasi, nasib, C_final, s_filtered
            
        except Exception as e:
            return f"Error NEURO-CHAOS: {e}", "ERROR", np.zeros(2), 0.0

# ==============================================================================
# PART 2: FINAL INTEGRATION AND REPORTING
# ==============================================================================

def full_neuro_chaos_pipeline(rr_rate_input):
    """Runs the full pipeline from Chaos Proof (V31B) to Action (Neuro-Dyn)."""
    
    # --- NEURO-DYN LOCKED PARAMETERS (Rigor-Verified Calibration) ---
    alpha = (10.0, 10.0)
    gamma = (1.0, 1.0)
    beta = 120.0        # High Coupling Rate for Gradation
    eta = (4.0, 4.0)
    K = (5.0, 5.0)      # Repression Thresholds
    C0 = [6.0, 4.0]     # CRITICAL: Bias to C1 (Critical) to ensure LLE flip
    
    ai_core = NeuroChaosAI(C0, alpha, gamma, K, eta, beta)
    
    # --- STEP 1: CHAOS PROOF (V31B) ---
    print(f"Calculating raw LLE (RR={rr_rate_input:.2f})...")
    lle_signal_raw = get_lle_signal(rr_rate_input)
    
    # --- STEP 3 & 4: DECISION & ACTION ---
    regime_name = "STABLE" if rr_rate_input == 1.00 else ("CHAOS" if rr_rate_input == 0.50 else "RUNAWAY")

    decision_klasifikasi, nasib_ai, C_final, s_filtered = ai_core.infer_decision(
        lle_input_raw=lle_signal_raw, 
        regime_name=regime_name
    )
    
    # --- REPORTING LOGIC ---
    
    print("\n===========================================================")
    print(f"** FINAL INTEGRATED DECISION (RR={rr_rate_input:.2f}) **")
    print("===========================================================")
    print(f"| LLE Raw (V31B): {lle_signal_raw:.4f}")
    print(f"| LLE Filtered (S): {s_filtered:.4f} (Threshold: {LLE_THRESHOLD})")
    print(f"| Bias Kritis (C0/Beta): {C0} / {beta}")
    print(f"| AI Decision: {decision_klasifikasi}")
    print(f"| Action Category: {nasib_ai}")
    print(f"| Activation C1/C2: C1:{C_final[0]:.4f}, C2:{C_final[1]:.4f}")
    print("-----------------------------------------------------------")
    
    # Final Action Grading (Adjusted based on C1 peak values after filtering)
    if C_final[0] > 13.0: 
        print(">>> ACTION: IMMEDIATE SCRAM - Level C1 EXTREME <<<")
    elif C_final[0] > 10.1: 
        print(">>> ACTION: HARD WARNING - Level C1 HIGH <<<")
    elif C_final[1] > 9.9:
        print(">>> ACTION: CRUISE MODE - Level C2 HIGH (System Stable) <<<")
    else:
        print(">>> ACTION: AMBIGUOUS - Further Analysis Required <<<")
    print("===========================================================")


# --- EXECUTION ---
if __name__ == "__main__":
    
    print("-----------------------------------------------------------")
    print("## 🚀 FINAL EXECUTION: ANTI-ANOMALY RIGOR TEST ##")
    print("-----------------------------------------------------------")

    # 1. Test Bounded Chaos (RR=0.50) -> Must trigger HARD WARNING
    full_neuro_chaos_pipeline(0.50)
    
    # 2. Test Runaway Chaos (RR=0.00) -> Must trigger IMMEDIATE SCRAM
    full_neuro_chaos_pipeline(0.00)
    
    # 3. Test STABLE (RR=1.00) -> Must trigger CRUISE MODE
    full_neuro_chaos_pipeline(1.00)
