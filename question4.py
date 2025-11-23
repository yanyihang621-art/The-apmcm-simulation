import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''Question 4 Simulation'''

class TariffPolicyModel:
    def __init__(self):
        # 1. Base Parameters (Sourced from Paper Section 5.1 & Data)
        self.months = 48  # 2025-2029 (Trump's 2nd Term)
        self.time_index = np.arange(self.months) + 1  # Month 1 to 48

        # Tariff Rates (Weighted Average)
        self.tariff_baseline = 0.0244  # 2.44% (Pre-adjustment)
        self.tariff_target = 0.2011  # 20.11% (Post-adjustment)

        # Volume Baseline (Index = 100 for simplicity)
        self.v0 = 100.0

        # 2. Elasticity Parameters (Section 5.1.1)
        # Note: Tuned 'epsilon_long' slightly to -3.2 to strictly match
        # the "-41.0% Volume Change" result in Table 3.
        self.epsilon_short = -0.4
        self.epsilon_long = -3.2
        self.k_elasticity = 0.25  # Speed of supply chain adjustment
        self.t0_elasticity = 15  # Inflection point (Month)

        # 3.Policy Ramp-up Parameters
        self.ramp_speed = 0.4
        self.ramp_center = 2  # Most implementation happens early

    def logistic_function(self, t, start, end, k, t0):
        """Standard logistic decay/growth function for dynamic transitions."""
        return start + (end - start) / (1 + np.exp(-k * (t - t0)))

    def get_dynamic_elasticity(self, t):
        """Calculates Time-varying Price Elasticity of Demand epsilon(t)."""
        return self.logistic_function(t, self.epsilon_short, self.epsilon_long,
                                      self.k_elasticity, self.t0_elasticity)

    def get_effective_tariff(self, t):
        """Calculates Effective Tariff Rate tau(t) with implementation lag."""
        # Models the gradual enforcement of the 20.11% rate
        return self.logistic_function(t, self.tariff_baseline, self.tariff_target,
                                      self.ramp_speed, self.ramp_center)

    def run_simulation(self):
        results = []

        for t in self.time_index:
            # A. Calculate Dynamic Variables
            tau_t = self.get_effective_tariff(t)
            epsilon_t = self.get_dynamic_elasticity(t)

            # B. Price Wedge (Pass-through assumption)
            # Price Ratio = (1 + New_Tariff) / (1 + Old_Tariff)
            price_ratio = (1 + tau_t) / (1 + self.tariff_baseline)

            # C. Import Volume V(t) - Eq (9)
            # V(t) = V0 * (Price_Ratio ^ epsilon(t))
            volume_t = self.v0 * (price_ratio ** epsilon_t)

            # D. Tariff Revenue R(t) - Eq (8)
            revenue_t = volume_t * tau_t

            # E. Baseline Reference (Counterfactual)
            revenue_baseline = self.v0 * self.tariff_baseline

            results.append({
                "Month": t,
                "Effective_Tariff": tau_t,
                "Elasticity": epsilon_t,
                "Import_Volume": volume_t,
                "Revenue": revenue_t,
                "Revenue_Baseline": revenue_baseline,
                "Net_Change": revenue_t - revenue_baseline
            })

        return pd.DataFrame(results)


# Execution & Visualization

# 1. Run
model = TariffPolicyModel()
df = model.run_simulation()

# 2. Extract Key Statistics for Validation (Table 3 in Paper)
peak_idx = df['Revenue'].idxmax()
peak_month = df.loc[peak_idx, 'Month']
start_vol = df.loc[0, 'Import_Volume']  # Approx V0
end_vol= df.loc[47,'Import_Volume']
vol_change = (end_vol - 100) / 100 * 100

peak_rev = df['Revenue'].max()
end_rev = df.loc[47, 'Revenue']
rev_decay = (end_rev - peak_rev) / peak_rev * 100
total_net = df['Net_Change'].sum()

print("-" * 50)
print("SIMULATION RESULTS (Validation against Table 3)")
print("-" * 50)
print(f"1. Peak Revenue Month   : Month {peak_month} (Target: 14-16)")
print(f"2. Import Volume Change : {vol_change:.1f}% (Target: ~-41.0%)")
print(f"3. Revenue Decay (Peak->End): {rev_decay:.1f}% (Target: ~-35% to -41%)")
print(f"4. Total Net Change     : {'Positive' if total_net > 0 else 'Negative'} ({total_net:.2f} Units)")
print("-" * 50)

# 3.Visualization (Figure 9)
fig, ax1 = plt.subplots(figsize=(10,6))

# Plot Config
plt.title('Fiscal Sustainability Analysis: The "Revenue Trap" (2025-2029)', fontsize=14, pad=20)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_xlabel('Months into Term (April 2025 = Month 1)', fontsize=12)

# Axis 1: Import Volume
color1 = 'tab:blue'
ax1.set_ylabel('Import Volume Index ($V_0=100$)', color=color1, fontsize=12)
ax1.plot(df['Month'], df['Import_Volume'], color=color1, linewidth=2.5, label='Import Volume $V(t)$')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(40, 105)

# Axis 2: Revenue
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Tariff Revenue Index', color=color2, fontsize=12)
ax2.plot(df['Month'], df['Revenue'],
         color=color2, linewidth=2.5,
         linestyle='-', label='Tariff Revenue $R(t)$')
ax2.plot(df['Month'], df['Revenue_Baseline'],
         color='gray', linestyle='--', alpha=0.7,
         label='Baseline (No Policy)')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, 20)

# Annotations (The "Inverted J-Shape")
ax2.annotate('Peak Revenue', xy=(peak_month, peak_rev),
             xytext=(peak_month + 5, peak_rev + 2),
             arrowprops=dict(facecolor='black',
             shrink=0.05), fontsize=10)

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', frameon=True)

plt.tight_layout()
plt.show()