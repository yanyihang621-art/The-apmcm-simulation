import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


'''Question 3 Simulation'''

# Set style for professional visualization
plt.style.use('ggplot')


def run_semiconductor_simulation():
    print(">>> Initializing Model: Semiconductor Dilemma...")

    # 1. High-End Chips: The Security-Efficiency Frontier
    # Control Variable: Export Control Intensity (alpha), 0=Open, 1=Blockade
    alpha = np.linspace(0, 1, 100)

    df_high = pd.DataFrame({'Alpha': alpha})

    # Model:Short-term Security Index (S_sec)
    # Assumption:Stricter controls increase security
    # Matches Figure 7 X-axis range
    df_high['S_Short'] = 40 + 60 * (df_high['Alpha'] ** 0.5)

    # Model:Economic Efficiency (W_econ)
    # Assumption:Losing China market (30% rev) hurts efficiency non-linearly
    # Matches Figure 7 Y-axis range [100, 40]
    df_high['Efficiency'] = 100 - 60 * (df_high['Alpha'] ** 1.5)

    # Model:The Security Paradox (Long-term Impact)
    # Logic:If Revenue (Efficiency) drops below threshold, R&D stalls.
    # Result:Long-term security collapses despite strict controls.
    RD_THRESHOLD = 65.0
    df_high['Penalty'] = np.where(df_high['Efficiency'] < RD_THRESHOLD, 0.4, 0.0)
    df_high['S_Long'] = df_high['S_Short'] * (1 - df_high['Penalty'])

    # Find Optimal Strategy: Maximize Long-term Security before collapse
    opt_idx = df_high['S_Long'].idxmax()
    optimal_high = df_high.iloc[opt_idx]

    # 2. Mid-Range Chips: Resilience vs Cost
    # Context:Automotive/Industrial chips. Tariffs drive FDI but raise costs.
    tariffs = np.linspace(0, 0.5, 50)  # 0% to 50% Tariff
    df_mid = pd.DataFrame({'Tariff': tariffs})

    # Resilience Index: Increases as manufacturing reshores (Logistic adoption)
    df_mid['Resilience'] = 20 + 60 * (1 / (1 + np.exp(-15 * (tariffs - 0.2))))

    # Industrial Cost Index: Linear pass-through of tariffs to auto makers
    df_mid['Cost'] = 100 * (1 + 0.8 * tariffs)

    # Optimal Mid-Range: Max Resilience s.t. Cost < 120 (Tolerance)
    valid_mid = df_mid[df_mid['Cost'] <= 120]
    optimal_mid = valid_mid.iloc[-1] if not valid_mid.empty else df_mid.iloc[0]

    # 3. Low-End Chips: Deadweight Loss Analysis (Section 4.3)
    # Context: Price sensitive, low margin. Tariffs = Pure Tax.
    P_world = 10.0
    Tariff_Rate = 0.40  # 40% Tariff Scenario
    # Demand Function: Q = 200 -8P (Elastic)
    demand = lambda p: np.maximum(0, 200 - 8 * p)

    P_tariff = P_world * (1 + Tariff_Rate)
    Q_free = demand(P_world)
    Q_tariff = demand(P_tariff)

    # Fiscal Calculations
    gov_revenue = (P_tariff - P_world) * Q_tariff
    dwl = 0.5 * (P_tariff - P_world) * (Q_free - Q_tariff)  # Triangle Area

    # 4. Visualization (Matching Paper Figures)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plt.subplots_adjust(wspace=0.3)

    # Plot1: High-End Frontier (Figure 7 in Paper)
    ax1 = axes[0]
    # Short-term frontier (The Trade-off)
    ax1.plot(df_high['S_Short'], df_high['Efficiency'], 'b--', alpha=0.6, label='Short-term Frontier')
    # Long-term reality (Paradox)
    sc = ax1.scatter(df_high['S_Long'], df_high['Efficiency'], c=df_high['Alpha'],
                     cmap='viridis', label='Long-term Reality')

    # Annotate Optimal Zone
    ax1.scatter(optimal_high['S_Long'], optimal_high['Efficiency'], color='red', s=150, marker='*', zorder=5)
    ax1.annotate(f'Optimal Zone\n(Alpha={optimal_high["Alpha"]:.2f})',
                 xy=(optimal_high['S_Long'], optimal_high['Efficiency']),
                 xytext=(optimal_high['S_Long'] - 15, optimal_high['Efficiency'] - 20),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    ax1.set_title('High-End: Security-Efficiency Frontier')
    ax1.set_xlabel('National Security Index (S_sec)')
    ax1.set_ylabel('Economic Efficiency (W_econ)')
    ax1.grid(True)

    # Plot2: Mid-Range Trade-off
    ax2 = axes[1]
    ax2.plot(df_mid['Tariff'] * 100, df_mid['Resilience'], 'g-', linewidth=2, label='Supply Resilience')
    ax2.set_ylabel('Resilience Index (0-100)', color='g')

    ax2r = ax2.twinx()
    ax2r.plot(df_mid['Tariff'] * 100, df_mid['Cost'], 'r--', linewidth=2, label='Mfg Cost')
    ax2r.set_ylabel('Industrial Cost Index (Base=100)', color='r')

    ax2.set_title('Mid-Range: Resilience vs. Cost')
    ax2.set_xlabel('Tariff Rate (%)')

    # Plot3: Low-End Deadweight Loss (Figure 8 in Paper)
    ax3 = axes[2]
    prices = np.linspace(5, 20, 100)
    ax3.plot(demand(prices), prices, 'k-', label='US Demand')

    # Fill Areas
    ax3.fill_betweenx([P_world, P_tariff], 0, Q_tariff, color='green', alpha=0.3, label='Gov Revenue')
    ax3.fill_betweenx([P_world, P_tariff], Q_tariff, [demand(p) for p in [P_world, P_tariff]],
                      color='gray', alpha=0.5, hatch='//', label='Deadweight Loss')

    ax3.set_title(f'Low-End: Tariff Impact ({Tariff_Rate:.0%})')
    ax3.set_xlabel('Quantity')
    ax3.set_ylabel('Price ($)')
    ax3.legend()

    plt.tight_layout()
    plt.show()

    # 5.Output
    report_data = {
        'Tier': ['High-End', 'High-End', 'Mid-Range', 'Low-End', 'Low-End'],
        'Metric': ['Optimal Control (Alpha)', 'Max Long-term Security', 'Optimal Tariff Rate', 'Fiscal Revenue',
                   'Deadweight Loss'],
        'Value': [f"{optimal_high['Alpha']:.2f}", f"{optimal_high['S_Long']:.1f}",
                  f"{optimal_mid['Tariff']:.1%}", f"${gov_revenue:.1f}M", f"${dwl:.1f}M"],
        'Outcome': ['Balanced Strategy', 'Avoids Innovation Collapse', 'Cost-Resilience Balance', 'Short-term Gain',
                    'Pure Efficiency Loss']
    }

    print("\n" + "=" * 70)
    print("FINAL MODEL OUTPUT: SEMICONDUCTOR POLICY ASSESSMENT")
    print("=" * 70)
    print(pd.DataFrame(report_data).to_string(index=False))
    print("-" * 70)
    print("Key Insight: High tariffs on Low-End chips create significant DWL,")
    print("while excessive export controls on High-End chips trigger a")
    print("'Security Paradox' by undermining R&D funding.")
    print("=" * 70)


if __name__ == "__main__":
    run_semiconductor_simulation()