import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''Question 1 Simulation'''

# 1. Model Parameters
# Price elasticity is typically negative and large for commodities
# Capacity elasticity is positive, reflecting supply reliability
BETA_PRICE = -2.5  # Price Sensitivity
BETA_CAP = 1.2  # Capacity/Reliability Weight

# Base Preference (Alpha): Brazil slightly favored, Argentina lower due to volatility
ALPHAS = {'USA': 0.0, 'Brazil': 0.5, 'Argentina': -0.5}

# Baseline Data Snapshot (Simulated Monthly Data)
# FOB Price ($/MT) and Capacity Index (normalized)
BASE_PRICES = {'USA': 450, 'Brazil': 440, 'Argentina': 430}
CAPACITIES = {'USA': 100, 'Brazil': 120, 'Argentina': 40}


def calculate_utility(country, tariff_rate):
    """
    Calculates the utility for a specific supplier country.
    Formula: U = Alpha + Beta1 * ln(Price * (1 + Tariff)) + Beta2 * ln(Capacity)
    """
    p_final = BASE_PRICES[country] * (1 + tariff_rate)
    cap = CAPACITIES[country]

    # The Multinomial Logit Core Equation
    utility = ALPHAS[country] + \
              BETA_PRICE * np.log(p_final) + \
              BETA_CAP * np.log(cap)
    return utility, p_final


def run_simulation(scenario_name, tariff_dict):
    """
    Runs a full market share simulation in the given tariff scenario.
    """
    print(f"\n--- Simulation Scenario: {scenario_name} ---")
    results = {'Country': [], 'Final_Price': [], 'Utility': [], 'Exp_U': [], 'Share': []}

    # 1.Calculate Utility and Exponentiated Utility (exp(U))
    sum_exp_utility = 0
    temp_data = []

    for country in ALPHAS.keys():
        tariff = tariff_dict.get(country, 0.0)
        u, p_final = calculate_utility(country, tariff)
        exp_u = np.exp(u)
        sum_exp_utility += exp_u

        temp_data.append((country, p_final, u, exp_u))

    # 2.Calculate Probability / Market Share
    # S_i = exp(U_i) / sum(exp(U_j))
    for country, p_final, u, exp_u in temp_data:
        share = exp_u / sum_exp_utility
        results['Country'].append(country)
        results['Final_Price'].append(p_final)
        results['Utility'].append(u)
        results['Exp_U'].append(exp_u)
        results['Share'].append(share)

        print(f"  [{country}] Tariff:{tariff_dict.get(country, 0):.0%} | "
              f"Landed Cost:${p_final:.1f} | Utility:{u:.2f} -> Share: {share:.1%}")

    return pd.DataFrame(results).set_index('Country')


# 2. Execution

# Scenario A
tariffs_baseline = {'USA': 0.03, 'Brazil': 0.03, 'Argentina': 0.03}
df_base = run_simulation("Baseline (Status Quo)", tariffs_baseline)

# Scenario B
tariffs_war = {'USA': 0.25, 'Brazil': 0.03, 'Argentina': 0.03}
df_war = run_simulation("Trade War (USA Tariff -> 25%)", tariffs_war)

# 3.Visualization
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(df_base))
width = 0.35

rects1 = ax.bar(x - width / 2, df_base['Share'], width, label='Baseline (3% Tariff)', color='skyblue')
rects2 = ax.bar(x + width / 2, df_war['Share'], width, label='Trade War (25% Tariff on US)', color='salmon')

ax.set_ylabel('Market Share (Probability)')
ax.set_title('Impact of Tariff Policy on Global Soybean Trade Structure\n(Multinomial Logit Model Simulation)')
ax.set_xticks(x)
ax.set_xticklabels(df_base.index)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)


# Display percentages
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1%}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
plt.tight_layout()
plt.show()

# Calculate US Share Drop
us_share_base = df_base.loc['USA', 'Share']
us_share_war = df_war.loc['USA', 'Share']
us_change = (us_share_war - us_share_base) / us_share_base

print(f"\n[Conclusion] USA Market Share Change: {us_change:.1%}")

print("Model simulation complete.")
