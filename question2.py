import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


'''Question 2 Simulation'''

# Class: AutoTradeModel
# Description: Simulates the impact of tariffs on US-Japan Auto Trade
#              based on the Partial Equilibrium Model defined in Question 2

class AutoTradeModel:
    def __init__(self):
        # --- Model Parameters (Calibrated from Paper Data) ---
        self.sigma = 4.0  # Elasticity of substitution (Paper Section 3.1)
        self.baseline_tariff = 0.025  # 2.5% Baseline tariff
        self.shock_tariff = 0.20  # 20% Reciprocal tariff (Hypothetical scenario)
        self.shock_year = 2025  # Year policy is implemented

        # Initial Volumes (in Million Units, approximated from Figure 1 & 4)
        self.init_import_vol = 1.62  # Q_JP: Direct imports
        self.init_fdi_vol = 1.30  # Q_US_jp: Japanese cars made in USA (Base capacity)

        # Sensitivity of FDI to tariff gaps (Lambda in Eq 4)
        # Tuned to ensure medium-term substitution matches paper conclusions
        self.lambda_fdi = 5.5

        # Time lag for factory construction (Simulating "Short Term Pain")
        self.adjustment_speed = 1.5

    def calculate_demand_change(self, new_tariff):
        """
        Calculates change in direct import demand based on CES utility.
        Formula: Delta Q / Q = -sigma * (Delta P / P)
        """
        price_ratio = (1 + new_tariff) / (1 + self.baseline_tariff)
        # CES Demand Function implication: Q ~ P^(-sigma)
        volume_ratio = price_ratio ** (-self.sigma)
        return self.init_import_vol * volume_ratio

    def calculate_fdi_response(self, tariff_gap, years_passed):
        """
        Calculates the Non-Tariff Response (Production Relocation).
        Based on Paper Eq (4): Delta Q = lambda * ln(1 + tau - tau0) * K
        """
        if years_passed < 0:
            return 0

        # The Log-linear response to tariff pressure
        potential_shift = self.lambda_fdi * np.log(1 + tariff_gap)

        # Logistic function to simulate the time delay of building factories
        # We cannot shift production instantly (Paper Section 3.3.2)
        time_factor = 1 / (1 + np.exp(-self.adjustment_speed * (years_passed - 1.5)))

        return potential_shift * time_factor

    def run_simulation(self, start_year=2020, end_year=2030):
        """
        Runs the simulation loop over the specified years.
        """
        years = np.arange(start_year, end_year + 1, 0.5)  # Half-year steps
        results = []

        print(f"{'=' * 60}")
        print(f"Running Simulation: Tariff Shock in {self.shock_year}")
        print(f"Elasticity (sigma):{self.sigma}, Baseline Tariff: {self.baseline_tariff * 100}%")
        print(f"{'=' * 60}")

        for t in years:
            # Determine current tariff regime
            if t < self.shock_year:
                current_tariff = self.baseline_tariff
                tariff_gap = 0
                years_post_shock = -1
            else:
                current_tariff = self.shock_tariff
                tariff_gap = current_tariff - self.baseline_tariff
                years_post_shock = t - self.shock_year

            # 1. Calculate Direct Imports (Q_JP) - Decreases with tariff
            q_imports = self.calculate_demand_change(current_tariff)

            # 2. Calculate FDI Production (Q_US_jp) - Increases with tariff gap
            fdi_shift = self.calculate_fdi_response(tariff_gap, years_post_shock)
            q_fdi = self.init_fdi_vol + fdi_shift

            # 3. Total Japanese Brand Supply
            q_total = q_imports + q_fdi

            results.append({
                'Year': t,
                'Tariff_Rate': current_tariff,
                'Direct_Imports_JP': q_imports,
                'Local_Production_FDI': q_fdi,
                'Total_Supply': q_total
            })

        return pd.DataFrame(results)


# Visualization Function
def plot_results(df):
    """
    Generates a Matplotlib chart mimicking Figure 6 in the paper.
    """
    plt.figure(figsize=(12, 6))
    plt.style.use('ggplot')  # Aesthetic style

    # Plotting lines
    plt.plot(df['Year'], df['Direct_Imports_JP'],
             label='Direct Imports ($Q_{JP}$)', color='navy', marker='o', markersize=4)

    plt.plot(df['Year'], df['Local_Production_FDI'],
             label='Japanese Cars Made in USA ($Q_{US}^{jp}$)', color='firebrick', marker='s', markersize=4)

    plt.plot(df['Year'], df['Total_Supply'],
             label='Total Japanese Brand Supply', color='gray', linestyle='--', alpha=0.7)

    # Adding the Policy Shock Line
    plt.axvline(x=2025, color='black', linestyle=':', linewidth=1.5, label='Policy Implementation')

    # Annotations
    plt.title('Simulation of Question 2: The Substitution Effect of FDI\n(Impact of Tariff Increase from 2.5% to 20%)',
              fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Volume (Million Units)', fontsize=12)
    plt.legend(loc='best', frameon=True)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Highlight the "Structural Shift"
    plt.text(2028.5, df['Direct_Imports_JP'].iloc[-1] + 0.1,'Structural Decline',
             color='navy', fontsize=9, ha='center')
    plt.text(2028.5, df['Local_Production_FDI'].iloc[-1] + 0.1,'FDI Substitution',
             color='firebrick', fontsize=9, ha='center')

    # Show plot
    plt.tight_layout()
    plt.show()


# Main
if __name__ == "__main__":
    # Instantiate & Run
    model = AutoTradeModel()
    df_results = model.run_simulation()

    # Output numerical analysis for specific key years
    print("\nSimulation Results (Key Years):")
    print(f"{'-' * 65}")
    print(f"{'Year':<10} | {'Tariff':<8} | {'Imports(M)':<12} | {'FDI(M)':<10} | {'Total(M)':<10}")
    print(f"{'-' * 65}")

    key_years = [2024.0, 2025.0, 2026.0, 2028.0, 2030.0]
    for y in key_years:
        row = df_results[df_results['Year'] == y].iloc[0]
        print(f"{row['Year']:<10.1f} | {row['Tariff_Rate']:.1%}     | "
              f"{row['Direct_Imports_JP']:<12.3f} | {row['Local_Production_FDI']:<10.3f} | "
              f"{row['Total_Supply']:<10.3f}")
    print(f"{'-' * 65}")
    print("\nInterpretation:")
    print("1. Short Term (2025): Imports drop sharply due to price elasticity (-sigma).")
    print("2. Medium Term (2026-2028): FDI ramps up as firms execute Non-Tariff Strategies.")
    print("3. Long Term (>2029): Supply structure inverts. 'Made in USA' by Japanese firms dominates.")

    plot_results(df_results)