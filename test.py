import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Constants
contract_time = 25  
total_periods = contract_time * 12  
initial_investment = 644_000_000  
monetary_correction_y = 0.1015  
monetary_correction_m = (1 + monetary_correction_y) ** (1 / 12)  
profit_y = 124_000_000
profit_increase_rate_y = 0.05  
profit_m = profit_y / 12  
profit_increase_rate_m = (1 + profit_increase_rate_y) ** (1 / 12)  

# Create DataFrame
df = pd.DataFrame({'period': range(1, total_periods + 1)})
df['year'] = (df['period'] - 1) // 12 + 1  
df['month'] = (df['period'] - 1) % 12 + 1  

# Calculate financial metrics
df['accumulated_correction'] = monetary_correction_m ** df['period']
df['corrected_investment'] = initial_investment * df['accumulated_correction']
df['period_profit'] = profit_m * (profit_increase_rate_m ** df['period'])
df['accumulated_profit'] = df['period_profit'].cumsum()

# Find payback point
payback_row = df[df['accumulated_profit'] >= df['corrected_investment']].head(1)
payback_reached = not payback_row.empty

# Plot results
plt.figure(figsize=(10, 5))
sns.lineplot(x=df['period'], y=df['corrected_investment'], label="Correção Monetária", color='orange')
sns.lineplot(x=df['period'], y=df['accumulated_profit'], label="Receita Bruta", color='blue')

# Mark payback point if reached
if payback_reached:
    payback_period = payback_row['period'].values[0]
    payback_value = payback_row['corrected_investment'].values[0]
    plt.scatter(payback_period, payback_value, color='red', zorder=3, label="Payback Point")

plt.xlabel("Periodo (em meses)")
plt.ylabel("Valor")
plt.title("Investimento x Retorno")
plt.legend()
plt.grid(True)
plt.show()

# If payback is not reached, find the max initial investment that allows payback in the last period
if not payback_reached:
    # Calculate the ratio of accumulated profit to accumulated correction for each period
    df['investment_ratio'] = df['accumulated_profit'] / df['accumulated_correction']

    # Determine the maximum possible initial investment allowing payback at any period
    max_initial_investment = df['investment_ratio'].max()

    print(f"Maximum possible initial investment for payback within the contract: {max_initial_investment:,.2f}")

    max_initial_investment = df['accumulated_profit'].iloc[-1] / df['accumulated_correction'].iloc[-1]
    print(f"Payback not reached. The maximum possible initial investment to break even in the last period is: {max_initial_investment:,.2f}")