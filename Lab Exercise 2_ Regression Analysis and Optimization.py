# For checking sir, I install "pip install pandas numpy matplotlib seaborn scikit-learn scipy"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
from scipy.optimize import minimize
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for cleaner and more professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Your dataset
data = {
    'Labor_Hours': [100, 120, 90, 110, 130, 95, 115, 140, 85, 105, 125, 135, 92, 108, 118, 88, 112, 128, 102, 122],
    'Machine_Hours': [50, 60, 45, 55, 65, 48, 58, 70, 42, 52, 62, 68, 46, 54, 59, 44, 56, 64, 51, 61],
    'Material_Cost': [1000, 1200, 950, 1100, 1300, 980, 1150, 1400, 900, 1050, 1250, 1350, 960, 1080, 1180, 920, 1120, 1280, 1020, 1220],
    'Output': [2500, 2800, 2300, 2600, 2900, 2400, 2700, 3100, 2200, 2550, 2850, 3000, 2350, 2580, 2750, 2250, 2650, 2880, 2520, 2820]
}

df = pd.DataFrame(data)

# Print dataset summary to the terminal
print("Dataset Summary:")
print(df.describe())

# Multiple Regression Analysis
X = df[['Labor_Hours', 'Machine_Hours', 'Material_Cost']]
y = df['Output']
model = LinearRegression()
model.fit(X, y)

# Predictions and residuals
y_pred = model.predict(X)
residuals = y - y_pred

# Calculate key statistics
r2 = r2_score(y, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
rmse = np.sqrt(np.mean(residuals**2))

# Calculate coefficients and their significance
X_with_intercept = np.column_stack([np.ones(len(X)), X])
beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
var_beta = np.sum(residuals**2) / (len(y) - X.shape[1] - 1) * np.diag(np.linalg.inv(X_with_intercept.T @ X_with_intercept))
se_beta = np.sqrt(var_beta)
t_stats = beta / se_beta
t_p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(y) - X.shape[1] - 1))

# Calculate F-statistic and its p-value
f_stat = (r2 / X.shape[1]) / ((1 - r2) / (len(y) - X.shape[1] - 1))
f_p_value = 1 - stats.f.cdf(f_stat, X.shape[1], len(y) - X.shape[1] - 1)

# Optimization using scipy.optimize
def objective_function(x):
    """Negative of predicted output for maximization via minimization"""
    return -(beta[0] + beta[1]*x[0] + beta[2]*x[1] + beta[3]*x[2])

# Constraints for optimization
constraints = [
    {'type': 'ineq', 'fun': lambda x: 200 - x[0]},  # Labor <= 200
    {'type': 'ineq', 'fun': lambda x: 100 - x[1]},  # Machine <= 100
    {'type': 'ineq', 'fun': lambda x: 2000 - x[2]}, # Material <= 2000
    {'type': 'ineq', 'fun': lambda x: x[0]},        # Labor >= 0
    {'type': 'ineq', 'fun': lambda x: x[1]},        # Machine >= 0
    {'type': 'ineq', 'fun': lambda x: x[2]}         # Material >= 0
]

# Initial guess and optimization solve
x0 = [100, 50, 1000]
result = minimize(objective_function, x0, method='SLSQP', constraints=constraints)
optimal_output = -result.fun

# Print regression analysis results to the terminal
print("\n--- REGRESSION ANALYSIS RESULTS ---")
print(f"\nR-squared: {r2:.4f}")
print(f"Adjusted R-squared: {adj_r2:.4f}")
print(f"F-statistic: {f_stat:.2f} (p-value: {f_p_value:.4f})")
print(f"\nResidual Analysis:")
print(f"Root Mean Square Error (RMSE): {rmse:.2f}")

print("\nCoefficients and t-statistics:")
print(f"{'Variable':<15} {'Coefficient':<15} {'t-stat':<10} {'p-value':<10}")
print("-" * 55)
print(f"{'Intercept':<15} {beta[0]:<15.2f} {t_stats[0]:<10.3f} {t_p_values[0]:<10.4f}")
print(f"{'Labor Hours':<15} {beta[1]:<15.2f} {t_stats[1]:<10.3f} {t_p_values[1]:<10.4f}")
print(f"{'Machine Hours':<15} {beta[2]:<15.2f} {t_stats[2]:<10.3f} {t_p_values[2]:<10.4f}")
print(f"{'Material Cost':<15} {beta[3]:<15.2f} {t_stats[3]:<10.3f} {t_p_values[3]:<10.4f}")

# Print optimization results to the terminal
print("\n--- OPTIMIZATION RESULTS ---")
print(f"Optimal Labor Hours: {result.x[0]:.2f}")
print(f"Optimal Machine Hours: {result.x[1]:.2f}")
print(f"Optimal Material Cost: {result.x[2]:.2f}")
print(f"Maximum Predicted Output: {optimal_output:.2f}")


# ---Visualizations---

# Create a figure with a 2x2 grid for main analysis plots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))
plt.suptitle('Manufacturing Production Analysis', fontsize=20, fontweight='bold', y=0.98)
fig.subplots_adjust(hspace=0.4, wspace=0.3)

# 1. Actual vs Predicted Output
ax1 = axes[0, 0]
ax1.scatter(y, y_pred, alpha=0.8, color='#0077b6', s=60, edgecolors='black')
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax1.set_title(f'Actual vs Predicted Output\n$R^2$ = {r2:.4f}', fontsize=14)
ax1.set_xlabel('Actual Output')
ax1.set_ylabel('Predicted Output')
ax1.grid(True, linestyle='--', alpha=0.6)

# 2. Residuals Plot
ax2 = axes[0, 1]
ax2.scatter(y_pred, residuals, alpha=0.8, color='#00b4d8', s=60, edgecolors='black')
ax2.axhline(y=0, color='red', linestyle='--', lw=2)
ax2.set_title('Residuals Plot', fontsize=14)
ax2.set_xlabel('Predicted Output')
ax2.set_ylabel('Residuals')
ax2.grid(True, linestyle='--', alpha=0.6)

# 3. Correlation Matrix Heatmap
ax3 = axes[1, 0]
corr_matrix = df[['Labor_Hours', 'Machine_Hours', 'Material_Cost', 'Output']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, 
            linewidths=.5, cbar_kws={'shrink': 0.8}, ax=ax3)
ax3.set_title('Feature Correlation Matrix', fontsize=14)
ax3.tick_params(axis='x', rotation=45)
ax3.tick_params(axis='y', rotation=0)

# 4. Residuals Distribution Histogram
ax4 = axes[1, 1]
sns.histplot(residuals, bins=10, kde=True, color='#48cae4', edgecolor='black', ax=ax4)
ax4.set_title('Residuals Distribution', fontsize=14)
ax4.set_xlabel('Residuals')
ax4.set_ylabel('Frequency')
ax4.axvline(x=0, color='red', linestyle='--', lw=2, label='Mean Residual')
ax4.legend()

plt.show()

# --- Optimization Results Plot ---
fig_opt, ax_opt = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
plt.suptitle('Optimization Results & Resource Impact', fontsize=20, fontweight='bold', y=0.98)
fig_opt.subplots_adjust(wspace=0.4)

# 1. Output Comparison
ax_out = ax_opt[0]
output_comparison = [df['Output'].mean(), optimal_output]
colors = ['#52b788', '#1a759f']
bars_out = ax_out.bar(['Current Average', 'Predicted Optimal'], output_comparison, color=colors)
ax_out.set_title('Current vs. Predicted Optimal Output', fontsize=14)
ax_out.set_ylabel('Output Units')
ax_out.grid(True, linestyle='--', alpha=0.6, axis='y')
for bar in bars_out:
    height = bar.get_height()
    ax_out.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.0f}', 
                ha='center', va='bottom', fontsize=12)

# 2. Regression Coefficients
ax_coef = ax_opt[1]
coefficients = beta[1:]
std_errors = se_beta[1:]
variables = ['Labor Hours', 'Machine Hours', 'Material Cost']
colors_coef = ['#38a3a5', '#57cc99', '#95d5b2']
bars_coef = ax_coef.bar(variables, coefficients, color=colors_coef, yerr=std_errors, 
                        capsize=5, edgecolor='black')
ax_coef.set_title('Regression Coefficients', fontsize=14)
ax_coef.set_ylabel('Coefficient Value')
ax_coef.grid(True, linestyle='--', alpha=0.6, axis='y')
for bar in bars_coef:
    height = bar.get_height()
    ax_coef.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.2f}', 
                 ha='center', va='bottom', fontsize=12)

plt.show()
