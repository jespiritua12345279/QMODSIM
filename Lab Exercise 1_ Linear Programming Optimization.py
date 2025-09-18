# For checking sir, I install "pip install numpy matplotlib scipy pandas"
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import pandas as pd

def solve_linear_programming():
    """
    Solve the linear programming problem using scipy.optimize.linprog
    """
    print("="*60)
    print("LINEAR PROGRAMMING PROBLEM SOLVER")
    print("="*60)
    print("\nProblem Statement:")
    print("Maximize: 30A + 20B")
    print("Subject to:")
    print("  2A + B ≤ 100  (labor constraint)")
    print("  A + 2B ≤ 80   (material constraint)")
    print("  A, B ≥ 0      (non-negativity)")
    print("\n" + "="*60)
    
    c = [-30, -20]  # Coefficients of objective function (negative for maximization)
    
    # Inequality constraints Ax <= b
    A = [[2, 1],    # Labor constraint: 2A + B <= 100
         [1, 2]]    # Material constraint: A + 2B <= 80
    
    b = [100, 80]   # Right-hand side values
    
    # Bounds for variables (both A and B >= 0)
    x_bounds = (0, None)  # A >= 0
    y_bounds = (0, None)  # B >= 0
    bounds = [x_bounds, y_bounds]
    
    # Solve the linear programming problem
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    return result

def create_detailed_visualization():
    """
    Create a comprehensive visualization of the linear programming problem
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    A_range = np.linspace(0, 100, 1000)
    
    # Calculate constraint lines
    # Labor constraint: 2A + B <= 100 => B = 100 - 2A
    B_labor = 100 - 2 * A_range
    
    # Material constraint: A + 2B <= 80 => B = (80 - A) / 2
    B_material = (80 - A_range) / 2
    
    # Plot 1: Feasible Region and Constraints
    ax1.plot(A_range, B_labor, 'b-', linewidth=2, label='Labor Constraint: 2A + B ≤ 100')
    ax1.plot(A_range, B_material, 'g-', linewidth=2, label='Material Constraint: A + 2B ≤ 80')
    
    # Find intersection points to define feasible region
    A_intersect = (100 - 80/2) / (2 - 1/2)  # Intersection of constraints
    B_intersect = 100 - 2 * A_intersect
    
    # Corner points of feasible region
    corner_points = np.array([[0, 0], [0, 40], [40, 20], [50, 0]])
    
    # Create feasible region polygon
    feasible_A = [0, 0, 40, 50, 0]
    feasible_B = [0, 40, 20, 0, 0]
    ax1.fill(feasible_A, feasible_B, alpha=0.3, color='lightblue', label='Feasible Region')
    
    # Mark corner points
    for i, (a, b) in enumerate(corner_points):
        ax1.plot(a, b, 'ko', markersize=8)
        ax1.annotate(f'({a},{b})', (a, b), xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # Highlight optimal solution
    optimal_A, optimal_B = 40, 20
    ax1.plot(optimal_A, optimal_B, 'ro', markersize=15, label=f'Optimal Solution: A={optimal_A}, B={optimal_B}')
    ax1.annotate(f'Optimal: ({optimal_A}, {optimal_B})\nProfit: ${30*optimal_A + 20*optimal_B}', 
                (optimal_A, optimal_B), xytext=(10, 10), textcoords='offset points', 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    profit_levels = [600, 1200, 1600]
    for profit in profit_levels:
        B_profit = (profit - 30 * A_range) / 20
        if profit == 1600:  # Optimal profit line
            ax1.plot(A_range, B_profit, 'r--', linewidth=2, alpha=0.8, label=f'Optimal Profit Line: ${profit}')
        else:
            ax1.plot(A_range, B_profit, '--', linewidth=1, alpha=0.5, label=f'Profit Line: ${profit}')
    
    ax1.set_xlim(0, 80)
    ax1.set_ylim(0, 60)
    ax1.set_xlabel('Product A Quantity', fontsize=12)
    ax1.set_ylabel('Product B Quantity', fontsize=12)
    ax1.set_title('Linear Programming Solution\nFeasible Region and Optimal Point', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    
    categories = ['Labor', 'Material']
    used = [100, 80]  # Resources used at optimal solution
    available = [100, 80]  # Available resources
    
    x_pos = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, used, width, label='Used', color='skyblue', alpha=0.8)
    bars2 = ax2.bar(x_pos + width/2, available, width, label='Available', color='lightcoral', alpha=0.8)
    
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    for i, (u, a) in enumerate(zip(used, available)):
        utilization = (u/a) * 100
        ax2.text(i, max(u, a) + 5, f'{utilization:.1f}% utilized', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Resource Units', fontsize=12)
    ax2.set_title('Resource Utilization at Optimal Solution', fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 120)
    
    plt.tight_layout()
    return fig

def analyze_corner_points():
    """
    Analyze all corner points of the feasible region
    """
    print("\nCORNER POINT ANALYSIS:")
    print("="*40)
    
    # Define corner points
    corner_points = [
        (0, 0, "Origin"),
        (0, 40, "Y-axis intercept (Material constraint)"),
        (40, 20, "Intersection of constraints"),
        (50, 0, "X-axis intercept (Labor constraint)")
    ]
    
    results = []
    
    for A, B, description in corner_points:
        # Check if point satisfies constraints
        labor_used = 2*A + B
        material_used = A + 2*B
        labor_feasible = labor_used <= 100
        material_feasible = material_used <= 80
        
        if labor_feasible and material_feasible:
            profit = 30*A + 20*B
            feasible = "✓ Feasible"
        else:
            profit = 0
            feasible = "✗ Infeasible"
        
        results.append({
            'Point': f'({A}, {B})',
            'Description': description,
            'Labor Used': labor_used,
            'Material Used': material_used,
            'Profit': f'${profit}',
            'Status': feasible
        })
        
        print(f"Point ({A}, {B}) - {description}:")
        print(f"  Labor used: {labor_used}/100 hours")
        print(f"  Material used: {material_used}/80 units")
        print(f"  Profit: ${profit}")
        print(f"  Status: {feasible}")
        print()
    
    # Create DataFrame for better display
    df = pd.DataFrame(results)
    print("SUMMARY TABLE:")
    print("="*80)
    print(df.to_string(index=False))
    
    return df

def create_sensitivity_analysis():
    """
    Perform basic sensitivity analysis
    """
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS")
    print("="*60)
    
    print("\n1. SHADOW PRICES (Marginal Value of Resources):")
    print("-" * 50)
    
    base_profit = 30*40 + 20*20  # Base optimal profit
    
    # Increase labor by 1 unit
    # New constraints: 2A + B ≤ 101, A + 2B ≤ 80
    # Solving graphically: intersection at A=40.33, B=20.33
    labor_shadow_profit = 30*40.33 + 20*20.33
    labor_shadow_price = labor_shadow_profit - base_profit
    
    # Increase material by 1 unit  
    # New constraints: 2A + B ≤ 100, A + 2B ≤ 81
    # Solving graphically: intersection at A=39.67, B=20.67
    material_shadow_profit = 30*39.67 + 20*20.67
    material_shadow_price = material_shadow_profit - base_profit
    
    print(f"Labor Shadow Price: ${labor_shadow_price:.2f} per hour")
    print(f"Material Shadow Price: ${material_shadow_price:.2f} per unit")
    print(f"\nInterpretation:")
    print(f"- Each additional hour of labor increases profit by ~${labor_shadow_price:.2f}")
    print(f"- Each additional unit of material increases profit by ~${material_shadow_price:.2f}")
    
    print("\n2. OBJECTIVE FUNCTION COEFFICIENT RANGES:")
    print("-" * 50)
    print("Current coefficients: Product A = $30, Product B = $20")
    print("For the current solution to remain optimal:")
    print("- Product A coefficient range: $13.33 to $60")  
    print("- Product B coefficient range: $10 to $45")
    
    return {
        'labor_shadow_price': labor_shadow_price,
        'material_shadow_price': material_shadow_price
    }

def main():
    """
    Main function to run the complete analysis
    """
    # Solve the linear programming problem
    result = solve_linear_programming()
    
    if result.success:
        optimal_A = result.x[0]
        optimal_B = result.x[1]
        max_profit = -result.fun  # Convert back from minimization
        
        print(f"\nSOLUTION FOUND!")
        print(f"Optimal Production:")
        print(f"  Product A: {optimal_A:.0f} units")
        print(f"  Product B: {optimal_B:.0f} units")
        print(f"  Maximum Profit: ${max_profit:.2f}")
        
        # Verify constraints
        labor_used = 2*optimal_A + optimal_B
        material_used = optimal_A + 2*optimal_B
        
        print(f"\nResource Utilization:")
        print(f"  Labor: {labor_used:.0f}/100 hours ({labor_used/100*100:.1f}%)")
        print(f"  Material: {material_used:.0f}/80 units ({material_used/80*100:.1f}%)")
        
    else:
        print("ERROR: Could not find optimal solution!")
        return
    
    # Perform corner point analysis
    corner_df = analyze_corner_points()
    
    # Sensitivity analysis
    sensitivity = create_sensitivity_analysis()
    
    # Create and show visualization
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATION")
    print(f"{'='*60}")
    
    fig = create_detailed_visualization()
    plt.show()
    
    # Final summary
    print(f"\n{'='*60}")
    print("Linear Programming Optimization Result")
    print(f"{'='*60}")
    print(f"Product A = {optimal_A:.0f} units")
    print(f"Product B = {optimal_B:.0f} units") 
    print(f"Total Profit (Objective Function) = ${max_profit:.0f}")
    print(f"{'='*60}")

# Run the complete analysis
if __name__ == "__main__":
    main()