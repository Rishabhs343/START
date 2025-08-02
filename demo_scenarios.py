from openpyxl import load_workbook

def demo_scenarios():
    print("🎭 DEMO: DIFFERENT SCENARIOS")
    print("=" * 60)
    
    # Load the workbook (with calculated values)
    wb = load_workbook("BonusSheet-Aug2025.xlsx", data_only=True)
    ws = wb.active
    
    print("📊 SCENARIO 1: Perfect Performance (Default)")
    print("-" * 40)
    
    # Calculate total with default values
    total_points = 0
    for row in range(2, 28):  # 26 workdays
        # Default: 9 completed hrs * 1 complexity * 1 QA * 100% efficiency = 9 points/day
        total_points += 9
    
    base_salary = 50000
    bonus_rate = (base_salary * 0.5) / (26 * 10)  # ₹96.15 per point
    monthly_bonus = min(total_points * bonus_rate, base_salary * 0.5)
    total_pay = base_salary + monthly_bonus
    
    print(f"   Total Points: {total_points}")
    print(f"   Bonus Rate: ₹{bonus_rate:.2f} per point")
    print(f"   Monthly Bonus: ₹{monthly_bonus:,.0f}")
    print(f"   Total Pay: ₹{total_pay:,.0f}")
    
    print("\n📊 SCENARIO 2: With Leave Days")
    print("-" * 40)
    
    # Assume 3 leave days
    leave_days = 3
    working_days = 26 - leave_days
    points_with_leave = working_days * 9
    bonus_with_leave = min(points_with_leave * bonus_rate, base_salary * 0.5)
    total_with_leave = base_salary + bonus_with_leave
    
    print(f"   Working Days: {working_days} (3 leave days)")
    print(f"   Total Points: {points_with_leave}")
    print(f"   Monthly Bonus: ₹{bonus_with_leave:,.0f}")
    print(f"   Total Pay: ₹{total_with_leave:,.0f}")
    print(f"   Loss due to leave: ₹{total_pay - total_with_leave:,.0f}")
    
    print("\n📊 SCENARIO 3: With Task Failures")
    print("-" * 40)
    
    # Assume 2 task failures
    failed_days = 2
    successful_days = 26 - failed_days
    points_with_failures = successful_days * 9
    bonus_with_failures = min(points_with_failures * bonus_rate, base_salary * 0.5)
    total_with_failures = base_salary + bonus_with_failures
    
    print(f"   Successful Days: {successful_days} (2 task failures)")
    print(f"   Total Points: {points_with_failures}")
    print(f"   Monthly Bonus: ₹{bonus_with_failures:,.0f}")
    print(f"   Total Pay: ₹{total_with_failures:,.0f}")
    print(f"   Loss due to failures: ₹{total_pay - total_with_failures:,.0f}")
    
    print("\n📊 SCENARIO 4: High Performance with Overtime")
    print("-" * 40)
    
    # Assume average 11 hours completed (2 OT per day)
    completed_hrs_avg = 11
    complexity_avg = 1.2  # Slightly complex tasks
    qa_avg = 1.1  # Good quality
    
    raw_points_per_day = completed_hrs_avg * complexity_avg * qa_avg
    efficiency_per_day = completed_hrs_avg / 9  # Assuming 9 available hours
    approved_points_per_day = efficiency_per_day * raw_points_per_day
    
    total_high_performance = 26 * approved_points_per_day
    bonus_high_performance = min(total_high_performance * bonus_rate, base_salary * 0.5)
    total_pay_high = base_salary + bonus_high_performance
    
    print(f"   Avg Daily Hours: {completed_hrs_avg}")
    print(f"   Avg Complexity: {complexity_avg}")
    print(f"   Avg QA Factor: {qa_avg}")
    print(f"   Avg Efficiency: {efficiency_per_day:.1%}")
    print(f"   Points per Day: {approved_points_per_day:.1f}")
    print(f"   Total Points: {total_high_performance:.0f}")
    print(f"   Monthly Bonus: ₹{bonus_high_performance:,.0f} (CAPPED)")
    print(f"   Total Pay: ₹{total_pay_high:,.0f}")
    
    print("\n📊 SCENARIO 5: Different Base Salaries")
    print("-" * 40)
    
    salaries = [30000, 40000, 50000, 75000, 100000]
    
    for salary in salaries:
        rate = (salary * 0.5) / (26 * 10)
        bonus = min(total_points * rate, salary * 0.5)
        total = salary + bonus
        print(f"   Base ₹{salary:,} → Bonus ₹{bonus:,.0f} → Total ₹{total:,}")
    
    print("\n" + "=" * 60)
    print("✨ KEY INSIGHTS:")
    print("   • Bonus is ALWAYS capped at 50% of base salary")
    print("   • Leave days = 0 points for that day")
    print("   • Task failures = 0 approved points for that day")
    print("   • Overtime increases efficiency and total points")
    print("   • Higher base salary = higher bonus potential")
    print("   • Max possible points = 26 days × 10 points = 260 points")

if __name__ == "__main__":
    demo_scenarios()