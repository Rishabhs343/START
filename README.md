# 🏆 Employee Performance Tracker & Bonus Calculator

## 📋 Overview

A comprehensive Excel-based system for tracking daily employee performance and automatically calculating monthly bonuses based on efficiency, task completion, and various performance factors.

## 📁 Files Generated

- **`BonusSheet-Aug2025.xlsx`** - Main Excel workbook with formulas and protection
- **`bonus_sheet_aug2025.py`** - Python script to generate the Excel sheet
- **`validate_bonus_sheet.py`** - Validation script to verify formulas
- **`demo_scenarios.py`** - Demonstration of different performance scenarios

## 🗂️ Sheet Structure

### Daily Performance Tracking Table (Rows 2-27)

| Column | Field | Type | Description |
|--------|-------|------|-------------|
| A | Date | Auto | Pre-filled Mon-Sat dates for August 2025 |
| B | Day | Auto | Day of week (Mon, Tue, etc.) |
| C | Meeting Hrs | Input | Hours spent in meetings (default: 0) |
| D | Assigned Hrs | Input | Hours assigned for work (default: 9) |
| E | Completed Hrs | Input | Hours of work completed (default: 9) |
| F | Complexity Factor | Input | Task complexity multiplier (default: 1) |
| G | QA Factor | Input | Quality assurance multiplier (default: 1) |
| H | Task Failed (Y/N) | Input | Whether task failed (default: N) |
| I | Leave Taken (Y/N) | Input | Whether on leave (default: N) |
| J | Auto-adjust Available Hrs | Formula | `=IF(Leave="Y", 0, 9-Meeting Hrs)` |
| K | % Efficiency | Formula | `=IF(Available=0, 0, Completed/Available)` |
| L | Raw Points | Formula | `=Completed × Complexity × QA` |
| M | OT Points | Formula | `=IF(Completed > Assigned, Completed-Assigned, 0)` |
| N | Approved Points | Formula | `=IF(Task Failed="Y", 0, ROUND(Efficiency × Raw Points, 2))` |

### Summary Section (Rows 31-36)

| Label | Formula | Description |
|-------|---------|-------------|
| Total Earned Points | `=SUM(N2:N27)` | Sum of all approved points |
| Workdays (W) | 26 | Total Mon-Sat workdays in August |
| Base Salary | ₹50,000 | Monthly base salary (editable) |
| Bonus Rate/Point | `=Base × 0.5 ÷ (W × 10)` | Rate per performance point |
| Monthly Bonus | `=MIN(Points × Rate, Base × 0.5)` | Bonus amount (capped at 50%) |
| Total Pay | `=Base + Bonus` | Total monthly compensation |

## 🔐 Protection & Security

- **Sheet Protection**: Enabled with password `secure123`
- **Locked Cells**: All formula cells are protected from editing
- **Unlocked Cells**: Only input fields are editable
  - Meeting Hrs, Assigned Hrs, Completed Hrs
  - Complexity Factor, QA Factor
  - Task Failed (Y/N), Leave Taken (Y/N)
  - Base Salary in summary section

## 🎯 Key Features

### ✅ Automatic Calculations
- **Efficiency**: Dynamically calculated based on available vs completed hours
- **Points System**: Raw points multiplied by complexity and QA factors
- **Overtime Tracking**: Extra points for hours beyond assigned time
- **Bonus Capping**: Maximum 50% of base salary regardless of performance

### ✅ Smart Logic
- **Leave Days**: Zero points when leave is taken
- **Task Failures**: Zero approved points for failed tasks
- **Meeting Time**: Automatically reduces available working hours
- **Error Prevention**: Division by zero protection in efficiency calculations

### ✅ Professional Formatting
- **Color Coding**: 
  - Blue headers with white text
  - Yellow background for input fields
  - Gray background for formula fields
  - Green background for summary labels
- **Borders**: Clean table borders throughout
- **Currency Format**: Proper ₹ formatting for monetary values
- **Percentage Format**: Efficiency displayed as percentage

## 📊 Example Scenarios

### Perfect Performance (Default Values)
- 26 workdays × 9 hours = 234 points
- Bonus: ₹22,500 (96.15 per point)
- Total Pay: ₹72,500

### With 3 Leave Days
- 23 working days × 9 hours = 207 points
- Bonus: ₹19,904
- Loss: ₹2,596

### With High Performance (11 hrs/day, 1.2 complexity, 1.1 QA)
- Total Points: 461 (efficiency bonus included)
- Bonus: ₹25,000 (CAPPED at 50%)
- Total Pay: ₹75,000

## 🚀 Usage Instructions

### Generate New Sheet
```bash
python3 bonus_sheet_aug2025.py
```

### Validate Formulas
```bash
python3 validate_bonus_sheet.py
```

### Run Scenario Analysis
```bash
python3 demo_scenarios.py
```

## 🔧 Customization

### Change Month/Year
Modify the `start_date` in the Python script:
```python
start_date = datetime(2025, 9, 1)  # For September 2025
```

### Adjust Base Salary
Change the default in the script or edit directly in Excel:
```python
base_salary = 75000  # New default
```

### Modify Bonus Cap
Update the formula to change the 50% cap:
```python
# Change 0.5 to desired percentage (e.g., 0.3 for 30%)
bonus_rate = f'=N{summary_start_row+2}*0.3/(N{summary_start_row+1}*10)'
```

## 📈 Performance Metrics

- **Maximum Points/Day**: 10 (assumed for bonus rate calculation)
- **Efficiency Range**: 0% to unlimited (>100% for overtime)
- **Bonus Range**: ₹0 to 50% of base salary
- **Workdays**: 26 (Mon-Sat in August 2025)

## ⚠️ Important Notes

1. **Password Protected**: Use `secure123` to unprotect sheet for structural changes
2. **Formula Integrity**: Only input fields should be modified during normal use
3. **Backup Recommended**: Keep original file before making changes
4. **Excel Compatibility**: Tested with modern Excel versions and LibreOffice Calc

## 🎯 Business Logic Summary

The system encourages:
- ✅ **Consistent Performance**: Daily tracking promotes regular work habits
- ✅ **Quality Focus**: QA factor rewards high-quality output
- ✅ **Efficiency**: Rewards completing more work in less time
- ✅ **Accountability**: Task failure tracking promotes responsibility
- ✅ **Transparency**: All calculations are visible and auditable

The bonus cap ensures:
- 💰 **Cost Control**: Maximum 50% salary increase regardless of performance
- 📊 **Fair Distribution**: Bonus rate scales with base salary
- 🎯 **Motivation**: Still rewards exceptional performance within limits 
