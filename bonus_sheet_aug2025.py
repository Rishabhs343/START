from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.styles.protection import Protection
from openpyxl.utils import get_column_letter
from datetime import datetime, timedelta

# Initialize workbook and sheet
wb = Workbook()
ws = wb.active
ws.title = "BonusSheet-Aug2025"

# Headers for the main tracking table
headers = [
    "Date", "Day", "Meeting Hrs", "Assigned Hrs", "Completed Hrs",
    "Complexity Factor", "QA Factor", "Task Failed (Y/N)", "Leave Taken (Y/N)",
    "Auto-adjust Available Hrs", "% Efficiency", "Raw Points", "OT Points", "Approved Points"
]

# Summary labels
summary_labels = [
    "Total Earned Points", "Workdays (W)", "Base Salary", "Bonus Rate/Point",
    "Monthly Bonus", "Total Pay (Basic + Bonus)"
]

# Styling
header_font = Font(bold=True, color="FFFFFF")
header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
summary_label_font = Font(bold=True)
summary_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
input_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
formula_fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
alignment_center = Alignment(horizontal="center", vertical="center")
alignment_right = Alignment(horizontal="right", vertical="center")

# Borders
thin_border = Border(
    left=Side(style='thin'),
    right=Side(style='thin'),
    top=Side(style='thin'),
    bottom=Side(style='thin')
)

# Add headers
for col_num, header in enumerate(headers, 1):
    cell = ws.cell(row=1, column=col_num, value=header)
    cell.font = header_font
    cell.fill = header_fill
    cell.alignment = alignment_center
    cell.border = thin_border

# Generate workdays for August 2025 (Mon-Sat only)
start_date = datetime(2025, 8, 1)
workdays = []
for i in range(31):  # August has 31 days
    day = start_date + timedelta(days=i)
    if day.weekday() < 6:  # Mon-Sat (0=Monday, 5=Saturday)
        workdays.append(day)

print(f"Generated {len(workdays)} workdays for August 2025")

# Fill data rows
for row_num, date in enumerate(workdays, 2):
    day_name = date.strftime("%a")
    
    # Date and Day (locked)
    date_cell = ws.cell(row=row_num, column=1, value=date.strftime("%Y-%m-%d"))
    date_cell.alignment = alignment_center
    date_cell.border = thin_border
    
    day_cell = ws.cell(row=row_num, column=2, value=day_name)
    day_cell.alignment = alignment_center
    day_cell.border = thin_border
    
    # Input fields (unlocked)
    meeting_hrs = ws.cell(row=row_num, column=3, value=0)
    meeting_hrs.fill = input_fill
    meeting_hrs.alignment = alignment_center
    meeting_hrs.border = thin_border
    
    assigned_hrs = ws.cell(row=row_num, column=4, value=9)
    assigned_hrs.fill = input_fill
    assigned_hrs.alignment = alignment_center
    assigned_hrs.border = thin_border
    
    completed_hrs = ws.cell(row=row_num, column=5, value=9)
    completed_hrs.fill = input_fill
    completed_hrs.alignment = alignment_center
    completed_hrs.border = thin_border
    
    complexity = ws.cell(row=row_num, column=6, value=1)
    complexity.fill = input_fill
    complexity.alignment = alignment_center
    complexity.border = thin_border
    
    qa_factor = ws.cell(row=row_num, column=7, value=1)
    qa_factor.fill = input_fill
    qa_factor.alignment = alignment_center
    qa_factor.border = thin_border
    
    task_failed = ws.cell(row=row_num, column=8, value="N")
    task_failed.fill = input_fill
    task_failed.alignment = alignment_center
    task_failed.border = thin_border
    
    leave_taken = ws.cell(row=row_num, column=9, value="N")
    leave_taken.fill = input_fill
    leave_taken.alignment = alignment_center
    leave_taken.border = thin_border
    
    # Formula fields (locked)
    # Auto-adjust Available Hrs = IF(Leave="Y", 0, 9 - Meeting Hrs)
    available_hrs = ws.cell(row=row_num, column=10, value=f'=IF(I{row_num}="Y",0,9-C{row_num})')
    available_hrs.fill = formula_fill
    available_hrs.alignment = alignment_center
    available_hrs.border = thin_border
    
    # % Efficiency = IF(Available=0, 0, Completed / Available)
    efficiency = ws.cell(row=row_num, column=11, value=f'=IF(J{row_num}=0,0,E{row_num}/J{row_num})')
    efficiency.fill = formula_fill
    efficiency.alignment = alignment_center
    efficiency.border = thin_border
    efficiency.number_format = '0.00%'
    
    # Raw Points = Completed * Complexity * QA
    raw_points = ws.cell(row=row_num, column=12, value=f'=E{row_num}*F{row_num}*G{row_num}')
    raw_points.fill = formula_fill
    raw_points.alignment = alignment_center
    raw_points.border = thin_border
    
    # OT Points = IF(Completed > Assigned, Completed - Assigned, 0)
    ot_points = ws.cell(row=row_num, column=13, value=f'=IF(E{row_num}>D{row_num},E{row_num}-D{row_num},0)')
    ot_points.fill = formula_fill
    ot_points.alignment = alignment_center
    ot_points.border = thin_border
    
    # Approved Points = IF(Task Failed = "Y", 0, ROUND(% Efficiency * Raw Points, 2))
    approved_points = ws.cell(row=row_num, column=14, value=f'=IF(H{row_num}="Y",0,ROUND(K{row_num}*L{row_num},2))')
    approved_points.fill = formula_fill
    approved_points.alignment = alignment_center
    approved_points.border = thin_border

# Summary section (starting at row 35)
summary_start_row = len(workdays) + 5
workdays_count = len(workdays)
base_salary = 50000

# Summary calculations with correct formulas
summary_values = [
    f'=SUM(N2:N{len(workdays)+1})',  # Total Earned Points
    workdays_count,                   # Workdays (W)
    base_salary,                     # Base Salary (editable)
    f'=N{summary_start_row+2}*0.5/(N{summary_start_row+1}*10)',  # Bonus Rate/Point = Base * 0.5 / (Workdays * 10)
    f'=MIN(N{summary_start_row}*N{summary_start_row+3},N{summary_start_row+2}*0.5)',  # Monthly Bonus (capped at 50%)
    f'=N{summary_start_row+2}+N{summary_start_row+4}'  # Total Pay
]

# Add summary section with proper formatting
for i, (label, val) in enumerate(zip(summary_labels, summary_values)):
    # Label cell
    label_cell = ws.cell(row=summary_start_row + i, column=13, value=label)
    label_cell.font = summary_label_font
    label_cell.fill = summary_fill
    label_cell.alignment = alignment_right
    label_cell.border = thin_border
    
    # Value cell
    value_cell = ws.cell(row=summary_start_row + i, column=14, value=val)
    value_cell.border = thin_border
    value_cell.alignment = alignment_center
    
    # Special formatting for currency values
    if i >= 2:  # Base Salary onwards
        value_cell.number_format = '₹#,##0'
    
    # Make Base Salary editable
    if i == 2:  # Base Salary
        value_cell.fill = input_fill

# Set column widths
column_widths = [12, 8, 12, 12, 12, 12, 10, 15, 15, 18, 12, 12, 10, 15]
for col, width in enumerate(column_widths, 1):
    ws.column_dimensions[get_column_letter(col)].width = width

# Add title
title_cell = ws.cell(row=summary_start_row - 2, column=13, value="SUMMARY")
title_cell.font = Font(bold=True, size=14)
title_cell.alignment = alignment_center

# Protect sheet and set up unlocked cells
ws.protection.sheet = True
ws.protection.password = "secure123"

# Unlock input cells (editable fields)
for row in range(2, len(workdays) + 2):
    for col in [3, 4, 5, 6, 7, 8, 9]:  # Meeting Hrs through Leave Taken
        ws.cell(row=row, column=col).protection = Protection(locked=False)

# Unlock Base Salary in summary
ws.cell(row=summary_start_row + 2, column=14).protection = Protection(locked=False)

# Save the file
file_path = "BonusSheet-Aug2025.xlsx"
wb.save(file_path)

print(f"✅ Excel file created successfully: {file_path}")
print(f"📅 Total workdays in August 2025: {len(workdays)}")
print("🔓 Editable fields:")
print("   - Meeting Hrs, Assigned Hrs, Completed Hrs")
print("   - Complexity Factor, QA Factor")
print("   - Task Failed (Y/N), Leave Taken (Y/N)")
print("   - Base Salary in summary section")
print("🔐 Sheet protected with password: secure123")
print("\n📊 Expected behavior:")
print("   - Leave = 'Y' → Available Hrs = 0, Efficiency = 0")
print("   - Task Failed = 'Y' → Approved Points = 0")
print("   - Bonus capped at 50% of base salary")
print("   - Max points per day = 10 (for bonus rate calculation)")