import pandas as pd
import os

def convert_xlsx_to_csv(xlsx_file):
    if not xlsx_file.endswith(".xlsx"):
        raise ValueError("Please provide a valid .xlsx file!")

    # Load the Excel file
    excel = pd.ExcelFile(xlsx_file)
    
    # Extract base file name and directory
    file_name = os.path.splitext(os.path.basename(xlsx_file))[0]
    output_dir = os.path.dirname(xlsx_file)

    # Convert each sheet to a CSV
    for sheet_name in excel.sheet_names:
        df = pd.read_excel(excel, sheet_name=sheet_name)
        csv_file_path = os.path.join(output_dir, f"{file_name}_{sheet_name}.csv")
        df.to_csv(csv_file_path, index=False)
        print(f"✅ Saved: {csv_file_path}")

if __name__ == "__main__":
    xlsx_path = "Mustard_oil.xlsx"
    
    convert_xlsx_to_csv(xlsx_path)