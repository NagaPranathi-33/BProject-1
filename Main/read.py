import csv
import sys
import os
import pandas as pd
# # Add the project root directory to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# # Now import  

def read_data(data):
    #Read data from the csv file
    #file_name = "Preprocessed//Preprocessed_"+data+".csv"                  #dataset location
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current script directory
    file_name = os.path.join(base_dir, "Preprocessed", f"{data}.csv")  
    datas = []
    with open(file_name, 'rt')as f:
        content = csv.reader(f)                        #read csv content
        for rows in content:                           #row of data
            tem = []
            for cols in rows:                          
                #tem.append(float(cols))
                if cols == '' or cols.isspace():
                    tem.append(0.0)  # or use np.nan if your model handles missing values
                else:
                    #tem.append(float(cols))  
                    try:
                        tem.append(float(cols))
                    except ValueError:
                        # Replace missing or invalid values with 0 (or any appropriate default)
                        tem.append(0.0)          
            datas.append(tem)                         
    if data=='Adult':
        d = len(datas)//3
        datas = datas[:d]
    return datas




def read_label(data):
    """Reads label data from a CSV file."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(base_dir, "Preprocessed", f"{data}.csv")  
    
    # ✅ Check if file exists before proceeding
    if not os.path.exists(file_name):
        print(f"❌ Error: Label file '{data}.csv' not found in 'Preprocessed' folder.")
        return None

    datas = []
    
    try:
        with open(file_name, 'rt') as f:
            content = csv.reader(f)  # Read CSV content
            
            for rows in content:
                if not rows:  # ✅ Skip empty rows
                    continue
                try:
                    # ✅ Convert values safely and handle errors
                    datas.append(int(float(rows[0])))  
                except ValueError:
                    print(f"⚠️ Warning: Skipping invalid label value: {rows}")
                    continue

        # ✅ Handle dataset-specific slicing (if needed)
        if data == 'Adult':
            d = len(datas) // 3
            datas = datas[:d]

        return datas

    except Exception as e:
        print(f"❌ Exception while reading labels: {e}")
        return None
