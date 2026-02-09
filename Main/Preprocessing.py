import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_DIR = os.path.join(BASE_DIR, "Preprocessed")
os.makedirs(PREPROCESSED_DIR, exist_ok=True)


def _dataset_candidates(dts):
    return [
        os.path.join(BASE_DIR, "dataset", f"{dts}.csv"),
        os.path.join(BASE_DIR, "Datasets", f"{dts}.csv"),
        os.path.join(BASE_DIR, "datasets", f"{dts}.csv"),
    ]


def resolve_dataset_csv(dts):
    # Keep explicit alias for legacy naming used by 7817_1_cleaned pathing.
    alias = {
        "7817_1_cleaned": "Amazon_Reviews",
    }
    names = [dts]
    if dts in alias:
        names.append(alias[dts])

    for name in names:
        for candidate in _dataset_candidates(name):
            if os.path.exists(candidate):
                return candidate

    search = ", ".join(_dataset_candidates(dts))
    raise FileNotFoundError(f"Dataset CSV not found for {dts}. Looked in: {search}")
import pandas as pd
import numpy as np
from scipy.stats import boxcox
from csv import reader

# def safe_boxcox(column):
#     # Convert to float type
#     column = column.astype(float)
#     # Find minimum value
#     min_val = np.min(column)
#     # If any value is <= 0, shift the entire column
#     if min_val <= 0:
#         column = column + abs(min_val) + 1  # make all values strictly positive
#     # Apply Box-Cox
#     transformed_col, best_lambda = boxcox(column)
    

#     return transformed_col, best_lambda


def safe_boxcox(column):
    """Safely applies Box-Cox transformation if data is suitable."""
    if np.all(column == column[0]):
        # Data is constant
        print("Skipping Box-Cox: constant column")
        return column, None
    try:
        # Box-Cox requires strictly positive values
        column = np.where(column <= 0, 1e-6, column)
        transformed_col, best_lambda = boxcox(column)
        return transformed_col, best_lambda
    except Exception as e:
        print(f"Skipping Box-Cox for this column due to: {e}")
        return column, None

def preprocessing(dts):

    def string_conversion(dts):
        # filename = 'dataset/' + dts + '.csv'  # dataset path
        filename = resolve_dataset_csv(dts)
        def load_csv(filename):  # read csv file
            dataset = list()
            with open(filename, 'r') as file:
                csv_reader = reader(file)
                for row in csv_reader:
                    if not row:
                        continue
                    dataset.append(row)
            return dataset

        # def convert(data):
        #     datas=[]
        #     for i in range(len(data)):
        #         tem=[]
        #         for j in range(len(data[i])):
        #             if data[i][j] =='0.0':
        #                 tem.append(0.1)
        #             else:
        #                 tem.append((float(data[i][j])))
        #         datas.append(tem)
        #     return datas

        def convert(data):
            temp = []
            for i in range(len(data)):
                tem = []
                for j in range(len(data[0])):
                    val = data[i][j]
                    if val == '' or val.strip() == '':  # handle empty strings
                        tem.append(0.0)  # or use np.nan if you plan to impute later
                    else:
                        try:
                            tem.append(float(val))
                        except ValueError:
                            tem.append(0.0)  # fallback for non-numeric values
                temp.append(tem)
            return np.array(temp)


        def find_class(Z,dts):
            clas = Z[:, len(Z[0]) - 1]  # slicing label
            lab = []
            if dts=='Adult':
                for i in range(len(clas)):
                    if clas[i]==' <=50K':
                        lab.append(0)
                    else:
                        lab.append(1)
            if dts=='Credit_Approval':
                for i in range(len(clas)):
                    if clas[i]=='-':
                        lab.append(0)
                    else:
                        lab.append(1)
            return lab

        def find_unique(X):
            uni = np.unique(X).tolist()
            Uni = []
            for i in range(len(uni)):
                if uni[i] != ' ?':
                    Uni.append(uni[i])
            return Uni

        def str_convert(A, x1):  # string to int conversion
            xx = []
            for i in range(len(A)):
                if A[i]==' ?':
                    xx.append('?')
                for j in range(len(x1)):
                    if (A[i] == x1[
                        j]):  # if original string value = unique string value, then store the index of unique string value in new list
                        xx.append(int(x1.index(A[i])))
            return xx

        # def find_missing(data):
        #     datas = []
        #     for i in range(len(data)):
        #         temp = []
        #         for j in range(len(data[i])):
        #             if data[i][j] == '?':  # replace '?' by '-1000'
        #                 data[i][j] = '-1000'  # for missing value imputation
        #             temp.append((float(data[i][j])))
        #             # try:
        #             #     temp.append(float(data[i][j]))
        #             # except ValueError:
        #             #     continue  # skip non-numeric values

        #         datas.append(temp)

        #     n_data = []
        #     for i in range(len(data)):
        #         temp = []
        #         for j in range(len(data[i])):
        #             if (data[i][j] != '-1000'):  # except -1000 add all other values to a list
        #                 temp.append(float(data[i][j]))
        #             else:
        #                 temp.append(0)  # to get mean of other values in column
        #         n_data.append(temp)
        #     n_data = np.array(n_data)
        #     Avg = np.mean(n_data, axis=0)  # average of column values
        #     for i in range(len(data)):
        #         for j in range(len(data[i])):
        #             if (data[i][j] == '-1000'):  # replace '-1000' by calculated average values
        #                 data[i][j] = float(Avg[j])  # missing values are replaced by column average
        #             else:
        #                 data[i][j] = float(data[i][j])
        #     data = np.array(data)
        #     return data
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        def find_missing(data):
            data = np.array(data)
            if len(data.shape) != 2:
                print(f"[Error] find_missing: Expected 2D array, but got shape {data.shape}")
                return data
            rows, cols = data.shape

            for col in range(cols):
                numeric_vals = []

                # Collect only valid numeric, non-empty entries
                for row in range(rows):
                    val = data[row][col].strip()
                    if val != '' and is_number(val):
                        numeric_vals.append(float(val))

                # If no numeric values, skip this column
                if len(numeric_vals) == 0:
                    continue

                mean_val = np.mean(numeric_vals)

                # Replace empty or non-numeric entries with mean
                for row in range(rows):
                    val = data[row][col].strip()
                    if val == '' or not is_number(val):
                        data[row][col] = str(mean_val)

            return data.tolist()




        data = load_csv(filename) # read data from csv file
        X = data[1:len(data)] # removing the headers
        if dts == 'Adult':
            ind = [1,3,5, 6, 7, 8, 9, 13]  # ind = index value of string format in Adult dataset
            X = np.array(X)
            X = X.transpose()
            for i in range(len(X)):
                if (i in ind):
                    uni = find_unique(X[i])
                    X[i] = str_convert(X[i], uni)  # string data conversion
            Z = np.transpose(X) # Transposing data
            datas = Z[:, 0:len(Z[0]) - 1]  # slicing data
            datas = find_missing(datas) # Find the missing value
            clas = find_class(Z,dts) # Class labels
            # np.savetxt("Preprocessed//" + dts + ".csv", datas, delimiter=',', fmt='%s')
            # np.savetxt("Preprocessed//" + dts + "_label.csv", clas, delimiter=',', fmt='%s')
            np.savetxt(os.path.join(PREPROCESSED_DIR, dts + ".csv"), datas, delimiter=',', fmt='%s')
            np.savetxt(os.path.join(PREPROCESSED_DIR, dts + "_label.csv"), clas, delimiter=',', fmt='%s')

        if dts == 'Credit_Approval':
            ind = [0,3,4, 5, 6, 8, 9, 11,12]  # ind = index value of string format in Adult dataset
            X = np.array(X)
            X = X.transpose()
            for i in range(len(X)):
                if (i in ind):
                    uni = find_unique(X[i])
                    X[i] = str_convert(X[i], uni)  # string data conversion
            Z = np.transpose(X)  # Transposing data
            datas = Z[:, 0:len(Z[0]) - 1]  # slicing data
            datas = find_missing(datas)  # Find the missing value
            clas = find_class(Z,dts)  # Class labels
            np.savetxt(os.path.join(PREPROCESSED_DIR, dts + ".csv"), datas, delimiter=',', fmt='%s')
            np.savetxt(os.path.join(PREPROCESSED_DIR, dts + "_label.csv"), clas, delimiter=',', fmt='%s')

        # if dts == '7817_1_cleaned':
        #     ind = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 19, 20, 22, 23, 24, 26]  # categorical string indices
        #     ignore = [0, 1, 7]  # columns like ID or names that must be dropped
        #     X = np.array(X)
        #     X = X.transpose()
    
        #     for i in range(len(X)):
        #         if i in ind:
        #             uni = find_unique(X[i])
        #             X[i] = str_convert(X[i], uni)
    
        #     Z = np.transpose(X)
        #     Z = np.delete(Z, ignore, axis=1)  # remove ID or non-numeric text columns
    
        #     datas = Z[:, :-1]  # features only
        #     datas = find_missing(datas)       # apply missing value fix only to numeric-convertible features
        #     clas = find_class(Z, dts)

        #     np.savetxt(os.path.join(PREPROCESSED_DIR, dts + ".csv"), datas, delimiter=',', fmt='%s')
        #     np.savetxt(os.path.join(PREPROCESSED_DIR, dts + "_label.csv"), clas, delimiter=',', fmt='%s')
        
        if dts == 'CreditCard':
            X = np.array(X).T

            # Detect categorical columns automatically
            categorical_indices = []
            for i in range(len(X)):
                sample = X[i][0]
                try:
                    float(sample)   # If fails → categorical
                except:
                    categorical_indices.append(i)

            # Convert detected categorical columns
            for i in categorical_indices:
                uni = find_unique(X[i])
                X[i] = str_convert(X[i], uni)

            Z = X.T
            datas = Z[:, :-1]
            datas = find_missing(datas)
            clas = Z[:, -1]

            np.savetxt(os.path.join(PREPROCESSED_DIR, "CreditCard.csv"),
                   datas, delimiter=',', fmt='%s')
            np.savetxt(os.path.join(PREPROCESSED_DIR, "CreditCard_label.csv"),
                   clas, delimiter=',', fmt='%s')
            return
        # elif dts.lower() == 'creditcard':
        #     print("Reading Credit Card Fraud Detection dataset...")

  
        #     # ✅ Path to your dataset (adjust if needed)
        #     file_path = "/content/drive/MyDrive/BProject1/146203/Main/dataset/creditcard.csv"

        #     # Load dataset
        #     df = pd.read_csv(file_path)

        #     print(f"Original shape: {df.shape}")
        #     print("Checking for missing values...")
        #     print(df.isnull().sum().sum(), "missing values found.")

        #     # Drop rows with NaN (though this dataset normally has none)
        #     df = df.dropna()

        #     # Feature-target split
        #     if 'Class' not in df.columns:
        #         raise ValueError("❌ Column 'Class' not found in creditcard.csv")

        #     X = df.drop(columns=['Class']).values  # Features
        #     y = df['Class'].values                 # Labels (0 = legit, 1 = fraud)

        #     print(f"Final shapes — Features: {X.shape}, Labels: {y.shape}")

        #     # Save to Preprocessed folder
        #     np.savetxt(
        #       "/content/drive/MyDrive/BProject1/146203/Main/Preprocessed/creditcard.csv",
        #       X, delimiter=',', fmt='%s'
        #     )

        #     np.savetxt(
        #         "/content/drive/MyDrive/BProject1/146203/Main/Preprocessed/creditcard_label.csv",
        #         y, delimiter=',', fmt='%s'
        #     )

        #     print("✅ Preprocessing complete. Files saved in /Main/Preprocessed/")

        elif dts == '7817_1_cleaned':
            # Read CSV
            df = pd.read_csv(resolve_dataset_csv(dts))

            # Drop completely irrelevant or unique identifier columns
            df.drop(columns=['id', 'asins', 'keys', 'ean', 'upc', 'reviews doRecommend'], inplace=True)

            # Handle missing values (you can customize this later)
            df.fillna("NA", inplace=True)

            # Identify categorical columns to encode
            cat_cols = ['brand', 'categories', 'colors', 'dimension', 'manufacturer', 'manufacturerNumber',
                'name', 'prices', 'reviews date',
                'reviews sourceURLs', 'reviews text', 'reviews title',
                'reviews username', 'weight']

            # Convert all values to string
            df[cat_cols] = df[cat_cols].astype(str)

            # Apply label encoding (you can improve this with OneHot if needed)
            for col in cat_cols:
                unique_vals = find_unique(df[col].values)
                df[col] = str_convert(df[col].values, unique_vals)

            # Final data preparation
            datas = df.drop(columns=['reviews rating']).values  # input features
            clas = df['reviews rating'].values  # target/label

            # Save the preprocessed data
            np.savetxt(os.path.join(PREPROCESSED_DIR, dts + ".csv"), datas, delimiter=',', fmt='%s')
            np.savetxt(os.path.join(PREPROCESSED_DIR, dts + "_label.csv"), clas, delimiter=',', fmt='%s')
        elif dts == 'adult_UCI':
            ind = [1, 3, 5, 6, 7, 8, 9, 13]  # adjust these based on adult_UCI column types
            filename = resolve_dataset_csv(dts)
    
            data = load_csv(filename)
            X = np.array(data[1:])  # skip header row
            X = X.transpose()
    
            # Convert categorical columns
            for i in range(len(X)):
                if i in ind:
                    uni = find_unique(X[i])
                    X[i] = str_convert(X[i], uni)
    
            Z = np.transpose(X)
            datas = Z[:, :-1]  # all columns except target
            datas = find_missing(datas)
            clas = find_class(Z, dts)  # you may need to modify this for adult_UCI labels
    
            # Save the preprocessed files
            np.savetxt(os.path.join(PREPROCESSED_DIR, "adult_UCI.csv"),
               datas, delimiter=',', fmt='%s')
            np.savetxt(os.path.join(PREPROCESSED_DIR, "adult_UCI_label.csv"),
               clas, delimiter=',', fmt='%s')
               
        # datas = convert(datas)
        # return datas
        # ✅ Actually call the function and store result
        datas = load_csv(filename)

        # ✅ Now safely convert the loaded data
        datas = convert(datas)

        return datas

    data = string_conversion(dts)

    #################### Box-Cox Transformation ##################
    datas = np.array(data).astype(str)      # Ensure all string format
    datas[datas == ''] = '0'                # Replace empty strings with zero
    datas = datas.astype(float)             # Convert to float safely

    datas = datas.T                         # Transpose for column-wise operation
    box_cox = []
    for i in range(datas.shape[0]):
        transformed_data, best_lambda = safe_boxcox(datas[i])
        box_cox.append(transformed_data)

    box_cox = np.array(box_cox).T           # Transpose back
    np.savetxt(os.path.join(PREPROCESSED_DIR, "Preprocessed_" + dts + ".csv"),
           box_cox, delimiter=',', fmt='%s')
    np.savetxt(os.path.join(PREPROCESSED_DIR, dts + ".csv"),
           box_cox, delimiter=',', fmt='%s')

    # data = string_conversion(dts)
    # ####################Box-Cox Transformation##################
    # datas = np.array(data) # converting list to array
    # box_cox = []
    # for i in range(len(datas)):
    #     # perform Box-Cox transformation on original data
    #     #transformed_data, best_lambda = boxcox(datas[i])
    #     transformed_data, best_lambda = safe_boxcox(datas[i])

    #     box_cox.append(transformed_data)
    # np.savetxt("Preprocessed//Preprocessed_"+dts+".csv", box_cox, delimiter=',', fmt='%s')

def transformation(dts):
    print("\nBox-Cox transformation of data..")
    preprocessing(dts)

