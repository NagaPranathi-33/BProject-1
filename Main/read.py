import csv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_DIR = os.path.join(BASE_DIR, "Preprocessed")


def _candidate_files(data, suffix=""):
    return [
        os.path.join(PREPROCESSED_DIR, f"{data}{suffix}.csv"),
        os.path.join(PREPROCESSED_DIR, f"Preprocessed_{data}{suffix}.csv"),
    ]


def _first_existing(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def read_data(data):
    # Read data from the csv file
    file_name = _first_existing(_candidate_files(data))
    if file_name is None:
        raise FileNotFoundError(
            f"Data file not found for '{data}'. Looked in: {_candidate_files(data)}"
        )

    datas = []
    with open(file_name, "rt") as f:
        content = csv.reader(f)
        for rows in content:
            tem = []
            for cols in rows:
                if cols == "" or cols.isspace():
                    tem.append(0.0)
                else:
                    try:
                        tem.append(float(cols))
                    except ValueError:
                        tem.append(0.0)
            datas.append(tem)

    if data == "Adult":
        d = len(datas) // 3
        datas = datas[:d]
    return datas


def read_label(data):
    """Reads label data from a CSV file."""
    # Primary: explicit label file.
    file_name = _first_existing(_candidate_files(data, "_label"))

    # Backward compatibility fallback: older code may have only one file.
    if file_name is None:
        file_name = _first_existing(_candidate_files(data))

    if file_name is None:
        print(f"❌ Error: Label file '{data}_label.csv' not found in 'Preprocessed' folder.")
        return None

    datas = []
    try:
        with open(file_name, "rt") as f:
            content = csv.reader(f)
            for rows in content:
                if not rows:
                    continue
                value = rows[0] if file_name.endswith("_label.csv") else rows[-1]
                try:
                    datas.append(int(float(value)))
                except ValueError:
                    print(f"⚠️ Warning: Skipping invalid label value: {rows}")
                    continue

        if data == "Adult":
            d = len(datas) // 3
            datas = datas[:d]

        return datas

    except Exception as e:
        print(f"❌ Exception while reading labels: {e}")
        return None
