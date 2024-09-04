import random
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['medical_reports']
collection = db['cbc_reports']

# Function to generate random values typical in dengue cases
def generate_dengue_cbc_report():
    return {
        "test_description": "Complete Blood Count (CBC)",
        "haematology_results": {
            "hemoglobin": {
                "result": round(random.uniform(10, 12), 1),
                "units": "gm%",
                "biological_reference_intervals": "11.5-16"
            },
            "total_wbc_count": {
                "result": random.randint(2500, 4000),
                "units": "Cells/cumm",
                "biological_reference_intervals": "4000-11000"
            },
            "differential_count": {
                "neutrophils": {
                    "result": random.randint(40, 70),
                    "units": "%",
                    "biological_reference_intervals": "40-70"
                },
                "lymphocytes": {
                    "result": random.randint(20, 45),
                    "units": "%",
                    "biological_reference_intervals": "20-45"
                },
                "eosinophils": {
                    "result": random.randint(1, 6),
                    "units": "%",
                    "biological_reference_intervals": "1-6"
                },
                "monocytes": {
                    "result": random.randint(1, 10),
                    "units": "%",
                    "biological_reference_intervals": "1-10"
                },
                "basophils": {
                    "result": random.randint(0, 1),
                    "units": "%",
                    "biological_reference_intervals": "0-1"
                }
            },
            "rbc_count": {
                "result": round(random.uniform(3.0, 4.5), 2),
                "units": "Million/cumm",
                "biological_reference_intervals": "3.50-5.50"
            },
            "platelet_count": {
                "result": round(random.uniform(0.5, 1.5), 2),
                "units": "Lakhs/cumm",
                "biological_reference_intervals": "1.5-4.5"
            },
            "pcv": {
                "result": round(random.uniform(30, 35), 1),
                "units": "%",
                "biological_reference_intervals": "35-55"
            },
            "mcv": {
                "result": round(random.uniform(75, 95), 1),
                "units": "fl",
                "biological_reference_intervals": "75-95"
            },
            "mch": {
                "result": round(random.uniform(26, 32), 1),
                "units": "pg",
                "biological_reference_intervals": "26-32"
            },
            "mchc": {
                "result": round(random.uniform(31, 36), 1),
                "units": "gm/dl",
                "biological_reference_intervals": "31-36"
            },
            "rdw": {
                "result": round(random.uniform(11.5, 14.5), 1),
                "units": "%",
                "biological_reference_intervals": "11.5-14.5"
            }
        }
    }

# Generate and insert 14 reports
reports = [generate_dengue_cbc_report() for _ in range(14)]
collection.insert_many(reports)

print(f"{len(reports)} reports inserted into the 'cbc_reports' collection.")
