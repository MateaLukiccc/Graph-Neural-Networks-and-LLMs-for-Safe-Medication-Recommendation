from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from faker import Faker
import random
import json

fake = Faker()

ICD9_CODES = {
    "401.9": "Unspecified essential hypertension",
    "250.00": "Type 2 diabetes mellitus without complications",
    "414.01": "Coronary atherosclerosis of native coronary artery",
    "427.31": "Atrial fibrillation",
    "272.4": "Other and unspecified hyperlipidemia",
    "584.9": "Acute kidney failure, unspecified",
    "530.81": "Esophageal reflux",
    "V58.69": "Long-term (current) use of other medications"
}

PROCEDURE_CODES = {
    "36.06": "Insertion of non-drug-eluting coronary artery stent(s)",
    "89.52": "Measurement of blood pressure",
    "39.61": "Extracorporeal circulation auxiliary to open heart surgery",
    "88.97": "Diagnostic ultrasound of heart",
    "99.04": "Injection or infusion of thrombolytic agent",
    "36.15": "Single vessel percutaneous transluminal coronary angioplasty (PTCA)"
}

ATC3_CODES = {
    "C09A": "ACE inhibitors, plain",
    "B01AA03": "Warfarin",
    "C07AB02": "Metoprolol",
    "A10B": "Blood glucose lowering drugs, excluding insulin",
    "N02B": "Other analgesics and antipyretics",
    "R03A": "Adrenergics, inhalants"
}

def generate_admission(keys=False):
    icd9_source = ICD9_CODES.keys() 
    proc_source = PROCEDURE_CODES.keys() 
    atc3_source = ATC3_CODES.keys() 
    
    icd9_sample = random.sample(list(icd9_source), random.randint(2, 2))
    proc_sample = random.sample(list(proc_source), random.randint(2, 3))
    atc3_sample = random.sample(list(atc3_source), random.randint(2, 3))

    return {
        "ICD9 Diagnosis": [code if keys else ICD9_CODES[code] for code in icd9_sample],
        "Procedures": [code if keys else PROCEDURE_CODES[code] for code in proc_sample],
        "Medications (ATC3)": [code if keys else ATC3_CODES[code] for code in atc3_sample]
    }
    
def generate_patient_record():
    num_admissions = random.randint(1, 3)
    print(num_admissions)
    admissions = [generate_admission() for _ in range(num_admissions)]
    
    patient_record = {
        "Patient Name": fake.name(),
        "Patient ID": fake.uuid4(),
        "Date of Birth": fake.date_of_birth(minimum_age=0, maximum_age=90).strftime("%Y-%m-%d"),
        "Sex": random.choice(["Male", "Female"]),
        "Blood Type": random.choice(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]),
        "Height_cm": random.randint(150, 200),
        "Weight_kg": random.randint(50, 120),
        "BMI": round(random.uniform(18.5, 35.0), 1),
        "Allergies": random.choice(["None", "Penicillin", "Aspirin", "Latex", "Peanuts"]),
        "Primary Physician": fake.name(),
        "Hospital": fake.company() + " Hospital",
        "Department": random.choice(["Cardiology", "Internal Medicine", "Surgery", "Neurology"]),
        "Ward": f"{random.randint(1, 10)}-{random.randint(1, 30)}",
        "Admission Date": fake.date_this_year().strftime("%Y-%m-%d"),
        "Discharge Date": fake.date_this_year().strftime("%Y-%m-%d"),
        "Visit Type": random.choice(["Inpatient", "Outpatient", "ER"]),
        "Diagnosis Summary": fake.sentence(nb_words=12),
        "Notes": fake.paragraph(nb_sentences=3),
        "Insurance Provider": fake.company(),
        "Policy Number": fake.bothify(text="????-########"),
        "Emergency Contact": fake.name(),
        "Emergency Phone": fake.phone_number(),
        "Discharge Status": random.choice(["Recovered", "Referred", "Deceased"]),
        "Follow-up Instructions": fake.sentence(nb_words=8),
        "Admissions": admissions
    }
    
    return patient_record

def create_pdf(patient_record, filename="medical_report.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    styleH = styles["Heading1"]

    # Title
    elements.append(Paragraph("Medical Report", styleH))
    elements.append(Spacer(1, 12))

    # Patient Info
    patient_info_data = [[k, str(v)] for k, v in patient_record.items() if k != "Admissions"]
    patient_table = Table(patient_info_data, colWidths=[150, 350])
    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ("VALIGN", (0,0), (-1,-1), "TOP")
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 12))

    # Admissions
    for i, adm in enumerate(patient_record["Admissions"], start=1):
        elements.append(Paragraph(f"Admission #{i}", styles["Heading2"]))
        adm_data = []
        for field, codes in adm.items():
            adm_data.append([field, ", ".join(codes)])
        adm_table = Table(adm_data, colWidths=[150, 350])
        adm_table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.black),
            ("VALIGN", (0,0), (-1,-1), "TOP")
        ]))
        elements.append(adm_table)
        elements.append(Spacer(1, 12))

    doc.build(elements)
    print(f"PDF generated: {filename}")

if __name__ == "__main__":
    record = generate_patient_record()
    create_pdf(record)
    print(json.dumps(record, indent=2))