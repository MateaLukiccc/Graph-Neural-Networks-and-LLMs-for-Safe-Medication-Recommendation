from inference.parser.parser import parse_pdf
from inference.gnn_model import GNNModel
import pandas as pd
import numpy as np
from typing import Dict, Any

def format_prediction_report(predictions: Dict[str, Any], confidence_threshold: float = 0.5) -> str:
    try:
        ids = predictions['predicted_med_ids']
        names = predictions['predicted_drug_names']
        probs = np.array(predictions['predicted_probs'])[ids] * 100
    except Exception as e:
        return f"Error: Failed to pair selected IDs with probabilities. {e}"

    df = pd.DataFrame({
        "Rank": range(1, len(ids) + 1),
        "ATC3 Code": names,
        "Prediction Probability": probs
    }).sort_values("Prediction Probability", ascending=False).reset_index(drop=True)

    bins = [0, confidence_threshold * 100, 70, 90, 100]
    labels = [
        "Not Recommended (Filtered)", 
        "Low Confidence (Borderline)", 
        "**Medium Confidence**", 
        "**High Confidence**"
    ]
    df["Recommendation Status"] = pd.cut(df["Prediction Probability"], bins=bins, labels=labels, right=False)
    df["Prediction Probability"] = df["Prediction Probability"].map("{:.2f}%".format)

    report = [
        "## Predicted Medication Recommendation Report",
        f"\nRecommendation Threshold: $\\ge {confidence_threshold*100:.0f}\\%$",
        f"Total Medications Recommended: **{len(df)}**\n",
        df.to_markdown(index=False),
        "\n\n***\n\n### Summary of Confidence"
    ]
    
    high = (probs >= 90).sum()
    med = ((probs >= 70) & (probs < 90)).sum()
    low = ((probs >= confidence_threshold * 100) & (probs < 70)).sum()

    report.extend([
        f"- **High Confidence:** {high} drug classes",
        f"- **Medium Confidence:** {med} drug classes",
        f"- **Low/Borderline Confidence:** {low} drug classes"
    ])

    return "\n".join(report)

if __name__ == "__main__":
    pdf_path = "example_data/example_codes.pdf"
    admissions = parse_pdf(pdf_path)
    for idx, adm in enumerate(admissions, 1):
        print(f"Admission #{idx}:")
        print("  ICD9_CODE:", adm["ICD9_CODE"])
        print("  PROCEDURE:", adm["PROCEDURE"])
        print("  ATC3:", adm["ATC3"])
        print()
    print(type(admissions))
    print(admissions)
    gnn_model = GNNModel(model_name="safe_drug_model", dim=32, device="cpu")
    data = GNNModel.parse_input_to_indices(admissions, gnn_model.voc['diag_voc'], gnn_model.voc['pro_voc'], gnn_model.voc['med_voc'])
    predictions = gnn_model.predict(data)
    print("Predictions:", format_prediction_report(predictions))



