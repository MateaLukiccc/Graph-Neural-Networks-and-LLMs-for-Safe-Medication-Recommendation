import dill
import numpy as np
import os
import torch
import torch.nn.functional as F
from training.compared_models.safedrug_xavier_kaiman import SafeDrugModel
from training.utils import build_mpnn 

# The model name and path to the pre-trained weights
MODEL_NAME = "SafeDrug"
RESUME_PATH = "A0_TARGET_0.06_JA_0.4372_DDI_0.06656.model" 

# Data paths 
VOC_PATH = "data/output/voc_final.pkl"
DDI_ADJ_PATH = "data/output/ddi_A_final.pkl"
DDI_MASK_PATH = "data/output/ddi_mask_H.pkl"
MOLECULE_PATH = "data/output/atc3toSMILES.pkl"

# Model hyperparameters (must match training)
DIM = 32

# Set device
DEVICE = torch.device("cpu")

class GNNModel:
    def __init__(self, model_name=MODEL_NAME, resume_path=RESUME_PATH, dim=DIM, device=DEVICE):
        
        print("Initializing GNNModel...")
        self.device = device
        
        # 1. Load data artifacts
        self.ddi_adj = dill.load(open(DDI_ADJ_PATH, "rb"))
        self.ddi_mask_H = dill.load(open(DDI_MASK_PATH, "rb"))
        self.molecule = dill.load(open(MOLECULE_PATH, "rb"))
        
        self.voc = dill.load(open(VOC_PATH, "rb"))
        self.diag_voc, self.pro_voc, self.med_voc = self.voc["diag_voc"], self.voc["pro_voc"], self.voc["med_voc"]
        self.voc_size = (len(self.diag_voc.idx2word), len(self.pro_voc.idx2word), len(self.med_voc.idx2word))
        
        # 2. Build MPNN components
        MPNNSet, N_fingerprint, average_projection = build_mpnn(
            self.molecule, self.med_voc.idx2word, 2, self.device
        )
        
        # 3. Initialize model
        self.model = SafeDrugModel(
            self.voc_size,
            self.ddi_adj,
            self.ddi_mask_H,
            MPNNSet,
            N_fingerprint,
            average_projection,
            emb_dim=dim,
            device=self.device,
        )
        self.model.to(device=self.device)
        
        # 4. Load weights
        full_resume_path = os.path.join("saved", resume_path)
        if not os.path.exists(full_resume_path):
            raise FileNotFoundError(f"Model file not found at: {full_resume_path}")
            
        state_dict = torch.load(full_resume_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        print(f"GNNModel initialized and weights loaded from: {full_resume_path}")

    @staticmethod
    def parse_input_to_indices(visits, diag_voc, pro_voc, med_voc):
        patient_seq = []
        for adm in visits:
            diag_codes = adm.get("ICD9", []) or adm.get("ICD9_CODE", [])
            pro_codes = adm.get("PRO_CODE", []) or adm.get("PROCEDURE", [])
            med_codes = adm.get("ATC3", []) or adm.get("MED", [])
            
            if not isinstance(diag_codes, list): diag_codes = list(diag_codes)
            if not isinstance(pro_codes, list): pro_codes = list(pro_codes)
            if not isinstance(med_codes, list): med_codes = list(med_codes)

            diag_idx = [diag_voc.word2idx[c] for c in diag_codes if c in diag_voc.word2idx]
            pro_idx = [pro_voc.word2idx[c] for c in pro_codes if c in pro_voc.word2idx]
            med_idx = [med_voc.word2idx[c] for c in med_codes if c in med_voc.word2idx]
            
            patient_seq.append([diag_idx, pro_idx, med_idx])
                    
        return patient_seq

    @torch.no_grad() 
    def predict(self, patient_history_indices):
        if not patient_history_indices:
            return {"predicted_med_ids": [], "predicted_probs": [], "predicted_drug_names": []}
                
        try:
            target_output, _ = self.model(patient_history_indices)
        except Exception as e:
            print(f"Error during model forward pass: {e}")
            return {"predicted_med_ids": [], "predicted_probs": [], "predicted_drug_names": []}
        
        target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
        y_pred_prob = target_output 
        y_pred_tmp = target_output.copy()
        y_pred_tmp[y_pred_tmp >= 0.5] = 1
        y_pred_tmp[y_pred_tmp < 0.5] = 0
        
        y_pred_label = np.where(y_pred_tmp == 1)[0]
        predicted_drug_names = [self.med_voc.idx2word[idx] for idx in y_pred_label]
        
        return {
            "predicted_med_ids": sorted(y_pred_label.tolist()),
            "predicted_probs": y_pred_prob.tolist(), 
            "predicted_drug_names": predicted_drug_names 
        }

if __name__ == "__main__":
    sample_raw_admissions = [
        {"ICD9_CODE": ["V58.69", "427.31", "414.01", "530.81", "584.9", "250.00"], 
         "PROCEDURE": ["39.61", "36.06", "89.52"], 
         "ATC3": ["C09A", "R03A", "C07AB02"]},
        {"ICD9_CODE": ["414.01", "272.4"], 
         "PROCEDURE": ["88.72"], 
         "ATC3": ["C10A", "B01A"]},
    ]

    try:
        gnn_model = GNNModel()
        patient_indices = GNNModel.parse_input_to_indices(
            visits=sample_raw_admissions,
            diag_voc=gnn_model.diag_voc,
            pro_voc=gnn_model.pro_voc,
            med_voc=gnn_model.med_voc
        )
        print("\nSuccessfully converted raw codes to indices.")
        prediction_results = gnn_model.predict(patient_indices)

        print("\n--- Prediction Results (For Next Visit) ---")
        print(f"Predicted Med IDs Count: {len(prediction_results['predicted_med_ids'])}")
        print(f"Predicted Drug Names: {prediction_results['predicted_drug_names']}")
    
    except FileNotFoundError as e:
        print("\nError: Data file not found. Ensure all paths are correct.")
        print(f"Missing file: {e}")
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")