from collections import defaultdict
from itertools import combinations
import sys
import warnings
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import dill
import numpy as np
import torch
from rdkit import Chem
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# -----------------------
# Utilities
# -----------------------
def llprint(message: str) -> None:
    """Write message to stdout without newline and flush immediately."""
    sys.stdout.write(message)
    sys.stdout.flush()


def get_n_params(model: Any) -> int:
    """Return number of parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters())


def transform_split(
    X: Sequence, Y: Sequence, train_size: float = 2.0 / 3.0, random_state: int = 1203
) -> Tuple:
    """Split (X, Y) into train / eval / test using two-stage splitting.

    Returns: x_train, x_eval, x_test, y_train, y_eval, y_test
    """
    x_train, x_remain, y_train, y_remain = train_test_split(
        X, Y, train_size=train_size, random_state=random_state
    )
    x_eval, x_test, y_eval, y_test = train_test_split(
        x_remain, y_remain, test_size=0.5, random_state=random_state
    )
    return x_train, x_eval, x_test, y_train, y_eval, y_test


# -----------------------
# Sequence / prediction helpers
# -----------------------
def sequence_output_process(
    output_logits: np.ndarray, filter_token: Iterable[int]
) -> Tuple[List[int], List[int]]:
    """Given logits for each example, return:
    - chosen label per example: highest-scoring label not in filter_token
    - sorted_predict: chosen labels sorted by their predicted probability (descending)
    """
    # argsort descending on last axis
    ranked_indices = np.argsort(output_logits, axis=-1)[:, ::-1]
    chosen = []
    chosen_probs = []

    for i in range(ranked_indices.shape[0]):
        # find first label for this sample that is not in filter_token
        label_found = None
        for label in ranked_indices[i]:
            if label in filter_token:
                continue
            label_found = int(label)
            break
        # If none found, set to -1 (or could be None)
        if label_found is None:
            chosen.append(-1)
            chosen_probs.append(-np.inf)
        else:
            chosen.append(label_found)
            chosen_probs.append(float(output_logits[i, label_found]))

    # sort chosen labels by their probability (descending), ignore -1 if you want
    sorted_pairs = sorted(
        ((p, lbl) for p, lbl in zip(chosen_probs, chosen) if lbl != -1),
        key=lambda x: x[0],
        reverse=True,
    )
    sorted_predict = [lbl for _, lbl in sorted_pairs]

    return chosen, sorted_predict


# -----------------------
# Metrics
# -----------------------
def _jaccard_sets(pred_set: Iterable[int], true_set: Iterable[int]) -> float:
    """Jaccard index between two sets (0..1)."""
    s1, s2 = set(pred_set), set(true_set)
    union = s1 | s2
    if not union:
        return 0.0
    return len(s1 & s2) / len(union)


def _precision_recall_for_example(pred_list: Sequence[int], true_array: np.ndarray) -> Tuple[float, float]:
    """Return (precision, recall) for one example.
    - precision = |pred ∩ true| / |pred| if pred non-empty else 0
    - recall = |pred ∩ true| / |true| if true non-empty else 0
    """
    true_set = set(np.where(true_array == 1)[0])
    pred_set = set(pred_list)
    if not pred_list:
        prec = 0.0
    else:
        prec = len(pred_set & true_set) / len(pred_set)
    if not true_set:
        rec = 0.0
    else:
        rec = len(pred_set & true_set) / len(true_set)
    return prec, rec


def _average_precision_recall_f1(all_prec: Sequence[float], all_rec: Sequence[float]) -> float:
    """Macro-average F1 across examples computed from per-example precision & recall lists."""
    f1s = []
    for p, r in zip(all_prec, all_rec):
        if p + r == 0:
            f1s.append(0.0)
        else:
            f1s.append(2.0 * p * r / (p + r))
    return float(np.mean(f1s))


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute average roc_auc across rows, return 0 if computation fails."""
    try:
        scores = [roc_auc_score(y_true[i], y_prob[i], average="macro") for i in range(len(y_true))]
        return float(np.mean(scores))
    except Exception:
        return 0.0


def _safe_avg_precision_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute average precision-recall AUC across rows, return 0 on failure."""
    try:
        scores = [
            average_precision_score(y_true[i], y_prob[i], average="macro")
            for i in range(len(y_true))
        ]
        return float(np.mean(scores))
    except Exception:
        return 0.0


def precision_at_k_from_ranked(y_true: np.ndarray, ranked_label_indices: Sequence[Sequence[int]], k: int) -> float:
    """Precision@k using precomputed ranked label lists (list of label indices per example)."""
    if len(ranked_label_indices) == 0:
        return 0.0
    total = 0.0
    for i, topk in enumerate(ranked_label_indices):
        TP = 0
        for lbl in topk[:k]:
            if y_true[i, lbl] == 1:
                TP += 1
        total += TP / float(k)
    return total / len(ranked_label_indices)


def multi_label_metric(y_gt: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Compute common multi-label metrics.

    Returns: (jaccard, pr_auc, avg_precision, avg_recall, avg_f1)
    """
    jaccards = [
        _jaccard_sets(np.where(y_pred[i] == 1)[0], np.where(y_gt[i] == 1)[0])
        for i in range(y_gt.shape[0])
    ]
    jaccard_mean = float(np.mean(jaccards))

    per_prec = []
    per_rec = []
    for i in range(y_gt.shape[0]):
        p, r = _precision_recall_for_example(list(np.where(y_pred[i] == 1)[0]), y_gt[i])
        per_prec.append(p)
        per_rec.append(r)

    avg_precision = float(np.mean(per_prec)) if per_prec else 0.0
    avg_recall = float(np.mean(per_rec)) if per_rec else 0.0
    avg_f1 = _average_precision_recall_f1(per_prec, per_rec)
    pr_auc = _safe_avg_precision_auc(y_gt, y_prob)


    return jaccard_mean, pr_auc, avg_precision, avg_recall, avg_f1


def sequence_metric(y_gt: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, y_label_ranked: Sequence[Sequence[int]]):
    """Sequence-style metric wrapper. Keeps same return signature as original.

    y_label_ranked: for each example, list of predicted labels sorted by model
    """
    # jaccard (using ranked predicted labels per example)
    jaccards = [
        _jaccard_sets(y_label_ranked[i], np.where(y_gt[i] == 1)[0]) for i in range(y_gt.shape[0])
    ]
    jaccard_mean = float(np.mean(jaccards))

    # average precision/recall/f1 computed from y_label_ranked (like in multi_label_metric)
    per_prec = []
    per_rec = []
    for i in range(y_gt.shape[0]):
        p, r = _precision_recall_for_example(list(y_label_ranked[i]), y_gt[i])
        per_prec.append(p)
        per_rec.append(r)
    avg_precision = float(np.mean(per_prec)) if per_prec else 0.0
    avg_recall = float(np.mean(per_rec)) if per_rec else 0.0
    avg_f1 = _average_precision_recall_f1(per_prec, per_rec)
    pr_auc = _safe_avg_precision_auc(y_gt, y_prob)

    return jaccard_mean, pr_auc, avg_precision, avg_recall, avg_f1


# -----------------------
# DDI rate
# -----------------------
def ddi_rate_score(record: Sequence[Sequence[Sequence[int]]], path: str = "../data/output/ddi_A_final.pkl") -> float:
    """Compute drug-drug interaction (DDI) rate from predicted records.

    `record` is expected to be a sequence of patients, each a sequence of admissions,
    each admission being an iterable of medication indices.
    """
    ddi_A = dill.load(open(path, "rb"))

    all_pairs = 0
    ddi_pairs = 0

    for patient in record:
        for admission in patient:
            meds = list(admission)
            # iterate unique pairs
            for i, j in combinations(range(len(meds)), 2):
                all_pairs += 1
                a, b = meds[i], meds[j]
                if ddi_A[a, b] == 1 or ddi_A[b, a] == 1:
                    ddi_pairs += 1

    if all_pairs == 0:
        return 0.0
    return ddi_pairs / all_pairs


# -----------------------
# Molecular fingerprint / MPNN building
# -----------------------
def create_atoms(mol: Chem.Mol, atom_map: Dict[Any, int]) -> np.ndarray:
    """Map RDKit molecule atoms -> integer tokens (atom type + aromatic flag)."""
    # base symbols
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    # mark aromatic atoms as tuple (symbol, 'aromatic') so they map separately
    for atom in mol.GetAromaticAtoms():
        idx = atom.GetIdx()
        symbols[idx] = (symbols[idx], "aromatic")
    return np.array([atom_map.setdefault(sym, len(atom_map)) for sym in symbols], dtype=np.int64)


def create_ijbonddict(mol: Chem.Mol, bond_map: Dict[str, int]) -> Dict[int, List[Tuple[int, int]]]:
    """Create adjacency dict: node -> list of (neighbor_idx, bond_token)."""
    i_jbond = defaultdict(list)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_token = bond_map.setdefault(str(bond.GetBondType()), len(bond_map))
        i_jbond[i].append((j, bond_token))
        i_jbond[j].append((i, bond_token))
    return i_jbond


def extract_fingerprints(
    radius: int,
    atoms: np.ndarray,
    i_jbond_dict: Dict[int, List[Tuple[int, int]]],
    fingerprint_map: Dict[Any, int],
    edge_map: Dict[Any, int],
) -> np.ndarray:
    """Weisfeiler-Lehman style fingerprint extraction.

    Returns token ids per node (numpy array).
    """
    if len(atoms) == 1 or radius == 0:
        return np.array([fingerprint_map.setdefault(int(a), len(fingerprint_map)) for a in atoms], dtype=np.int64)

    # initialize node tokens to atom tokens
    nodes = [int(a) for a in atoms]
    # convert i_jbond_dict to a mutable structure for iterating
    i_jedge_dict = dict(i_jbond_dict)

    for _ in range(radius):
        # compute new node fingerprints
        nodes_new = []
        for i, j_edge in i_jedge_dict.items():
            neighbors = sorted(((nodes[j], edge) for j, edge in j_edge))
            fingerprint = (nodes[i], tuple(neighbors))
            nodes_new.append(fingerprint_map.setdefault(fingerprint, len(fingerprint_map)))
        # update edge IDs to incorporate both-side node tokens
        i_jedge_dict_new = defaultdict(list)
        for i, j_edge in i_jedge_dict.items():
            for j, edge in j_edge:
                both_side = tuple(sorted((nodes[i], nodes[j])))
                edge_token = edge_map.setdefault((both_side, edge), len(edge_map))
                i_jedge_dict_new[i].append((j, edge_token))
        nodes = nodes_new
        i_jedge_dict = i_jedge_dict_new

    return np.array(nodes, dtype=np.int64)


def build_mpnn(
    molecule_map: Dict[Any, Iterable[str]],
    med_voc: Dict[Any, Any],
    radius: int = 1,
    device: str = "cpu",
) -> Tuple[List[Tuple[torch.LongTensor, torch.FloatTensor, int]], int, torch.FloatTensor]:
    """Build MPNN dataset for a set of molecules. Takes molecules (SMILES) and drug vocab, 
    and returns everything needed (graph tensors + projection matrix) to train an MPNN on them.

    molecule_map: mapping from med_id -> iterable of SMILES strings
    med_voc: mapping used to iterate (keeps order and grouping)
    Returns: (list_of_data, n_fingerprints, average_projection_tensor)
    where each element in list_of_data is (fingerprint_tensor, adjacency_tensor, molecular_size)
    """
    # Use plain dicts + setdefault to assign incremental ids
    atom_map: Dict[Any, int] = {}
    bond_map: Dict[str, int] = {}
    fingerprint_map: Dict[Any, int] = {}
    edge_map: Dict[Any, int] = {}

    mpnn_set = []
    average_index: List[int] = []

    # med_voc expected to map index->atc3 code in original; we iterate to get grouped SMILES
    for _, atc3 in med_voc.items():
        smiles_list = list(molecule_map.get(atc3, []))
        counter = 0
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                mol = Chem.AddHs(mol)
                atoms = create_atoms(mol, atom_map)
                molecular_size = len(atoms)
                i_jbond_dict = create_ijbonddict(mol, bond_map)
                fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint_map, edge_map)
                adjacency = Chem.GetAdjacencyMatrix(mol)

                # If fingerprints shorter than adjacency (shouldn't happen often), pad with a token id
                if fingerprints.shape[0] < adjacency.shape[0]:
                    pad_len = adjacency.shape[0] - fingerprints.shape[0]
                    pad_token = fingerprint_map.setdefault("__PAD__", len(fingerprint_map))
                    fingerprints = np.concatenate([fingerprints, np.full((pad_len,), pad_token, dtype=np.int64)])

                fp_tensor = torch.LongTensor(fingerprints).to(device)
                adj_tensor = torch.FloatTensor(adjacency).to(device)
                mpnn_set.append((fp_tensor, adj_tensor, molecular_size))
                counter += 1
            except Exception:
                # skip problematic molecules
                continue

        average_index.append(counter)

    # Build average projection matrix: rows = number of med groups, cols = total molecules
    n_cols = sum(average_index)
    n_rows = len(average_index)
    if n_rows == 0 or n_cols == 0:
        average_projection = torch.zeros((n_rows, n_cols), dtype=torch.float32)
        return mpnn_set, len(fingerprint_map), average_projection

    proj = np.zeros((n_rows, n_cols), dtype=np.float32)
    col = 0
    for i, count in enumerate(average_index):
        if count > 0:
            proj[i, col : col + count] = 1.0 / count
        col += count

    return mpnn_set, len(fingerprint_map), torch.FloatTensor(proj)
