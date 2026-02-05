"""
evaluate_validity.py — Compare Validity: Old vs New Approach

This script compares:
1. OLD approach: Generate coords → infer bonds → low validity
2. NEW approach: Generate topology → add coords → high validity

Metrics:
- Validity rate
- Uniqueness
- Diversity
- RMSD (if ground truth available)
"""

import os
import sys
import json
import argparse
from collections import Counter
from tqdm import tqdm

import torch
import numpy as np

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold
    import selfies as sf
except ImportError:
    print("Please install: pip install rdkit selfies")
    sys.exit(1)


def compute_validity(mols: list) -> float:
    """Compute validity rate."""
    valid = 0
    for mol in mols:
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                valid += 1
            except:
                pass
    return valid / len(mols) if mols else 0


def compute_uniqueness(smiles_list: list) -> float:
    """Compute uniqueness rate (unique / valid)."""
    unique = len(set(smiles_list))
    return unique / len(smiles_list) if smiles_list else 0


def compute_novelty(generated_smiles: list, training_smiles: set) -> float:
    """Compute novelty (not in training set)."""
    novel = sum(1 for s in generated_smiles if s not in training_smiles)
    return novel / len(generated_smiles) if generated_smiles else 0


def compute_diversity(mols: list) -> float:
    """Compute internal diversity using Tanimoto distance."""
    if len(mols) < 2:
        return 0
    
    # Compute fingerprints
    fps = []
    for mol in mols:
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)
        except:
            pass
    
    if len(fps) < 2:
        return 0
    
    # Compute pairwise Tanimoto distances
    distances = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            distances.append(1 - sim)  # Distance = 1 - similarity
    
    return np.mean(distances) if distances else 0


def compute_scaffold_diversity(mols: list) -> float:
    """Compute scaffold diversity."""
    scaffolds = set()
    for mol in mols:
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
            scaffolds.add(scaffold_smiles)
        except:
            pass
    
    return len(scaffolds) / len(mols) if mols else 0


def compute_property_distribution(mols: list) -> dict:
    """Compute molecular property distributions."""
    props = {
        'mw': [],      # Molecular weight
        'logp': [],    # LogP
        'hbd': [],     # H-bond donors
        'hba': [],     # H-bond acceptors
        'rotatable': [],  # Rotatable bonds
        'tpsa': [],    # Topological polar surface area
    }
    
    for mol in mols:
        try:
            props['mw'].append(Descriptors.MolWt(mol))
            props['logp'].append(Descriptors.MolLogP(mol))
            props['hbd'].append(Descriptors.NumHDonors(mol))
            props['hba'].append(Descriptors.NumHAcceptors(mol))
            props['rotatable'].append(Descriptors.NumRotatableBonds(mol))
            props['tpsa'].append(Descriptors.TPSA(mol))
        except:
            pass
    
    # Compute statistics
    stats = {}
    for key, values in props.items():
        if values:
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    return stats


def load_mols_from_sdf(path: str) -> list:
    """Load molecules from SDF file."""
    suppl = Chem.SDMolSupplier(path, removeHs=True)
    return [mol for mol in suppl if mol is not None]


def load_mols_from_json(path: str) -> list:
    """Load molecules from JSON/JSONL file."""
    mols = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            smiles = data.get('smiles', None)
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    mols.append(mol)
    return mols


def evaluate(generated_path: str, 
             reference_path: str = None,
             output_path: str = None) -> dict:
    """
    Comprehensive evaluation of generated molecules.
    """
    print(f"Loading generated molecules from {generated_path}...")
    
    # Load generated molecules
    if generated_path.endswith('.sdf'):
        mols = load_mols_from_sdf(generated_path)
    else:
        mols = load_mols_from_json(generated_path)
    
    print(f"Loaded {len(mols)} molecules")
    
    # Get SMILES
    smiles_list = []
    for mol in mols:
        try:
            smiles_list.append(Chem.MolToSmiles(mol))
        except:
            pass
    
    # Load reference (training) set if provided
    training_smiles = set()
    if reference_path and os.path.exists(reference_path):
        print(f"Loading reference data from {reference_path}...")
        with open(reference_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                if 'smiles' in data:
                    training_smiles.add(data['smiles'])
        print(f"Loaded {len(training_smiles)} reference molecules")
    
    # Compute metrics
    print("\nComputing metrics...")
    
    metrics = {
        'num_generated': len(mols),
        'validity': compute_validity(mols),
        'uniqueness': compute_uniqueness(smiles_list),
        'diversity': compute_diversity(mols),
        'scaffold_diversity': compute_scaffold_diversity(mols),
    }
    
    if training_smiles:
        metrics['novelty'] = compute_novelty(smiles_list, training_smiles)
    
    # Property distributions
    metrics['properties'] = compute_property_distribution(mols)
    
    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Total molecules:    {metrics['num_generated']}")
    print(f"  Validity:           {metrics['validity']*100:.1f}%")
    print(f"  Uniqueness:         {metrics['uniqueness']*100:.1f}%")
    print(f"  Diversity:          {metrics['diversity']:.3f}")
    print(f"  Scaffold Diversity: {metrics['scaffold_diversity']:.3f}")
    if 'novelty' in metrics:
        print(f"  Novelty:            {metrics['novelty']*100:.1f}%")
    
    if 'properties' in metrics and metrics['properties']:
        print("\nProperty Distributions:")
        for prop, stats in metrics['properties'].items():
            print(f"  {prop}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved metrics to {output_path}")
    
    return metrics


def compare_approaches(old_results_path: str, new_results_path: str):
    """
    Side-by-side comparison of old vs new approach.
    """
    print("=" * 60)
    print("COMPARISON: Old Approach vs NExT-Mol Approach")
    print("=" * 60)
    
    # Load results
    with open(old_results_path, 'r') as f:
        old_metrics = json.load(f)
    
    with open(new_results_path, 'r') as f:
        new_metrics = json.load(f)
    
    # Print comparison
    print(f"\n{'Metric':<25} {'Old Approach':>15} {'NExT-Mol':>15} {'Improvement':>15}")
    print("-" * 70)
    
    for key in ['validity', 'uniqueness', 'diversity', 'novelty']:
        old_val = old_metrics.get(key, 0)
        new_val = new_metrics.get(key, 0)
        
        if old_val > 0:
            improvement = (new_val - old_val) / old_val * 100
            impr_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
        else:
            impr_str = "N/A"
        
        if key == 'validity':
            print(f"{key:<25} {old_val*100:>14.1f}% {new_val*100:>14.1f}% {impr_str:>15}")
        else:
            print(f"{key:<25} {old_val:>15.3f} {new_val:>15.3f} {impr_str:>15}")
    
    print("\n" + "=" * 60)
    print("Key Insight: NExT-Mol achieves near-100% validity by")
    print("pre-validating topology before generating 3D coordinates!")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate generated molecules')
    parser.add_argument('--generated', type=str, required=True,
                        help='Path to generated molecules (SDF or JSONL)')
    parser.add_argument('--reference', type=str, default=None,
                        help='Path to reference/training data (for novelty)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save evaluation metrics')
    parser.add_argument('--compare', type=str, nargs=2, default=None,
                        metavar=('OLD', 'NEW'),
                        help='Compare two evaluation result files')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_approaches(args.compare[0], args.compare[1])
    else:
        evaluate(args.generated, args.reference, args.output)


if __name__ == '__main__':
    main()
