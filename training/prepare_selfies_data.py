"""
prepare_selfies_data.py — Convert QM9 to SELFIES format

This script:
1. Loads QM9 molecules (SMILES format)
2. Converts to SELFIES (100% valid representation)
3. Extracts graph information and coordinates
4. Saves as training data for conformer diffusion
"""

import os
import sys
import json
import argparse
from tqdm import tqdm
import torch
import numpy as np

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import selfies as sf
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("Please install: pip install selfies rdkit")
    sys.exit(1)


def smiles_to_selfies(smiles: str) -> str:
    """Convert SMILES to SELFIES."""
    try:
        return sf.encoder(smiles)
    except:
        return None


def mol_to_data(mol: Chem.Mol, selfies: str) -> dict:
    """
    Extract all data from RDKit mol.
    
    Returns dict with:
    - selfies: SELFIES string
    - smiles: SMILES string
    - atom_types: list of atomic numbers
    - edge_index: adjacency list
    - bond_types: bond orders
    - coordinates: 3D positions (if available)
    """
    if mol is None:
        return None
    
    smiles = Chem.MolToSmiles(mol)
    
    # Atom types
    atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    
    # Bonds
    edges_src = []
    edges_dst = []
    bond_orders = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        bt = bond.GetBondType()
        if bt == Chem.BondType.SINGLE:
            order = 1
        elif bt == Chem.BondType.DOUBLE:
            order = 2
        elif bt == Chem.BondType.TRIPLE:
            order = 3
        elif bt == Chem.BondType.AROMATIC:
            order = 4
        else:
            order = 1
        
        # Bidirectional
        edges_src.extend([i, j])
        edges_dst.extend([j, i])
        bond_orders.extend([order, order])
    
    # Coordinates
    coords = None
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        coords = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])
    
    return {
        'selfies': selfies,
        'smiles': smiles,
        'atom_types': atom_types,
        'edge_index': [edges_src, edges_dst],
        'bond_types': bond_orders,
        'coordinates': coords,
        'num_atoms': len(atom_types)
    }


def process_qm9_json(input_path: str, output_path: str, max_molecules: int = None):
    """
    Process QM9 JSONL file and convert to SELFIES format.
    """
    print(f"Processing {input_path}...")
    
    success = 0
    failed = 0
    
    with open(output_path, 'w') as fout:
        with open(input_path, 'r') as fin:
            for i, line in enumerate(tqdm(fin)):
                if max_molecules and i >= max_molecules:
                    break
                
                try:
                    data = json.loads(line.strip())
                    smiles = data.get('smiles', None)
                    
                    if smiles is None:
                        failed += 1
                        continue
                    
                    # Convert to SELFIES
                    selfies = smiles_to_selfies(smiles)
                    if selfies is None:
                        failed += 1
                        continue
                    
                    # Get mol
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        failed += 1
                        continue
                    
                    # Add 3D coords if not present
                    if mol.GetNumConformers() == 0:
                        # Use coordinates from data if available
                        if 'coordinates' in data and data['coordinates']:
                            conf = Chem.Conformer(mol.GetNumAtoms())
                            for j, coord in enumerate(data['coordinates']):
                                conf.SetAtomPosition(j, coord)
                            mol.AddConformer(conf)
                        else:
                            # Generate with ETKDG
                            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                    
                    # Extract data
                    mol_data = mol_to_data(mol, selfies)
                    if mol_data is None:
                        failed += 1
                        continue
                    
                    # Add original coordinates if available
                    if 'coordinates' in data and data['coordinates']:
                        mol_data['coordinates'] = data['coordinates']
                    
                    # Write
                    fout.write(json.dumps(mol_data) + '\n')
                    success += 1
                    
                except Exception as e:
                    failed += 1
                    if failed < 10:
                        print(f"Warning: {e}")
    
    print(f"Done! Success: {success}, Failed: {failed}")
    print(f"Saved to: {output_path}")


def process_sdf(input_path: str, output_path: str, max_molecules: int = None):
    """
    Process SDF file.
    """
    print(f"Processing SDF: {input_path}...")
    
    suppl = Chem.SDMolSupplier(input_path, removeHs=False)
    
    success = 0
    failed = 0
    
    with open(output_path, 'w') as fout:
        for i, mol in enumerate(tqdm(suppl)):
            if max_molecules and i >= max_molecules:
                break
            
            if mol is None:
                failed += 1
                continue
            
            try:
                smiles = Chem.MolToSmiles(mol)
                selfies = smiles_to_selfies(smiles)
                
                if selfies is None:
                    failed += 1
                    continue
                
                mol_data = mol_to_data(mol, selfies)
                if mol_data is None:
                    failed += 1
                    continue
                
                fout.write(json.dumps(mol_data) + '\n')
                success += 1
                
            except Exception as e:
                failed += 1
    
    print(f"Done! Success: {success}, Failed: {failed}")


def create_vocabulary(data_path: str, vocab_path: str):
    """
    Build SELFIES vocabulary from data.
    """
    print("Building SELFIES vocabulary...")
    
    all_tokens = set(['<pad>', '<bos>', '<eos>', '<unk>'])
    
    with open(data_path, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line.strip())
            selfies_str = data.get('selfies', '')
            
            tokens = list(sf.split_selfies(selfies_str))
            all_tokens.update(tokens)
    
    # Create vocabulary
    token_to_idx = {tok: i for i, tok in enumerate(sorted(all_tokens))}
    
    with open(vocab_path, 'w') as f:
        json.dump({'token_to_idx': token_to_idx}, f, indent=2)
    
    print(f"Vocabulary size: {len(token_to_idx)}")
    print(f"Saved to: {vocab_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare SELFIES data from QM9')
    parser.add_argument('--input', type=str, required=True,
                        help='Input file (JSONL or SDF)')
    parser.add_argument('--output', type=str, default='data/qm9_selfies.jsonl',
                        help='Output JSONL file')
    parser.add_argument('--max_molecules', type=int, default=None,
                        help='Maximum molecules to process')
    parser.add_argument('--build_vocab', action='store_true',
                        help='Also build SELFIES vocabulary')
    parser.add_argument('--vocab_path', type=str, default='data/selfies_vocab.json',
                        help='Path to save vocabulary')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # Process based on file type
    if args.input.endswith('.sdf'):
        process_sdf(args.input, args.output, args.max_molecules)
    else:
        process_qm9_json(args.input, args.output, args.max_molecules)
    
    # Build vocabulary
    if args.build_vocab:
        os.makedirs(os.path.dirname(args.vocab_path) if os.path.dirname(args.vocab_path) else '.', exist_ok=True)
        create_vocabulary(args.output, args.vocab_path)


if __name__ == '__main__':
    main()
