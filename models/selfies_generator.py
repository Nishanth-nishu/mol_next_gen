"""
selfies_generator.py — SELFIES-based Topology Generator

SELFIES (SELF-referencing Embedded Strings) guarantees 100% chemical validity.
Every valid SELFIES string corresponds to a valid molecular graph.

This module provides:
1. SELFIESGenerator: Generate valid SELFIES strings
2. Conversion utilities: SELFIES → RDKit Mol → Graph tensors
"""

import torch
import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Tuple, Optional, Dict
import random
import json
import os


# =============================================================================
# SELFIES UTILITIES
# =============================================================================

def smiles_to_selfies(smiles: str) -> Optional[str]:
    """Convert SMILES to SELFIES. Returns None if conversion fails."""
    try:
        return sf.encoder(smiles)
    except:
        return None


def selfies_to_smiles(selfies_str: str) -> Optional[str]:
    """Convert SELFIES to SMILES. Returns None if conversion fails."""
    try:
        return sf.decoder(selfies_str)
    except:
        return None


def selfies_to_mol(selfies_str: str) -> Optional[Chem.Mol]:
    """Convert SELFIES to RDKit Mol object."""
    smiles = selfies_to_smiles(selfies_str)
    if smiles is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Chem.SanitizeMol(mol)
        return mol
    except:
        return None


def mol_to_graph_tensors(mol: Chem.Mol) -> Dict[str, torch.Tensor]:
    """
    Convert RDKit Mol to graph tensors.
    
    Returns:
        dict with:
        - atom_types: (N,) atomic numbers
        - edge_index: (2, E) bond connections (undirected)
        - bond_types: (E,) bond orders (1=single, 2=double, 3=triple, 4=aromatic)
        - num_atoms: int
    """
    # Atom types
    atom_types = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], 
                               dtype=torch.long)
    
    # Bonds (undirected, so add both directions)
    edges_src = []
    edges_dst = []
    bond_orders = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Bond type
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
            order = 1  # Default
        
        # Add both directions
        edges_src.extend([i, j])
        edges_dst.extend([j, i])
        bond_orders.extend([order, order])
    
    if len(edges_src) > 0:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        bond_types = torch.tensor(bond_orders, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        bond_types = torch.zeros(0, dtype=torch.long)
    
    return {
        'atom_types': atom_types,
        'edge_index': edge_index,
        'bond_types': bond_types,
        'num_atoms': len(atom_types)
    }


# =============================================================================
# SELFIES VOCABULARY
# =============================================================================

class SELFIESVocabulary:
    """Vocabulary for SELFIES tokens."""
    
    def __init__(self, selfies_list: Optional[List[str]] = None):
        self.token_to_idx = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx_to_token = {0: '<pad>', 1: '<bos>', 2: '<eos>', 3: '<unk>'}
        self.next_idx = 4
        
        if selfies_list:
            self.build_vocab(selfies_list)
    
    def build_vocab(self, selfies_list: List[str]):
        """Build vocabulary from list of SELFIES strings."""
        for selfies_str in selfies_list:
            tokens = list(sf.split_selfies(selfies_str))
            for token in tokens:
                if token not in self.token_to_idx:
                    self.token_to_idx[token] = self.next_idx
                    self.idx_to_token[self.next_idx] = token
                    self.next_idx += 1
    
    def encode(self, selfies_str: str, max_len: int = 100) -> torch.Tensor:
        """Encode SELFIES string to tensor."""
        tokens = list(sf.split_selfies(selfies_str))
        indices = [self.token_to_idx.get('<bos>')]
        for token in tokens[:max_len-2]:
            indices.append(self.token_to_idx.get(token, self.token_to_idx['<unk>']))
        indices.append(self.token_to_idx.get('<eos>'))
        
        # Pad
        while len(indices) < max_len:
            indices.append(self.token_to_idx['<pad>'])
        
        return torch.tensor(indices, dtype=torch.long)
    
    def decode(self, indices: torch.Tensor) -> str:
        """Decode tensor to SELFIES string."""
        tokens = []
        for idx in indices.tolist():
            token = self.idx_to_token.get(idx, '<unk>')
            if token == '<eos>':
                break
            if token not in ['<pad>', '<bos>', '<unk>']:
                tokens.append(token)
        return ''.join(tokens)
    
    def __len__(self):
        return len(self.token_to_idx)
    
    def save(self, path: str):
        """Save vocabulary to file."""
        with open(path, 'w') as f:
            json.dump({'token_to_idx': self.token_to_idx}, f)
    
    def load(self, path: str):
        """Load vocabulary from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.token_to_idx = data['token_to_idx']
        self.idx_to_token = {int(v): k for k, v in self.token_to_idx.items()}
        self.next_idx = max(self.token_to_idx.values()) + 1


# =============================================================================
# SELFIES GENERATOR
# =============================================================================

class SELFIESGenerator:
    """
    Generate valid SELFIES strings.
    
    Options:
    1. Sample from training data (simple, fast)
    2. Use pretrained language model (better diversity)
    3. Train custom language model
    """
    
    def __init__(self, 
                 data_path: Optional[str] = None,
                 mode: str = 'sample',  # 'sample', 'pretrained', 'custom'
                 vocab: Optional[SELFIESVocabulary] = None):
        """
        Args:
            data_path: Path to JSONL file with SELFIES strings
            mode: Generation mode
            vocab: SELFIES vocabulary (required for 'custom' mode)
        """
        self.mode = mode
        self.vocab = vocab
        self.selfies_pool = []
        
        if data_path and os.path.exists(data_path):
            self._load_data(data_path)
    
    def _load_data(self, data_path: str):
        """Load SELFIES strings from file."""
        with open(data_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                if 'selfies' in data:
                    self.selfies_pool.append(data['selfies'])
                elif 'smiles' in data:
                    selfies = smiles_to_selfies(data['smiles'])
                    if selfies:
                        self.selfies_pool.append(selfies)
        print(f"Loaded {len(self.selfies_pool)} SELFIES strings")
    
    def generate_one(self) -> Optional[str]:
        """Generate a single SELFIES string."""
        if self.mode == 'sample':
            return self._sample_from_pool()
        elif self.mode == 'pretrained':
            return self._generate_pretrained()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def generate(self, num_molecules: int = 100) -> List[str]:
        """Generate multiple SELFIES strings."""
        results = []
        for _ in range(num_molecules):
            selfies = self.generate_one()
            if selfies:
                results.append(selfies)
        return results
    
    def _sample_from_pool(self) -> Optional[str]:
        """Sample from training data pool."""
        if not self.selfies_pool:
            return None
        return random.choice(self.selfies_pool)
    
    def _generate_pretrained(self) -> Optional[str]:
        """Generate using pretrained model (placeholder)."""
        # TODO: Integrate with HuggingFace ChemGPT or similar
        raise NotImplementedError("Pretrained generation not yet implemented")
    
    def selfies_to_graph(self, selfies_str: str) -> Optional[Dict[str, torch.Tensor]]:
        """Convert SELFIES to graph tensors."""
        mol = selfies_to_mol(selfies_str)
        if mol is None:
            return None
        return mol_to_graph_tensors(mol)
    
    def generate_with_graph(self, num_molecules: int = 100) -> List[Dict]:
        """
        Generate SELFIES and convert to graphs.
        
        Returns list of dicts with 'selfies', 'smiles', and graph tensors.
        """
        results = []
        attempts = 0
        max_attempts = num_molecules * 2
        
        while len(results) < num_molecules and attempts < max_attempts:
            attempts += 1
            selfies = self.generate_one()
            if selfies is None:
                continue
            
            graph = self.selfies_to_graph(selfies)
            if graph is None:
                continue
            
            smiles = selfies_to_smiles(selfies)
            
            results.append({
                'selfies': selfies,
                'smiles': smiles,
                **graph
            })
        
        return results


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    # Test SELFIES conversion
    test_smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C']
    
    print("Testing SELFIES conversion:")
    for smi in test_smiles:
        selfies = smiles_to_selfies(smi)
        back_smi = selfies_to_smiles(selfies)
        mol = selfies_to_mol(selfies)
        valid = mol is not None
        print(f"  {smi} -> {selfies} -> {back_smi} (valid: {valid})")
    
    # Test graph conversion
    print("\nTesting graph conversion:")
    mol = Chem.MolFromSmiles('CCO')
    graph = mol_to_graph_tensors(mol)
    print(f"  Ethanol: atoms={graph['atom_types'].tolist()}, "
          f"edges={graph['edge_index'].shape}, bonds={graph['bond_types'].tolist()}")
