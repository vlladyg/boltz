from dataclasses import replace
from typing import Optional

import numpy as np

from boltz.data import const
from boltz.data.crop.cropper import Cropper
from boltz.data.types import Tokenized


class ProteinProteinAffinityCropper(Cropper):
    """Protein-protein affinity cropper focusing on interface residues from both proteins."""

    def __init__(
        self,
        neighborhood_size: int = 10,
        max_tokens_protein: int = 200,
        interface_cutoff: float = 8.0,
        min_interface_residues: int = 5,
        balance_proteins: bool = True,
    ) -> None:
        """Initialize the protein-protein affinity cropper.

        Parameters
        ----------
        neighborhood_size : int
            Size of sequence neighborhood around interface residues.
        max_tokens_protein : int
            Maximum tokens per protein chain.
        interface_cutoff : float
            Distance cutoff (Angstroms) for defining interface residues.
        min_interface_residues : int
            Minimum number of interface residues to include from each protein.
        balance_proteins : bool
            Whether to balance representation between receptor and binder proteins.
        """
        self.neighborhood_size = neighborhood_size
        self.max_tokens_protein = max_tokens_protein
        self.interface_cutoff = interface_cutoff
        self.min_interface_residues = min_interface_residues
        self.balance_proteins = balance_proteins

    def _detect_interface_residues(
        self, 
        receptor_tokens: np.ndarray, 
        binder_tokens: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Detect interface residues between receptor and binder proteins.
        
        Parameters
        ----------
        receptor_tokens : np.ndarray
            Tokens from the receptor protein.
        binder_tokens : np.ndarray
            Tokens from the binder protein.
            
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Interface residue indices for receptor and binder respectively.
        """
        # Compute pairwise distances between receptor and binder centers
        receptor_coords = receptor_tokens["center_coords"]
        binder_coords = binder_tokens["center_coords"]
        
        # Distance matrix: receptor x binder
        dist_matrix = np.sqrt(
            np.sum(
                (receptor_coords[:, None, :] - binder_coords[None, :, :]) ** 2,
                axis=-1
            )
        )
        
        # Find interface residues (any residue within cutoff of the other protein)
        receptor_interface_mask = np.any(dist_matrix <= self.interface_cutoff, axis=1)
        binder_interface_mask = np.any(dist_matrix <= self.interface_cutoff, axis=0)
        
        receptor_interface_indices = np.where(receptor_interface_mask)[0]
        binder_interface_indices = np.where(binder_interface_mask)[0]

        return receptor_interface_indices, binder_interface_indices

    def crop(
        self,
        data: Tokenized,
        max_tokens: int,
        max_atoms: Optional[int] = None,
    ) -> Tokenized:
        """Crop data focusing on protein-protein interface.

        Parameters
        ----------
        data : Tokenized
            The tokenized data.
        max_tokens : int
            Maximum number of tokens to crop.
        max_atoms : Optional[int]
            Maximum number of atoms to consider.

        Returns
        -------
        Tokenized
            The cropped data focusing on protein-protein interface.
        """
        # Get token data
        token_data = data.tokens
        token_bonds = data.bonds

        # Filter to resolved tokens
        valid_tokens = token_data[token_data["resolved_mask"]]

        if not valid_tokens.size:
            msg = "No valid tokens in structure"
            raise ValueError(msg)

        # Separate receptor and binder proteins
        binder_tokens = valid_tokens[valid_tokens["affinity_mask"]]
        receptor_tokens = valid_tokens[
            (~valid_tokens["affinity_mask"]) & 
            (valid_tokens["mol_type"] == const.chain_type_ids["PROTEIN"])
        ]

        if not binder_tokens.size or not receptor_tokens.size:
            msg = "Need both receptor and binder proteins for protein-protein cropping"
            raise ValueError(msg)

        # Detect interface residues
        receptor_interface_idx, binder_interface_idx = self._detect_interface_residues(
            receptor_tokens, binder_tokens
        )

        # Ensure minimum interface residues
        if len(receptor_interface_idx) < self.min_interface_residues:
            # Expand to include closest residues
            receptor_coords = receptor_tokens["center_coords"]
            binder_coords = binder_tokens["center_coords"]
            
            # Find closest receptor residues to any binder residue
            min_dists = np.min(
                np.sqrt(
                    np.sum(
                        (receptor_coords[:, None, :] - binder_coords[None, :, :]) ** 2,
                        axis=-1
                    )
                ), axis=1
            )
            receptor_interface_idx = np.argsort(min_dists)[:self.min_interface_residues]

        if len(binder_interface_idx) < self.min_interface_residues:
            # Similarly for binder
            receptor_coords = receptor_tokens["center_coords"]
            binder_coords = binder_tokens["center_coords"]
            
            min_dists = np.min(
                np.sqrt(
                    np.sum(
                        (binder_coords[:, None, :] - receptor_coords[None, :, :]) ** 2,
                        axis=-1
                    )
                ), axis=1
            )
            binder_interface_idx = np.argsort(min_dists)[:self.min_interface_residues]

        # Collect cropped tokens starting with interface residues
        cropped: set[int] = set()
        total_atoms = 0

        # Process receptor interface residues
        receptor_interface_tokens = receptor_tokens[receptor_interface_idx]
        for token in receptor_interface_tokens:
            if len(cropped) >= max_tokens:
                break
                
            # Get neighborhood around this interface residue
            chain_tokens = token_data[token_data["asym_id"] == token["asym_id"]]
            new_tokens = self._get_token_neighborhood(token, chain_tokens)
            
            # Check limits
            new_indices = set(new_tokens["token_idx"]) - cropped
            new_atoms = np.sum(new_tokens[np.isin(new_tokens["token_idx"], list(new_indices))]["atom_num"])
            
            if ((len(new_indices) > (max_tokens - len(cropped))) or 
                ((max_atoms is not None) and ((total_atoms + new_atoms) > max_atoms))):
                break
                
            cropped.update(new_indices)
            total_atoms += new_atoms

        # Process binder interface residues
        binder_interface_tokens = binder_tokens[binder_interface_idx]
        for token in binder_interface_tokens:
            if len(cropped) >= max_tokens:
                break
                
            # Get neighborhood around this interface residue
            chain_tokens = token_data[token_data["asym_id"] == token["asym_id"]]
            new_tokens = self._get_token_neighborhood(token, chain_tokens)
            
            # Check limits
            new_indices = set(new_tokens["token_idx"]) - cropped
            new_atoms = np.sum(new_tokens[np.isin(new_tokens["token_idx"], list(new_indices))]["atom_num"])
            
            if ((len(new_indices) > (max_tokens - len(cropped))) or 
                ((max_atoms is not None) and ((total_atoms + new_atoms) > max_atoms))):
                break
                
            cropped.update(new_indices)
            total_atoms += new_atoms

        # Get cropped tokens sorted by index
        token_data = token_data[sorted(cropped)]

        # Filter bonds to only those within cropped tokens
        indices = token_data["token_idx"]
        token_bonds = token_bonds[np.isin(token_bonds["token_1"], indices)]
        token_bonds = token_bonds[np.isin(token_bonds["token_2"], indices)]
        
        return replace(data, tokens=token_data, bonds=token_bonds)

    def _get_token_neighborhood(self, query_token: np.ndarray, chain_tokens: np.ndarray) -> np.ndarray:
        """Get sequence neighborhood around a query token."""
        if len(chain_tokens) <= self.neighborhood_size:
            return chain_tokens
        
        # Build residue-index window around query token
        min_idx = query_token["res_idx"] - self.neighborhood_size
        max_idx = query_token["res_idx"] + self.neighborhood_size
        
        # Start with query token
        tokens = chain_tokens[chain_tokens["res_idx"] == query_token["res_idx"]]
        
        # Expand neighborhood until we have enough tokens
        current_min = current_max = query_token["res_idx"]
        while tokens.size < self.neighborhood_size and (current_min > min_idx or current_max < max_idx):
            if current_min > min_idx:
                current_min -= 1
            if current_max < max_idx:
                current_max += 1
            
            tokens = chain_tokens[
                (chain_tokens["res_idx"] >= current_min) & 
                (chain_tokens["res_idx"] <= current_max)
            ]
        
        return tokens 
