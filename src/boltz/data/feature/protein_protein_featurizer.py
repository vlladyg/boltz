import math
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import torch
from torch import Tensor
from rdkit.Chem import Mol

from boltz.data import const
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.types import Tokenized


class ProteinProteinFeaturizer(Boltz2Featurizer):
    """
    Featurizer for protein-protein complexes that extends Boltz2Featurizer.
    
    This class handles protein-protein binding problems for affinity prediction
    by creating appropriate masks and features while leveraging the full 
    functionality of the parent Boltz2Featurizer.
    """

    def __init__(self):
        """Initialize the protein-protein featurizer."""
        super().__init__()

    def _identify_protein_chains(
        self, 
        data: Tokenized,
    ) -> Tuple[List[int], List[int]]:
        """
        Identify receptor and binder chains using schema-defined affinity_mask.
        
        Parameters
        ----------
        data : Tokenized
            The input data with affinity_mask already set during tokenization.
            
        Returns
        -------
        Tuple[List[int], List[int]]
            Lists of receptor and binder chain IDs.
        """
        receptor_chains = []
        binder_chains = []
        
        # Use the affinity_mask from schema to identify chains
        for token in data.tokens:
            if token["mol_type"] == const.chain_type_ids["PROTEIN"]:
                chain_id = token["asym_id"]
                
                if token["affinity_mask"]:
                    # This is the binder protein (defined in schema)
                    if chain_id not in binder_chains:
                        binder_chains.append(chain_id)
                else:
                    # This is the receptor protein
                    if chain_id not in receptor_chains:
                        receptor_chains.append(chain_id)
        
        return receptor_chains, binder_chains

    def _create_protein_protein_masks(
        self, 
        data: Tokenized,
        receptor_chain_ids: List[int],
        binder_chain_ids: List[int],
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """
        Create receptor, binder, and interface masks for protein-protein complexes.
        
        Parameters
        ----------
        data : Tokenized
            The input data with affinity_mask already set.
        receptor_chain_ids : List[int]
            Chain IDs for the receptor protein(s).
        binder_chain_ids : List[int]
            Chain IDs for the binder protein(s).
        max_tokens : int, optional
            Maximum number of tokens for padding.
            
        Returns
        -------
        Dict[str, Tensor]
            Dictionary containing receptor_mask, binder_mask, and interface_mask.
        """
        num_tokens = len(data.tokens)
        
        # Initialize masks
        receptor_mask = torch.zeros(num_tokens, dtype=torch.bool)
        binder_mask = torch.zeros(num_tokens, dtype=torch.bool)
        
        # Fill masks based on chain assignments
        for i, token in enumerate(data.tokens):
            if token["asym_id"] in receptor_chain_ids:
                receptor_mask[i] = True
            elif token["asym_id"] in binder_chain_ids:
                binder_mask[i] = True
        
        # Create interface mask (receptor-binder interactions)
        interface_mask = torch.zeros(num_tokens, num_tokens, dtype=torch.bool)
        
        # Mark receptor-binder pairs in interface
        for i in range(num_tokens):
            for j in range(num_tokens):
                if (receptor_mask[i] and binder_mask[j]) or (binder_mask[i] and receptor_mask[j]):
                    interface_mask[i, j] = True
        
        # Add binder-binder interactions for binding site analysis
        for i in range(num_tokens):
            for j in range(num_tokens):
                if binder_mask[i] and binder_mask[j]:
                    interface_mask[i, j] = True
        
        # Pad if needed
        if max_tokens is not None and num_tokens < max_tokens:
            pad_len = max_tokens - num_tokens
            receptor_mask = torch.cat([receptor_mask, torch.zeros(pad_len, dtype=torch.bool)])
            binder_mask = torch.cat([binder_mask, torch.zeros(pad_len, dtype=torch.bool)])
            
            # Pad interface mask
            interface_mask = torch.cat([
                torch.cat([interface_mask, torch.zeros(num_tokens, pad_len, dtype=torch.bool)], dim=1),
                torch.zeros(pad_len, max_tokens, dtype=torch.bool)
            ], dim=0)
        
        return {
            "receptor_mask": receptor_mask,
            "binder_mask": binder_mask,
            "interface_mask": interface_mask,
        }

    def _compute_protein_protein_features(
        self,
        data: Tokenized,
        receptor_chain_ids: List[int],
        binder_chain_ids: List[int],
    ) -> Dict[str, Tensor]:
        """
        Compute protein-protein specific features.
        
        Parameters
        ----------
        data : Tokenized
            The input data to the model.
        receptor_chain_ids : List[int]
            Chain IDs for the receptor protein(s).
        binder_chain_ids : List[int]
            Chain IDs for the binder protein(s).
            
        Returns
        -------
        Dict[str, Tensor]
            Protein-protein specific features.
        """
        features = {}
        
        # Chain type information
        num_tokens = len(data.tokens)
        chain_type = torch.zeros(num_tokens, dtype=torch.long)
        
        for i, token in enumerate(data.tokens):
            if token["asym_id"] in receptor_chain_ids:
                chain_type[i] = 0  # Receptor
            elif token["asym_id"] in binder_chain_ids:
                chain_type[i] = 1  # Binder
            else:
                chain_type[i] = 2  # Other
        
        features["protein_chain_type"] = chain_type
        
        # Compute approximate molecular weights for affinity scaling
        receptor_tokens = sum(1 for token in data.tokens if token["asym_id"] in receptor_chain_ids)
        binder_tokens = sum(1 for token in data.tokens if token["asym_id"] in binder_chain_ids)
        
        # Approximate MW calculation (assuming ~110 Da per residue with removed hydrogens                           )
        receptor_mw = receptor_tokens * 100.0
        binder_mw = binder_tokens * 100.0
        
        features["receptor_mw"] = torch.tensor(receptor_mw, dtype=torch.float)
        features["binder_mw"] = torch.tensor(binder_mw, dtype=torch.float)
        
        features["affinity_mw"] = torch.tensor(100.0, dtype=torch.float)

        # Complex size features for affinity normalization
        total_interface_size = receptor_tokens + binder_tokens
        features["interface_size"] = torch.tensor(total_interface_size, dtype=torch.float)
        
        return features

    def process(
        self,
        data: Tokenized,
        random: np.random.Generator,
        molecules: dict[str, Mol],
        training: bool,
        max_seqs: int,

        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        num_ensembles: int = 1,
        ensemble_sample_replacement: bool = False,
        disto_use_ensemble: Optional[bool] = False,
        fix_single_ensemble: Optional[bool] = True,
        max_tokens: Optional[int] = None,
        max_atoms: Optional[int] = None,
        pad_to_max_seqs: bool = False,
        compute_symmetries: bool = False,
        binder_pocket_conditioned_prop: Optional[float] = 0.0,
        contact_conditioned_prop: Optional[float] = 0.0,
        binder_pocket_cutoff_min: Optional[float] = 4.0,
        binder_pocket_cutoff_max: Optional[float] = 20.0,
        binder_pocket_sampling_geometric_p: Optional[float] = 0.0,
        only_ligand_binder_pocket: Optional[bool] = False,
        only_pp_contact: Optional[bool] = True,  # Default True for protein-protein
        single_sequence_prop: Optional[float] = 0.0,
        msa_sampling: bool = False,
        override_bfactor: float = False,
        override_method: Optional[str] = None,
        compute_frames: bool = False,
        override_coords: Optional[Tensor] = None,
        bfactor_md_correction: bool = False,
        compute_constraint_features: bool = False,
        inference_pocket_constraints: Optional[
            list[tuple[int, list[tuple[int, int]], float]]
        ] = None,
        inference_contact_constraints: Optional[
            list[tuple[tuple[int, int], tuple[int, int], float]]
        ] = None,
        compute_affinity: bool = True,  # Default True for affinity prediction
        **kwargs,
    ) -> dict[str, Tensor]:
        """
        Process protein-protein complex data into features for affinity prediction.
        
        This method extends the parent process method to handle protein-protein
        specific features while maintaining all the functionality of the base class.
        
        Parameters
        ----------
        data : Tokenized
            The input data to the model.
        random : np.random.Generator
            Random number generator.
        molecules : dict[str, Mol]
            Dictionary of molecules.
        training : bool
            Whether the model is in training mode.
        max_seqs : int
            Maximum number of MSA sequences.

        **kwargs
            Additional arguments passed to parent process method.
            
        Returns
        -------
        dict[str, Tensor]
            Features for protein-protein affinity prediction.
        """
        # Identify receptor and binder chains using schema-defined affinity_mask
        receptor_chains, binder_chains = self._identify_protein_chains(data)
        
        # Call parent process method with protein-protein specific settings
        features = super().process(
            data=data,
            random=random,
            molecules=molecules,
            training=training,
            max_seqs=max_seqs,
            atoms_per_window_queries=atoms_per_window_queries,
            min_dist=min_dist,
            max_dist=max_dist,
            num_bins=num_bins,
            num_ensembles=num_ensembles,
            ensemble_sample_replacement=ensemble_sample_replacement,
            disto_use_ensemble=disto_use_ensemble,
            fix_single_ensemble=fix_single_ensemble,
            max_tokens=max_tokens,
            max_atoms=max_atoms,
            pad_to_max_seqs=pad_to_max_seqs,
            compute_symmetries=compute_symmetries,
            binder_pocket_conditioned_prop=binder_pocket_conditioned_prop,
            contact_conditioned_prop=contact_conditioned_prop,
            binder_pocket_cutoff_min=binder_pocket_cutoff_min,
            binder_pocket_cutoff_max=binder_pocket_cutoff_max,
            binder_pocket_sampling_geometric_p=binder_pocket_sampling_geometric_p,
            only_ligand_binder_pocket=only_ligand_binder_pocket,
            only_pp_contact=only_pp_contact,
            single_sequence_prop=single_sequence_prop,
            msa_sampling=msa_sampling,
            override_bfactor=override_bfactor,
            override_method=override_method,
            compute_frames=compute_frames,
            override_coords=override_coords,
            bfactor_md_correction=bfactor_md_correction,
            compute_constraint_features=compute_constraint_features,
            inference_pocket_constraints=inference_pocket_constraints,
            inference_contact_constraints=inference_contact_constraints,
            compute_affinity=compute_affinity,
            **kwargs
        )
        
        # Add protein-protein specific masks
        protein_masks = self._create_protein_protein_masks(
            data, receptor_chains, binder_chains, max_tokens
        )
        features.update(protein_masks)
        
        # Add protein-protein specific features
        protein_features = self._compute_protein_protein_features(
            data, receptor_chains, binder_chains
        )
        features.update(protein_features)
        
        # Add metadata for downstream processing
        features["receptor_chain_ids"] = torch.tensor(receptor_chains, dtype=torch.long)
        features["binder_chain_ids"] = torch.tensor(binder_chains, dtype=torch.long)
        features["protein_protein_mode"] = torch.tensor(True, dtype=torch.bool)
        
        return features

    def process_protein_complex_affinity(
        self,
        data: Tokenized,
        random: np.random.Generator,
        molecules: dict[str, Mol],
        receptor_chain_ids: List[int],
        binder_chain_ids: List[int],
        affinity_value: Optional[float] = None,
        affinity_type: str = "Kd",
        training: bool = True,
        **process_kwargs,
    ) -> dict[str, Tensor]:
        """
        Convenience method for processing protein complexes with affinity data.
        
        Parameters
        ----------
        data : Tokenized
            The input data to the model.
        random : np.random.Generator
            Random number generator.
        molecules : dict[str, Mol]
            Dictionary of molecules.
        receptor_chain_ids : List[int]
            Chain IDs for receptor protein(s).
        binder_chain_ids : List[int]
            Chain IDs for binder protein(s).
        affinity_value : float, optional
            Experimental affinity value.
        affinity_type : str
            Type of affinity measurement (Kd, Ki, IC50, etc.).
        training : bool
            Whether in training mode.
        **process_kwargs
            Additional arguments for the process method.
            
        Returns
        -------
        dict[str, Tensor]
            Features with affinity information.
        """
        # Process with explicit chain assignments
        features = self.process(
            data=data,
            random=random,
            molecules=molecules,
            training=training,
            receptor_chain_ids=receptor_chain_ids,
            binder_chain_ids=binder_chain_ids,
            compute_affinity=True,
            **process_kwargs
        )
        
        # Add affinity target information if available
        if affinity_value is not None:
            features["affinity_target"] = torch.tensor(affinity_value, dtype=torch.float)
            features["affinity_type"] = affinity_type
            
            # Convert to standard units (log scale) if needed
            if affinity_type.lower() in ["kd", "ki", "ic50"]:
                # Convert to -log10(M) scale
                log_affinity = -math.log10(affinity_value * 1e-9)  # Assuming nM input
                features["log_affinity_target"] = torch.tensor(log_affinity, dtype=torch.float)
        
        return features


def create_protein_protein_featurizer() -> ProteinProteinFeaturizer:
    """
    Factory function to create a protein-protein featurizer.
        
    Returns
    -------
    ProteinProteinFeaturizer
        Configured featurizer for protein-protein complexes.
    """
    return ProteinProteinFeaturizer() 