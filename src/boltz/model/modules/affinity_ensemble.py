import torch
from torch import nn
from typing import Dict, Optional
import numpy as np

from boltz.data import const
from boltz.model.modules.affinity import AffinityModule
from boltz.model.modules.trunkv2 import InputEmbedder

from typing import Optional
from boltz.data.tokenize.boltz2 import TokenData, token_astuple
from boltz.data import const
from boltz.data.feature.protein_protein_featurizer import ProteinProteinFeaturizer
from boltz.data.mol import load_molecules
from boltz.data.parse.schema import parse_ccd_residue
from rdkit.Chem import Mol

class EnsembleProteinAffinityModule():
    """
    Ensemble-based protein-protein affinity prediction module.
    
    This module predicts protein-protein binding affinity by treating each residue
    of the binder protein as a single-token "ligand" and using the original 
    protein-ligand affinity module for prediction. The final prediction is obtained
    by ensemble averaging across all binder residues.
    """

    def __init__(
        self,
        input_embedder: InputEmbedder,
        affinity_module: AffinityModule,
        atomic_affinity: bool,
        ensemble_sampling_strategy: str = "top_k",  # "random", "top_k", "all"
        min_ensemble_size: int = 5,
        ccd_templates: Optional[Dict[str, Mol]] = None,
        mol_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the ensemble protein affinity module.
        
        Parameters
        ----------
        affinity_module : AffinityModule
            Pre-initialized protein-ligand affinity module to use for predictions
        atomic_affinity : bool
            Whether to use atomic-level tokenization (following ligand pattern)
        min_ensemble_size : int
            Minimum number of residues to include in ensemble
        ensemble_sampling_strategy : str
            Strategy for selecting residues: "random", "top_k", "all"
        ccd_templates : Optional[Dict[str, Mol]]
            Pre-loaded CCD templates for amino acids
        mol_dir : Optional[str]
            Directory containing CCD molecule files for loading on demand
        """
        super().__init__()
        
        # Use the provided affinity module
        self.input_embedder = input_embedder
        self.affinity_module = affinity_module
        self.atomic_affinity = atomic_affinity

        self.max_ensemble_size = 10 if self.atomic_affinity else 20
        self.min_ensemble_size = min_ensemble_size
        self.ensemble_sampling_strategy = ensemble_sampling_strategy
        self.featurizer = ProteinProteinFeaturizer()
        
        # CCD template management
        self.ccd_templates = ccd_templates or {}
        self.mol_dir = mol_dir

    def _get_amino_acid_ccd_template(self, res_name: str) -> Optional[Mol]:
        """
        Get CCD template for an amino acid.
        
        Parameters
        ----------
        res_name : str
            Three-letter amino acid code (e.g., 'ALA', 'GLY')
            
        Returns
        -------
        Optional[Mol]
            RDKit molecule with CCD template information, or None if not found
        """
        # Check if we already have the template cached
        if res_name in self.ccd_templates:
            return self.ccd_templates[res_name]
        
        # Try to load from mol_dir if available
        if self.mol_dir is not None:
            try:
                loaded_mols = load_molecules(self.mol_dir, [res_name])
                mol = loaded_mols[res_name]
                self.ccd_templates[res_name] = mol  # Cache for future use
                return mol
            except (ValueError, KeyError, FileNotFoundError):
                # CCD template not found, will fall back to simple approach
                pass
        
        return None

    def _identify_binder_residues(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Identify residues that belong to the binder protein.
        
        Parameters
        ----------
        feats : Dict[str, torch.Tensor]
            Feature dictionary containing affinity_token_mask
            
        Returns
        -------
        torch.Tensor
            Indices of binder residues
        """
        affinity_mask = feats["affinity_token_mask"]
        
        binder_indices = torch.where(affinity_mask[0] > 0)[0]
        
        return binder_indices

    def _select_ensemble_residues(
        self, 
        binder_indices: torch.Tensor,
        feats: Dict[str, torch.Tensor],
        x_pred: torch.Tensor,
        multiplicity: int = 1,
    ) -> torch.Tensor:
        """
        Select which binder residues to include in the ensemble.
        
        Parameters
        ----------
        binder_indices : torch.Tensor
            All binder residue indices
        feats : Dict[str, torch.Tensor]
            Feature dictionary
        x_pred : torch.Tensor
            Predicted atom coordinates (all atoms)
            
        Returns
        -------
        torch.Tensor
            Selected residue indices for ensemble
        """
        num_binder_residues = len(binder_indices)
        
        if num_binder_residues == 0:
            raise ValueError("No binder residues found for ensemble affinity prediction")
        
        # Ensure we have at least min_ensemble_size residues
        ensemble_size = min(
            max(num_binder_residues, self.min_ensemble_size),
            self.max_ensemble_size
        )
        
        if self.ensemble_sampling_strategy == "all" or num_binder_residues <= ensemble_size:
            return binder_indices
        
        elif self.ensemble_sampling_strategy == "random":
            # Randomly sample residues
            selected_indices = torch.randperm(num_binder_residues)[:ensemble_size]
            return binder_indices[selected_indices]
        
        elif self.ensemble_sampling_strategy == "top_k":
            # Select residues based on distance to receptor center
            
            receptor_mask = feats["receptor_mask"].to(torch.bool)
            
            if receptor_mask.sum() == 0:
                # Fallback to random if no receptor found
                selected_indices = torch.randperm(num_binder_residues)[:ensemble_size]
                return binder_indices[selected_indices]
            
            # Convert atom coordinates to token center coordinates
            token_to_rep_atom = feats["token_to_rep_atom"]
            token_to_rep_atom = feats["token_to_rep_atom"]
            token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)
            if len(x_pred.shape) == 4:
                B, mult, N, _ = x_pred.shape
                x_pred = x_pred.reshape(B * mult, N, -1)
            else:
                BM, N, _ = x_pred.shape
                B = BM // multiplicity
                mult = multiplicity
            
            # Get token center coordinates using representative atom mapping
            token_coords = torch.bmm(
                token_to_rep_atom.float(), 
                x_pred
            )[0]  # [tokens, 3]
            
            # Get receptor and binder coordinates
            receptor_indices = torch.where(receptor_mask[0] > 0)[0]
            receptor_coords = token_coords[receptor_indices]
            binder_coords = token_coords[binder_indices]
            
            # Compute minimum distances from each binder residue to any receptor residue
            distances = torch.cdist(binder_coords, receptor_coords)
            min_distances, _ = distances.min(dim=1)
            
            # Select closest residues
            _, closest_indices = torch.topk(min_distances, k=ensemble_size, largest=False)
            return binder_indices[closest_indices], min_distances[closest_indices]
        
        else:
            raise ValueError(f"Unknown ensemble sampling strategy: {self.ensemble_sampling_strategy}")

    def _compute_ensemble_weights(
        self, 
        min_distances: torch.Tensor,
        sigma: float = 4.0
    ):
        """
        Create feature dictionary where only one residue is marked as the ligand.
        
        Parameters
        ----------
        min_distances : torch.Tensor
            Distances of selected residues
       
        Returns
        -------
        torch.Tensor
            weights for scores and probabilities
        """
        weights = torch.ones(len(min_distances), device=min_distances.device)
        
        # Weight by minimum distance to receptor
        distance_weights = 1.0 / (1.0 + min_distances/sigma)
        
        weights = weights * distance_weights

        return  weights / weights.sum()
        
    def _retokenize_residue_as_atoms(self, tokenized_data, residue_token_idx):
        """
        Retokenize a specific residue using CCD template information for ligand-like properties.
        
        This enhanced version uses CCD templates to get proper bond information,
        atomic properties, and geometric constraints like ligands do.
        """
        # Get the residue token
        residue_token = tokenized_data.tokens[residue_token_idx]
        
        # Get the underlying structure data
        struct = tokenized_data.structure
        
        # Find the residue in the structure
        res_idx = residue_token['res_idx']
        asym_id = residue_token['asym_id']

        res_name = residue_token['res_name'] if 'res_name' in residue_token.dtype.names else None

        
        # Find the chain and residue
        chain = None
        residue = None
        for c in struct.chains:
            if c['asym_id'] == asym_id:
                chain = c
                res_start = c['res_idx']
                res_end = c['res_idx'] + c['res_num']
                for r in struct.residues[res_start:res_end]:
                    if r['res_idx'] == res_idx:
                        residue = r
                        break
                break
        
        if chain is None or residue is None:
            raise ValueError(f"Could not find residue {res_idx} in chain {asym_id}")
        
        # Get atom data for this residue
        atom_start = residue['atom_idx']
        atom_end = residue['atom_idx'] + residue['atom_num']
        atom_data = struct.atoms[atom_start:atom_end]
        
        # Get coordinates (using first ensemble)
        offset = struct.ensemble[0]['atom_coord_idx']
    
        atom_coords = struct.coords[offset + atom_start : offset + atom_end]['coords']
        
        # Try to get CCD template for enhanced properties
        ccd_mol = None
        parsed_ccd_residue = None
        if res_name:
            ccd_mol = self._get_amino_acid_ccd_template(res_name)
            if ccd_mol is not None:
                try:
                    # Parse the CCD residue to get bond and constraint information
                    parsed_ccd_residue = parse_ccd_residue(res_name, ccd_mol, res_idx, drop_leaving_atoms=True)
                except Exception as e:
                    print(f"Warning: Could not parse CCD template for {res_name}: {e}")
                    parsed_ccd_residue = None

        # Create atomic tokens with enhanced CCD information
        atomic_tokens = []
        unk_token = const.unk_token["PROTEIN"]
        unk_id = const.token_ids[unk_token]
        
        # Start token numbering from the original residue token index
        token_idx = tokenized_data.tokens[residue_token_idx][0]
        
        # Create atom name to index mapping for CCD bond lookup
        atom_name_to_token_idx = {}

        for i, atom in enumerate(atom_data):
            # Token is present if atom is present
            is_present = residue['is_present'] & atom['is_present']
            atom_index = atom_start + i
            
            # Get enhanced properties from CCD template if available
            atom_name = atom['name'] if 'name' in atom.dtype.names else f"ATOM_{i}"
            element = atom['element'] if 'element' in atom.dtype.names else 6  # Default to carbon
            charge = 0  # Default charge
            
            # If we have CCD parsed residue, try to get enhanced atom properties
            if parsed_ccd_residue is not None:
                for ccd_atom in parsed_ccd_residue.atoms:
                    if ccd_atom.name == atom_name:
                        if hasattr(ccd_atom, 'element'):
                            element = ccd_atom.element
                        if hasattr(ccd_atom, 'charge'):
                            charge = ccd_atom.charge
                        break
            
            # Create atomic token with enhanced properties
            atomic_token = TokenData(
                token_idx=token_idx,
                atom_idx=atom_index,
                atom_num=1,  # Each atomic token represents 1 atom
                res_idx=residue['res_idx'],
                res_type=unk_id,  # Use UNK token like NONPOLYMER
                res_name=residue['name'],
                sym_id=chain['sym_id'],
                asym_id=chain['asym_id'],
                entity_id=chain['entity_id'],
                mol_type=const.chain_type_ids["NONPOLYMER"],  # Mark as ligand-like
                center_idx=atom_index,  # Each atom is its own center
                disto_idx=atom_index,   # Each atom is its own disto
                center_coords=atom_coords[i],
                disto_coords=atom_coords[i],
                resolved_mask=is_present,
                disto_mask=is_present,
                modified=False,  # NONPOLYMER atoms are not modified
                frame_rot=np.eye(3).flatten(),
                frame_t=np.zeros(3),
                frame_mask=False,
                cyclic_period=chain['cyclic_period'],
                affinity_mask=True,  # Mark for affinity prediction
            )
            
            atomic_tokens.append(token_astuple(atomic_token))
            atom_name_to_token_idx[atom_name] = token_idx
            token_idx += 1
    

        # Store enhanced bond information for potential use in featurization
        enhanced_bonds = []
        if parsed_ccd_residue is not None and parsed_ccd_residue.bonds:
            # Map CCD bonds to our token indices
            for bond in parsed_ccd_residue.bonds:
                # CCD bonds reference atom indices within the parsed residue
                if (bond.atom_1 < len(parsed_ccd_residue.atoms) and 
                    bond.atom_2 < len(parsed_ccd_residue.atoms)):
                    
                    atom1_name = parsed_ccd_residue.atoms[bond.atom_1].name
                    atom2_name = parsed_ccd_residue.atoms[bond.atom_2].name
    
                    # Find corresponding token indices
                    if (atom1_name in atom_name_to_token_idx and 
                        atom2_name in atom_name_to_token_idx):
                        
                        token1_idx = atom_name_to_token_idx[atom1_name]
                        token2_idx = atom_name_to_token_idx[atom2_name]
    
                        # Is the bond type right
                        enhanced_bonds.append((token1_idx, token2_idx, bond.type + 1))
        
        # Store enhanced information for potential future use
    
        return atomic_tokens, enhanced_bonds
    
    def _create_enhanced_bonds(self, tokenized_data, atomic_tokens, enhanced_bonds, residue_token_idx):
        """
        Create enhanced bond array incorporating CCD template bond information.
        
        Parameters
        ----------
        tokenized_data : Tokenized
            Original tokenized structure
        atomic_tokens : list
            List of atomic tokens for the retokenized residue
        enhanced_bonds : list
            Bond information from CCD template
        residue_token_idx : int
            Index of the original residue token being replaced
            
        Returns
        -------
        np.ndarray
            Enhanced bond array with CCD bond information
        """
        from boltz.data.types import TokenBondV2
        
        # Start with original bonds, filtering out any that involve the residue being replaced
        original_bonds = tokenized_data.bonds
        filtered_bonds = []
        
        # Get the range of original token indices that will be replaced
        num_atomic = len(atomic_tokens)
        original_token_idx = tokenized_data.tokens[residue_token_idx][0]  # token_idx field
        
        # Filter out bonds involving the original residue token
        for bond in original_bonds:
            token1, token2, bond_type = bond # TokenBondV2 format (token1, token2, bond_type)
                
            # Keep bonds that don't involve the original residue token
            if token1 != original_token_idx and token2 != original_token_idx:
                # Adjust token indices for tokens that come after the insertion point
                if token1 > original_token_idx:
                    token1 += (num_atomic - 1)
                if token2 > original_token_idx:
                    token2 += (num_atomic - 1)
                    
                filtered_bonds.append((token1, token2, bond_type))
        
        # Add enhanced bonds from CCD template
        for bond_info in enhanced_bonds:
            # TokenBondV2 format
            filtered_bonds.append(bond_info)
        
        # Convert to appropriate numpy array
        return np.array(filtered_bonds, dtype=TokenBondV2)

    def _create_retokenized_structure(self, tokenized_data, residue_token_idx):
        """Create new tokenized structure with one residue converted to atomic tokens and enhanced bonds."""
        
        # Get atomic tokens for the selected residue - this now returns enhanced information
        atomic_tokens, enhanced_bonds = self._retokenize_residue_as_atoms(tokenized_data, residue_token_idx)
        
        # Create new token array
        original_tokens = tokenized_data.tokens
        num_original = len(original_tokens)
        num_atomic = len(atomic_tokens)
        num_new = num_original - 1 + num_atomic  # Remove 1 residue, add N atoms
        
        # Create new token array
        new_tokens = np.zeros(num_new, dtype=original_tokens.dtype)
        
        # Copy tokens before the residue
        if residue_token_idx > 0:
            new_tokens[:residue_token_idx] = original_tokens[:residue_token_idx]
        
        # Insert atomic tokens
        for i, atomic_token in enumerate(atomic_tokens):
            new_tokens[residue_token_idx + i] = atomic_token
        
        # Copy tokens after the residue
        remaining_start = residue_token_idx + num_atomic
        original_remaining_start = residue_token_idx + 1
        if original_remaining_start < num_original:
            # Update token indices for remaining tokens
            remaining_tokens = original_tokens[original_remaining_start:].copy().tolist()
            # Adjust token_idx for all remaining tokens
            for i in range(len(remaining_tokens)):
                remaining_tokens[i] = list(remaining_tokens[i])
                remaining_tokens[i][0] += (num_atomic - 1)  # token_idx adjustment
                remaining_tokens[i] = tuple(remaining_tokens[i])
            
            new_tokens[remaining_start:] = remaining_tokens
        
        # Create enhanced bonds incorporating CCD template information
        enhanced_bond_array = self._create_enhanced_bonds(
            tokenized_data, atomic_tokens, enhanced_bonds, residue_token_idx
        )
        
        # Create new tokenized object with atomic tokens and enhanced bonds
        from boltz.data.types import Tokenized
        new_tokenized = Tokenized(
            tokens=new_tokens,
            bonds=enhanced_bond_array,  # Use enhanced bonds with CCD information
            structure=tokenized_data.structure,
            msa=tokenized_data.msa,
            record=tokenized_data.record,
            residue_constraints=tokenized_data.residue_constraints,
            templates=tokenized_data.templates,
            template_tokens=tokenized_data.template_tokens,
            template_bonds=tokenized_data.template_bonds,
            extra_mols=tokenized_data.extra_mols,
        )
        
        return new_tokenized
    
    def _create_atomic_features(self, feats, residue_token_idx):
        """Create features from retokenized structure using standard featurizer."""
        
        # Create retokenized structure with atomic tokens
        atomic_tokenized = self._create_retokenized_structure(feats['featurizer_args'][0]['data'], residue_token_idx)

        num_atomic = len(atomic_tokenized.tokens) - len(feats['featurizer_args'][0]['data'].tokens)
        # Use standard featurizer to generate features
        random_gen = np.random.default_rng(42)
        
        atomic_feats = self.featurizer.process(**feats['featurizer_args'][0])
        
        for key, value in atomic_feats.items():
            if isinstance(value, torch.Tensor):
                atomic_feats[key] = atomic_feats[key].unsqueeze(0).to(device = feats["affinity_token_mask"].device)
        
        new_affinity_mask = torch.zeros_like(atomic_feats["affinity_token_mask"])
        new_affinity_mask[0, residue_token_idx:residue_token_idx+num_atomic] = 1.0
        atomic_feats["affinity_token_mask"] = new_affinity_mask
        
        return atomic_feats
    
    def _create_single_residue_features(
        self, 
        feats: Dict[str, torch.Tensor], 
        residue_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Create feature dictionary where only one residue is marked as the ligand.
        
        Parameters
        ----------
        feats : Dict[str, torch.Tensor]
            Original feature dictionary
        residue_idx : int
            Index of the residue to treat as ligand
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Modified feature dictionary
        """
        # Create a copy of features
        single_residue_feats = {}
        for key, value in feats.items():
            single_residue_feats[key] = value.clone() if isinstance(value, torch.Tensor) else value

        # Create new affinity mask with only one residue active
        new_affinity_mask = torch.zeros_like(feats["affinity_token_mask"])
        new_affinity_mask[0, residue_idx] = 1.0
        single_residue_feats["affinity_token_mask"] = new_affinity_mask
        
        # Update mol_type to mark this residue as NONPOLYMER (ligand-like)
        new_mol_type = feats["mol_type"].clone()
        new_mol_type[0, residue_idx] = const.chain_type_ids["NONPOLYMER"]
        single_residue_feats["mol_type"] = new_mol_type
        
        return single_residue_feats

    def forward(
        self,
        z: torch.Tensor,
        x_pred: torch.Tensor,
        feats: Dict[str, torch.Tensor],
        run_recycling: nn.Module,
        multiplicity: int = 1,
        recycling_steps: int = 1,
        use_kernels: bool = False,
        ensemble_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass using ensemble averaging over binder residues.
        
        Parameters
        ----------
        s_inputs : torch.Tensor
            Token embeddings
        z : torch.Tensor
            Pair embeddings  
        x_pred : torch.Tensor
            Predicted coordinates
        feats : Dict[str, torch.Tensor]
            Feature dictionary
        multiplicity : int
            Multiplicity factor
        use_kernels : bool
            Whether to use optimized kernels
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Averaged affinity predictions
        """
        # Identify binder residues
        binder_indices = self._identify_binder_residues(feats)
        
        # Select ensemble residues
        ensemble_indices, min_distances = self._select_ensemble_residues(binder_indices, feats, x_pred, multiplicity)
        # Collect predictions from each ensemble member
        ensemble_predictions = []
        ensemble_probabilities = []

        for residue_idx in ensemble_indices:
            # Create single-residue features
            #
            if self.atomic_affinity:
                single_residue_feats = self._create_atomic_features(feats, residue_idx)
            else:
                single_residue_feats = self._create_single_residue_features(feats, residue_idx.item())   
            
            pad_token_mask = single_residue_feats["token_pad_mask"][0]
            rec_mask = single_residue_feats["receptor_mask"][0].to(torch.bool)
            rec_mask = rec_mask * pad_token_mask
            # Create cross_pair_mask for this single residue
            lig_mask = single_residue_feats["affinity_token_mask"][0].to(torch.bool)  # single ligand residue
            lig_mask = lig_mask * pad_token_mask
            
            cross_pair_mask = (
                lig_mask[:, None] * rec_mask[None, :]
                + rec_mask[:, None] * lig_mask[None, :]
                + lig_mask[:, None] * lig_mask[None, :]
            )

            s_inputs = self.input_embedder(single_residue_feats, affinity=True)
            if self.atomic_affinity:
                z = run_recycling(single_residue_feats, 1).detach()
            # Apply mask to z
            z_masked = z * cross_pair_mask[None, :, :, None]
            
            # Run affinity prediction for this single residue
            single_pred = self.affinity_module(
                s_inputs=s_inputs,
                z=z_masked,
                x_pred=x_pred,
                feats=single_residue_feats,
                multiplicity=multiplicity,
                use_kernels=use_kernels,
            )
            
            ensemble_predictions.append(single_pred["affinity_pred_value"])
            # Convert logits to probabilities first, then ensemble average
            single_probability = torch.sigmoid(single_pred["affinity_logits_binary"])
            ensemble_probabilities.append(single_probability)
        
        # Ensemble averaging
        if len(ensemble_predictions) > 0:
            if ensemble_weights:
                weights = self._compute_ensemble_weights(min_distances)
            else:
                weights = torch.ones(len(min_distances), device=min_distances.device) / len(min_distances)

            # Average predictions and probabilities (following original Boltz2 approach)
            avg_prediction = (torch.stack(ensemble_predictions) * weights[:, None, None]).sum(dim=0)
            avg_probability = (torch.stack(ensemble_probabilities) * weights[:, None, None]).sum(dim=0)
            # Convert averaged probability back to logits for consistency
            avg_logits = torch.logit(torch.clamp(avg_probability, min=1e-7, max=1-1e-7))
            
            # Calculate prediction variance for uncertainty estimation
            if len(ensemble_predictions) > 1:
                pred_variance = torch.stack(ensemble_predictions).var(dim=0)
                uncertainty = torch.sqrt(pred_variance)
            else:
                uncertainty = torch.zeros_like(avg_prediction)
            
            # Also return per-residue affinity values and probabilities
            binder_affinity_values = torch.stack(ensemble_predictions)           # [num_members, batch, 1]
            binder_affinity_probabilities = torch.stack(ensemble_probabilities) # [num_members, batch, 1]
            binder_indices = ensemble_indices

            return {
                "affinity_pred_value": avg_prediction,
                "affinity_logits_binary": avg_logits,
                "affinity_probability_binary": avg_probability,
                "ensemble_uncertainty": uncertainty,
                "ensemble_size": torch.tensor(len(ensemble_predictions), dtype=torch.long),
                "ensemble_residue_indices": ensemble_indices,
                "binder_affinity_values": binder_affinity_values,
                "binder_affinity_probabilities": binder_affinity_probabilities,
                "binder_residue_indices": binder_indices,
            }
        else:
            # Fallback if no ensemble members
            batch_size = s_inputs.shape[0]
            device = s_inputs.device
            return {
                "affinity_pred_value": torch.zeros((batch_size, 1), device=device),
                "affinity_logits_binary": torch.zeros((batch_size, 1), device=device),
                "affinity_probability_binary": torch.zeros((batch_size, 1), device=device),
                "ensemble_uncertainty": torch.zeros((batch_size, 1), device=device),
                "ensemble_size": torch.tensor(0, dtype=torch.long),
                "ensemble_residue_indices": torch.tensor([], dtype=torch.long),
                "binder_affinity_values": torch.zeros((0, batch_size, 1), device=device),
                "binder_affinity_probabilities": torch.zeros((0, batch_size, 1), device=device),
                "binder_residue_indices": torch.tensor([], dtype=torch.long),
            }


def create_ensemble_protein_affinity_module(
    input_embedder: InputEmbedder,
    affinity_module: AffinityModule,
    atomic_affinity: bool,
    adaptive: bool = True,
    ccd_templates: Optional[Dict[str, Mol]] = None,
    mol_dir: Optional[str] = None,
    **kwargs
) -> EnsembleProteinAffinityModule:
    """
    Factory function to create an ensemble protein affinity module.
    
    Parameters
    ----------
    input_embedder : InputEmbedder
        Input embedder for the model
    affinity_module : AffinityModule
        Pre-initialized protein-ligand affinity module to use for predictions
    atomic_affinity : bool
        Whether to use atomic-level tokenization
    adaptive : bool
        Whether to use the adaptive interface-aware version
    ccd_templates : Optional[Dict[str, Mol]]
        Pre-loaded CCD templates for amino acids
    mol_dir : Optional[str]
        Directory containing CCD molecule files for loading on demand
    **kwargs
        Additional arguments passed to the module constructor
        
    Returns
    -------
    EnsembleProteinAffinityModule
        Configured ensemble affinity module
    """
    return EnsembleProteinAffinityModule(
            input_embedder=input_embedder,
            affinity_module=affinity_module,
            atomic_affinity=atomic_affinity,
            ccd_templates=ccd_templates,
            mol_dir=mol_dir,
            **kwargs
        ) 
