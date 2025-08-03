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
        **kwargs
    ):
        """
        Initialize the ensemble protein affinity module.
        
        Parameters
        ----------
        affinity_module : AffinityModule
            Pre-initialized protein-ligand affinity module to use for predictions
        min_ensemble_size : int
            Minimum number of residues to include in ensemble
        ensemble_sampling_strategy : str
            Strategy for selecting residues: "random", "top_k", "all"
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
        Retokenize a specific residue using Boltz's NONPOLYMER atomic tokenization logic.
        
        This uses the exact same code path as ligand tokenization (lines 260-307 in boltz2.py).
        """
        # Get the residue token
        residue_token = tokenized_data.tokens[residue_token_idx]
        
        # Get the underlying structure data
        struct = tokenized_data.structure
        
        # Find the residue in the structure
        res_idx = residue_token['res_idx']
        asym_id = residue_token['asym_id']
        
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
        
        # Create atomic tokens using NONPOLYMER tokenization logic
        atomic_tokens = []
        unk_token = const.unk_token["PROTEIN"]
        unk_id = const.token_ids[unk_token]
        
        # Start token numbering from the original residue token index
        token_idx = tokenized_data.tokens[residue_token_idx][0]
        
        for i, atom in enumerate(atom_data):
            # Token is present if atom is present
            is_present = residue['is_present'] & atom['is_present']
            atom_index = atom_start + i
            
            # Create atomic token (following lines 277-302 in boltz2.py exactly)
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
            token_idx += 1
        
        return atomic_tokens
    
    def _create_retokenized_structure(self, tokenized_data, residue_token_idx):
        """Create new tokenized structure with one residue converted to atomic tokens."""
        
        # Get atomic tokens for the selected residue
        atomic_tokens = self._retokenize_residue_as_atoms(tokenized_data, residue_token_idx)
        
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
                #print(remaining_tokens[i])
                remaining_tokens[i] = list(remaining_tokens[i])
                remaining_tokens[i][0] += (num_atomic - 1)  # token_idx adjustment
                remaining_tokens[i] = tuple(remaining_tokens[i])
            
            new_tokens[remaining_start:] = remaining_tokens
        
        # Create new tokenized object with atomic tokens
        from boltz.data.types import Tokenized
        new_tokenized = Tokenized(
            tokens=new_tokens,
            bonds=tokenized_data.bonds,  # Keep original bonds for simplicity
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


class AdaptiveEnsembleProteinAffinityModule(EnsembleProteinAffinityModule):
    """
    Adaptive ensemble protein affinity module with interface-aware residue selection.
    
    This extends the basic ensemble approach by adaptively selecting residues
    based on their proximity to the protein-protein interface.
    """

    def __init__(
        self,
        affinity_module: AffinityModule,
        interface_cutoff: float = 8.0,
        **kwargs
    ):
        """
        Initialize adaptive ensemble module.
        
        Parameters
        ----------
        affinity_module : AffinityModule
            Pre-initialized protein-ligand affinity module to use for predictions
        interface_cutoff : float
            Distance cutoff for defining interface residues
        """
        super().__init__(
            affinity_module=affinity_module,
            **kwargs
        )
        self.interface_cutoff = interface_cutoff

    def _select_ensemble_residues(
        self, 
        binder_indices: torch.Tensor,
        feats: Dict[str, torch.Tensor],
        x_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select ensemble residues based on interface proximity.
        
        Prioritizes residues that are close to the protein-protein interface.
        """
        num_binder_residues = len(binder_indices)
        
        if num_binder_residues == 0:
            raise ValueError("No binder residues found for ensemble affinity prediction")
        
        # Get receptor residues
        receptor_mask = feats["mol_type"] == const.chain_type_ids["PROTEIN"]
        receptor_mask = receptor_mask & (feats["affinity_token_mask"] == 0)
        receptor_indices = torch.where(receptor_mask)[0]
        
        if len(receptor_indices) == 0:
            # Fallback to parent method
            return super()._select_ensemble_residues(binder_indices, feats, x_pred)
        
        # Convert atom coordinates to token center coordinates
        token_to_rep_atom = feats["token_to_rep_atom"]
        
        # Handle different x_pred shapes
        if len(x_pred.shape) == 4:
            # Shape: [batch, multiplicity, atoms, 3] -> use first sample
            x_pred_atoms = x_pred[0, 0]  # [atoms, 3]
        elif len(x_pred.shape) == 3:
            # Shape: [batch*multiplicity, atoms, 3] -> use first batch
            x_pred_atoms = x_pred[0]  # [atoms, 3]
        else:
            # Shape: [atoms, 3]
            x_pred_atoms = x_pred
        
        # Get token center coordinates using representative atom mapping
        token_coords = torch.bmm(
            token_to_rep_atom[None].float(), 
            x_pred_atoms[None]
        )[0]  # [tokens, 3]
        
        # Compute interface residues using token coordinates
        binder_coords = token_coords[binder_indices]
        receptor_coords = token_coords[receptor_indices]
        
        # Compute minimum distances from each binder residue to any receptor residue
        distances = torch.cdist(binder_coords, receptor_coords)
        min_distances, _ = distances.min(dim=1)
        
        # Select interface residues (within cutoff)
        interface_mask = min_distances <= self.interface_cutoff
        interface_indices = binder_indices[interface_mask]
        
        # If we have enough interface residues, use them preferentially
        if len(interface_indices) >= self.min_ensemble_size:
            ensemble_size = min(len(interface_indices), self.max_ensemble_size)
            
            if len(interface_indices) <= ensemble_size:
                return interface_indices
            else:
                # Select closest interface residues
                interface_distances = min_distances[interface_mask]
                _, closest_interface = torch.topk(
                    interface_distances, k=ensemble_size, largest=False
                )
                return interface_indices[closest_interface]
        else:
            # Supplement interface residues with closest non-interface residues
            ensemble_size = min(num_binder_residues, self.max_ensemble_size)
            _, closest_all = torch.topk(min_distances, k=ensemble_size, largest=False)
            return binder_indices[closest_all]


def create_ensemble_protein_affinity_module(
    affinity_module: AffinityModule,
    adaptive: bool = True,
    **kwargs
) -> EnsembleProteinAffinityModule:
    """
    Factory function to create an ensemble protein affinity module.
    
    Parameters
    ----------
    affinity_module : AffinityModule
        Pre-initialized protein-ligand affinity module to use for predictions
    adaptive : bool
        Whether to use the adaptive interface-aware version
    **kwargs
        Additional arguments passed to the module constructor
        
    Returns
    -------
    EnsembleProteinAffinityModule
        Configured ensemble affinity module
    """
    if adaptive:
        return AdaptiveEnsembleProteinAffinityModule(
            affinity_module=affinity_module,
            **kwargs
        )
    else:
        return EnsembleProteinAffinityModule(
            affinity_module=affinity_module,
            **kwargs
        ) 