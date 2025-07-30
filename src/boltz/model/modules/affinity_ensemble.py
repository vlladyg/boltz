import torch
from torch import nn
from typing import Dict, Optional
import numpy as np

from boltz.data import const
from boltz.model.modules.affinity import AffinityModule


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
        affinity_module: AffinityModule,
        ensemble_sampling_strategy: str = "top_k",  # "random", "top_k", "all"
        max_ensemble_size: int = 20,
        min_ensemble_size: int = 5,
        **kwargs
    ):
        """
        Initialize the ensemble protein affinity module.
        
        Parameters
        ----------
        affinity_module : AffinityModule
            Pre-initialized protein-ligand affinity module to use for predictions
        max_ensemble_size : int
            Maximum number of residues to include in ensemble (for computational efficiency)
        min_ensemble_size : int
            Minimum number of residues to include in ensemble
        ensemble_sampling_strategy : str
            Strategy for selecting residues: "random", "top_k", "all"
        """
        super().__init__()
        
        # Use the provided affinity module
        self.affinity_module = affinity_module
        
        self.max_ensemble_size = max_ensemble_size
        self.min_ensemble_size = min_ensemble_size
        self.ensemble_sampling_strategy = ensemble_sampling_strategy

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
            return binder_indices[closest_indices]
        
        else:
            raise ValueError(f"Unknown ensemble sampling strategy: {self.ensemble_sampling_strategy}")

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
        #new_mol_type[0, residue_idx] = const.chain_type_ids["NONPOLYMER"]
        single_residue_feats["mol_type"] = new_mol_type
        
        return single_residue_feats

    def forward(
        self,
        s_inputs: torch.Tensor,
        z: torch.Tensor,
        x_pred: torch.Tensor,
        feats: Dict[str, torch.Tensor],
        multiplicity: int = 1,
        use_kernels: bool = False,
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
        ensemble_indices = self._select_ensemble_residues(binder_indices, feats, x_pred, multiplicity)
        
        # Collect predictions from each ensemble member
        ensemble_predictions = []
        ensemble_probabilities = []


        pad_token_mask = feats["token_pad_mask"][0]
        rec_mask = feats["receptor_mask"][0].to(torch.bool)
        rec_mask = rec_mask * pad_token_mask
        for residue_idx in ensemble_indices:
            # Create single-residue features
            single_residue_feats = self._create_single_residue_features(feats, residue_idx.item())
            
            # Create cross_pair_mask for this single residue
            lig_mask = single_residue_feats["affinity_token_mask"][0].to(torch.bool)  # single ligand residue
            lig_mask = lig_mask * pad_token_mask
            
            cross_pair_mask = (
                lig_mask[:, None] * rec_mask[None, :]
                + rec_mask[:, None] * lig_mask[None, :]
                + lig_mask[:, None] * lig_mask[None, :]
            )

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
            print("Ensemble probablilities")
            print(ensemble_probabilities)
            # Average predictions and probabilities (following original Boltz2 approach)
            avg_prediction = torch.stack(ensemble_predictions).mean(dim=0)
            avg_probability = torch.stack(ensemble_probabilities).mean(dim=0)
            # Convert averaged probability back to logits for consistency
            avg_logits = torch.logit(torch.clamp(avg_probability, min=1e-7, max=1-1e-7))
            
            # Calculate prediction variance for uncertainty estimation
            if len(ensemble_predictions) > 1:
                pred_variance = torch.stack(ensemble_predictions).var(dim=0)
                uncertainty = torch.sqrt(pred_variance)
            else:
                uncertainty = torch.zeros_like(avg_prediction)
            
            return {
                "affinity_pred_value": avg_prediction,
                "affinity_logits_binary": avg_logits,
                "affinity_probability_binary": avg_probability,
                "ensemble_uncertainty": uncertainty,
                "ensemble_size": torch.tensor(len(ensemble_predictions), dtype=torch.long),
                "ensemble_residue_indices": ensemble_indices,
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