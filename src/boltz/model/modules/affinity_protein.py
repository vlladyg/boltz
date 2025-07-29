import torch
from torch import nn
from typing import Dict, Optional, Tuple

import boltz.model.layers.initialize as init
from boltz.model.layers.pairformer import PairformerNoSeqModule
from boltz.model.modules.encodersv2 import PairwiseConditioning
from boltz.model.modules.transformersv2 import DiffusionTransformer
from boltz.model.modules.utils import LinearNoBias
from boltz.model.modules.affinity import AffinityModule, AffinityHeadsTransformer


class ProteinProteinAffinityModule(AffinityModule):
    """
    Extended AffinityModule that supports protein-protein binding affinity prediction.
    Uses exactly the same weights as the original module with no additional parameters.
    """

    def __init__(
        self,
        token_s: int,
        token_z: int,
        protein_ligand_mode: bool,
        pairformer_args: dict,
        transformer_args: dict,
        num_dist_bins: int = 64,
        max_dist: float = 22.0,
        use_cross_transformer: bool = False,
        groups: dict = {},
    ):
        """Initialize using exactly the same parameters as original module."""
        super().__init__(
            token_s=token_s,
            token_z=token_z,
            pairformer_args=pairformer_args,
            transformer_args=transformer_args,
            num_dist_bins=num_dist_bins,
            max_dist=max_dist,
            use_cross_transformer=use_cross_transformer,
            groups=groups,
        )
        
        self.protein_ligand_mode = protein_ligand_mode

    def _detect_protein_ligand_mode(self, feats: Dict[str, torch.Tensor]) -> bool:
        """Detect if we're dealing with protein-protein complexes based on features."""
        has_receptor_mask = "receptor_mask" in feats
        has_binder_mask = "binder_mask" in feats
        has_interface_mask = "interface_mask" in feats
        
        if "mol_type" in feats:
            protein_tokens = (feats["mol_type"] == 0).sum()
            total_tokens = feats["mol_type"].shape[0]
            protein_ratio = protein_tokens.float() / total_tokens.float()
            
            if protein_ratio > 0.5:
                return True
        
        return has_receptor_mask or has_binder_mask or has_interface_mask

    def _create_protein_protein_masks(
        self, 
        feats: Dict[str, torch.Tensor], 
        multiplicity: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create masks for protein-protein interactions."""
        pad_token_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        
        if "receptor_mask" in feats and "binder_mask" in feats:
            rec_mask = feats["receptor_mask"].repeat_interleave(multiplicity, 0)
            binder_mask = feats["binder_mask"].repeat_interleave(multiplicity, 0)
        else:
            rec_mask = (feats["mol_type"] == 0).repeat_interleave(multiplicity, 0)
            binder_mask = (
                feats["affinity_token_mask"]
                .repeat_interleave(multiplicity, 0)
                .to(torch.bool)
            )
        
        rec_mask = rec_mask * pad_token_mask
        binder_mask = binder_mask * pad_token_mask
        
        return pad_token_mask, rec_mask, binder_mask

    def _create_cross_pair_mask(
        self, 
        rec_mask: torch.Tensor, 
        binder_mask: torch.Tensor,
        include_binder_binder: bool = True
    ) -> torch.Tensor:
        """Create cross-pair mask for protein-protein interactions."""
        cross_pair_mask = (
            binder_mask[:, :, None] * rec_mask[:, None, :]
            + rec_mask[:, :, None] * binder_mask[:, None, :]
        )
        
        if include_binder_binder:
            cross_pair_mask = cross_pair_mask + (
                binder_mask[:, :, None] * binder_mask[:, None, :]
            )
        
        return cross_pair_mask

    def forward(
        self,
        s_inputs: torch.Tensor,
        z: torch.Tensor,
        x_pred: torch.Tensor,
        feats: Dict[str, torch.Tensor],
        multiplicity: int = 1,
        use_kernels: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass using only original weights."""
        # Detect protein-protein mode if not explicitly set
        if not hasattr(self, 'protein_ligand_mode'):
            self.protein_ligand_mode = self._detect_protein_ligand_mode(feats)
        
        # Use original normalization and projection
        z = self.z_linear(self.z_norm(z))
        z = z.repeat_interleave(multiplicity, 0)

        # Add sequence-to-pairwise projections (original)
        z = (
            z
            + self.s_to_z_prod_in1(s_inputs)[:, :, None, :]
            + self.s_to_z_prod_in2(s_inputs)[:, None, :, :]
        )

        # Compute distance features (original)
        token_to_rep_atom = feats["token_to_rep_atom"]
        token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)
        
        if len(x_pred.shape) == 4:
            B, mult, N, _ = x_pred.shape
            x_pred = x_pred.reshape(B * mult, N, -1)
        else:
            BM, N, _ = x_pred.shape
            B = BM // multiplicity
            mult = multiplicity
            
        x_pred_repr = torch.bmm(token_to_rep_atom.float(), x_pred)
        d = torch.cdist(x_pred_repr, x_pred_repr)

        # Create distance bins (original)
        distogram = (d.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        distogram = self.dist_bin_pairwise_embed(distogram)

        # Add pairwise conditioning (original)
        z = z + self.pairwise_conditioner(z_trunk=z, token_rel_pos_feats=distogram)

        # Create masks based on mode
        if self.protein_ligand_mode:
            pad_token_mask, rec_mask, binder_mask = self._create_protein_protein_masks(feats, multiplicity)
            cross_pair_mask = self._create_cross_pair_mask(rec_mask, binder_mask)
        else:
            # Original small molecule logic
            pad_token_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
            rec_mask = (feats["mol_type"] == 0).repeat_interleave(multiplicity, 0)
            rec_mask = rec_mask * pad_token_mask
            binder_mask = (
                feats["affinity_token_mask"]
                .repeat_interleave(multiplicity, 0)
                .to(torch.bool)
            )
            binder_mask = binder_mask * pad_token_mask
            cross_pair_mask = (
                binder_mask[:, :, None] * rec_mask[:, None, :]
                + rec_mask[:, :, None] * binder_mask[:, None, :]
                + binder_mask[:, :, None] * binder_mask[:, None, :]
            )

        # Process through pairformer (original)
        z = self.pairformer_stack(
            z,
            pair_mask=cross_pair_mask,
            use_kernels=use_kernels,
        )

        # Use original affinity heads
        out_dict = self.affinity_heads(z=z, feats=feats, multiplicity=multiplicity)

        # For protein-protein mode, we reuse the original affinity predictions
        # but scale them appropriately for protein-protein interactions
        if self.protein_ligand_mode:
            # Scale the affinity prediction based on interface size
            interface_size = cross_pair_mask.sum(dim=(1, 2))
            scaling_factor = torch.log1p(interface_size) / torch.log(torch.tensor(100.0))
            
            # Apply scaling to affinity prediction
            out_dict["affinity_pred_value"] = out_dict["affinity_pred_value"] * scaling_factor.unsqueeze(-1)
            
            # Compute binding probability using original logits
            out_dict["affinity_probability_binary"] = torch.sigmoid(out_dict["affinity_logits_binary"])

        return out_dict


def create_protein_protein_affinity_module(
    token_s: int,
    token_z: int,
    pairformer_args: dict,
    transformer_args: dict,
    protein_ligand_mode: bool = False,
    **kwargs
) -> ProteinProteinAffinityModule:
    """Factory function to create a protein-protein affinity module."""
    return ProteinProteinAffinityModule(
        token_s=token_s,
        token_z=token_z,
        pairformer_args=pairformer_args,
        transformer_args=transformer_args,
        protein_ligand_mode=protein_ligand_mode,
        **kwargs
    )