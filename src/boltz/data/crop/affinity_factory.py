"""Factory for selecting appropriate affinity cropper based on interaction type."""

from typing import Optional

from boltz.data import const
from boltz.data.crop.affinity import AffinityCropper
from boltz.data.crop.affinity_protein import ProteinProteinAffinityCropper
from boltz.data.crop.cropper import Cropper
from boltz.data.types import Tokenized


def create_affinity_cropper(
    data: Tokenized,
    neighborhood_size: int = 10,
    max_tokens_protein: int = 200,
    interface_cutoff: float = 8.0,
    min_interface_residues: int = 5,
    force_protein_protein: bool = False,
) -> Cropper:
    """Create the appropriate affinity cropper based on interaction type.
    
    Parameters
    ----------
    data : Tokenized
        The tokenized data to analyze for interaction type.
    neighborhood_size : int
        Size of sequence neighborhood around interface/binding residues.
    max_tokens_protein : int
        Maximum tokens per protein chain.
    interface_cutoff : float
        Distance cutoff (Angstroms) for defining interface residues.
    min_interface_residues : int
        Minimum number of interface residues for protein-protein cropper.
    force_protein_protein : bool
        Force use of protein-protein cropper regardless of detection.
        
    Returns
    -------
    Cropper
        The appropriate cropper instance.
    """
    # Detect interaction type
    valid_tokens = data.tokens[data.tokens["resolved_mask"]]
    binder_tokens = valid_tokens[valid_tokens["affinity_mask"]]
    
    # Check if binder is protein or ligand
    is_protein_protein = (
        force_protein_protein or 
        (binder_tokens.size > 0 and 
         all(binder_tokens["mol_type"] == const.chain_type_ids["PROTEIN"]))
    )
    
    if is_protein_protein:
        return ProteinProteinAffinityCropper(
            neighborhood_size=neighborhood_size,
            max_tokens_protein=max_tokens_protein,
            interface_cutoff=interface_cutoff,
            min_interface_residues=min_interface_residues,
        )
    else:
        return AffinityCropper(
            neighborhood_size=neighborhood_size,
            max_tokens_protein=max_tokens_protein,
        )


def detect_interaction_type(data: Tokenized) -> str:
    """Detect the type of affinity interaction in the data.
    
    Parameters
    ----------
    data : Tokenized
        The tokenized data to analyze.
        
    Returns
    -------
    str
        Either "protein-protein" or "protein-ligand".
    """
    valid_tokens = data.tokens[data.tokens["resolved_mask"]]
    binder_tokens = valid_tokens[valid_tokens["affinity_mask"]]
    
    if binder_tokens.size == 0:
        return "unknown"
    
    if all(binder_tokens["mol_type"] == const.chain_type_ids["PROTEIN"]):
        return "protein-protein"
    elif any(binder_tokens["mol_type"] == const.chain_type_ids["NONPOLYMER"]):
        return "protein-ligand"
    else:
        return "unknown" 