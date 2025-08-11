#!/bin/bash



boltz predict affinity_prot_prot.yaml --model boltz2_ensemble --atomic_affinity --use_msa_server --cache=/shared/.boltz --no_kernels --override
