from ptn.dists.cp_moe import CP_MOE
from ptn.dists.cp_sigma_lsf import CP_SIGMA_LSF
from ptn.dists.mps_bm_dmrg import MPS_BM_DMRG
from ptn.dists.mps_bm_lsf import MPS_BM_LSF
from ptn.dists.mps_sigma_lsf import MPS_SIGMA_LSF
from ._abc import AbstractDisributionHeadConfig, AbstractDisributionHeadOutput

dists = {
    # ----------------------------------
    # CP Dists
    # ----------------------------------
    "cp_lsf_sigma": CP_SIGMA_LSF,  # Non-Negative Canonical Polyadic w/ LSF
    "cp_moe": CP_MOE,  # Mixture of Experts
    # ----------------------------------
    # MPS Dists
    # ----------------------------------
    "mps_lsf_sigma": MPS_SIGMA_LSF,  # Non-Negative MPS w/ LSF
    "mps_lsf_bm": MPS_BM_LSF,  # Born Machine w/ LSF
    "mps_dmrg_bm": MPS_BM_DMRG,  # Born Machine w/ DMRG)
}
