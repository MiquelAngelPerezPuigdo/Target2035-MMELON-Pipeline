"""
DREAM x CACHE Target 2035 Drug Discovery Challenge
Scoring Functions Mapping and Target Engineering Module.

This module contains flexible, customizable formulations to map DNA-Encoded Library (DEL)
selection experimental metrics under multiple conditions (Target, Inhibitor, No-Target Control)
into a unified scoring representation (0 to 1) for MAMMAL model training.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray | pd.Series, temperature: float = 1.0, bias: float = 0.0) -> np.ndarray:
    """Standard parameterized Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-temperature * (x - bias)))


def score_tier1_binary_hard(
    df: pd.DataFrame,
    z_pgk2_col: str = "zscore_PGK2",
    z_inh_col: str = "zscore_PGK2_with_inhibitor",
    z_ntc_col: str = "zscore_NTC",
    pgk2_threshold: float = 2.0,
    inh_threshold: float = 1.0,
    ntc_threshold: float = 1.0,
) -> np.ndarray:
    """
    Tier 1: Non-Innovative Binary Hard Thresholding.
    
    A compound is labeled as 1 (Active) if it satisfies:
      - PGK2 enrichment is high (above pgk2_threshold)
      - NTC background is low (below ntc_threshold)
      - Inhibitor competitive enrichment is low (below inh_threshold)
    Otherwise, labeled as 0.
    """
    z_pgk2 = df[z_pgk2_col] if z_pgk2_col in df.columns else 0.0
    z_inh = df[z_inh_col] if z_inh_col in df.columns else 0.0
    z_ntc = df[z_ntc_col] if z_ntc_col in df.columns else 0.0

    active_mask = (z_pgk2 >= pgk2_threshold) & (z_ntc <= ntc_threshold) & (z_inh <= inh_threshold)
    return active_mask.astype(float).to_numpy()


def score_tier2_soft_sigmoid(
    df: pd.DataFrame,
    z_pgk2_col: str = "zscore_PGK2",
    z_inh_col: str = "zscore_PGK2_with_inhibitor",
    z_ntc_col: str = "zscore_NTC",
    temperature: float = 1.5,
    bias: float = 1.5,
) -> np.ndarray:
    """
    Tier 2: Specificity Difference Score (Continuous Soft-Labeling).
    
    Computes a raw competitive difference score:
      S = Z_PGK2 - max(Z_NTC, Z_inh)
    And maps it to a continuous 0-1 range using a parameterized Sigmoid.
    """
    z_pgk2 = df[z_pgk2_col] if z_pgk2_col in df.columns else 0.0
    z_inh = df[z_inh_col] if z_inh_col in df.columns else 0.0
    z_ntc = df[z_ntc_col] if z_ntc_col in df.columns else 0.0

    raw_diff = z_pgk2 - np.maximum(z_ntc, z_inh)
    return sigmoid(raw_diff, temperature=temperature, bias=bias).to_numpy()


def score_tier3_bayesian_reads(
    df: pd.DataFrame,
    count_pgk2_col: str = "count_PGK2",
    count_inh_col: str = "count_PGK2_with_inhibitor",
    count_ntc_col: str = "count_NTC",
    temperature: float = 1.0,
    bias: float = 1.0,
) -> np.ndarray:
    """
    Tier 3: Bayesian Read-Count Log-Ratio Scoring.
    
    Uses raw read count enrichment ratios as soft training probabilities, 
    accounting for low-read count noise by adding pseudo-counts:
      ratio = log2((count_PGK2 + 1) / (max(count_NTC, count_inh) + 1))
    And squashes into [0, 1] using a Sigmoid function.
    """
    c_pgk2 = df[count_pgk2_col] if count_pgk2_col in df.columns else 0.0
    c_inh = df[count_inh_col] if count_inh_col in df.columns else 0.0
    c_ntc = df[count_ntc_col] if count_ntc_col in df.columns else 0.0

    max_control = np.maximum(c_ntc, c_inh)
    log_ratio = np.log2((c_pgk2 + 1.0) / (max_control + 1.0))
    return sigmoid(log_ratio, temperature=temperature, bias=bias).to_numpy()


def map_scores(
    df: pd.DataFrame,
    scheme: str = "tier2",
    **kwargs,
) -> np.ndarray:
    """
    Modular entry point to map raw experimental values to a 0-1 target value.
    
    Supported schemes:
      - 'tier1' / 'binary_hard': Tier 1 Binary Hard Thresholding.
      - 'tier2' / 'soft_sigmoid': Tier 2 Specificity Difference Score (Sigmoid).
      - 'tier3' / 'bayesian_reads': Tier 3 Read-Count Log-Ratio (Sigmoid).
    """
    scheme_clean = scheme.lower().replace("_", "")
    if scheme_clean in ("tier1", "binaryhard"):
        return score_tier1_binary_hard(df, **kwargs)
    elif scheme_clean in ("tier2", "softsigmoid"):
        return score_tier2_soft_sigmoid(df, **kwargs)
    elif scheme_clean in ("tier3", "bayesianreads"):
        return score_tier3_bayesian_reads(df, **kwargs)
    else:
        raise ValueError(f"Unknown scoring scheme: '{scheme}'. Choose from 'tier1', 'tier2', or 'tier3'.")
