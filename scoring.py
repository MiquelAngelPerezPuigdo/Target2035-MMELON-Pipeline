"""
DREAM x CACHE Target 2035 Drug Discovery Challenge
Scoring Functions Mapping and Target Engineering Module.

This module contains flexible, customizable formulations to map DNA-Encoded Library (DEL)
selection experimental metrics under multiple conditions (Target, Inhibitor, No-Target Control)
into a unified scoring representation (0 to 1) for MMELON model training.

NOTE: This scoring function script is the PRIMARY arena for experimentation and target engineering. 
Adjusting thresholds, parameters, and formula weights here is where the "play" needs to be done 
to find the optimal signal-to-noise ratio for training.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray | pd.Series, temperature: float = 1.0, bias: float = 0.0) -> np.ndarray:
    """Standard parameterized Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-temperature * (x - bias)))


def score_tier2_soft_sigmoid(
    df: pd.DataFrame,
    z_pgk2_col: str = "zscore_PGK2",
    z_inh_col: str = "zscore_PGK2_with_inhibitor",
    z_ntc_col: str = "zscore_NTC",
    temperature: float = 1.5,
    bias: float = 1.5,
) -> np.ndarray:
    """
    Tier 2: Specificity Difference Score (Continuous Soft-Labeling) - STRONGLY RECOMMENDED.
    
    Computes a raw competitive difference score:
      S = Z_PGK2 - max(Z_NTC, Z_inh)
    And maps it to a continuous 0-1 range using a parameterized Sigmoid.
    
    TIP: A target score threshold of 0.5 (under standard temperature=1.5 and bias=1.5)
    provides an mathematically optimal midpoint boundary to separate pocket-specific 
    binders from non-specific ones.
    """
    z_pgk2 = df[z_pgk2_col] if z_pgk2_col in df.columns else 0.0
    z_inh = df[z_inh_col] if z_inh_col in df.columns else 0.0
    z_ntc = df[z_ntc_col] if z_ntc_col in df.columns else 0.0

    raw_diff = z_pgk2 - np.maximum(z_ntc, z_inh)
    return sigmoid(raw_diff, temperature=temperature, bias=bias).to_numpy()


def score_bcm_recommended(
    df: pd.DataFrame,
    count_pgk2_col: str = "count_PGK2",
    count_inh_col: str = "count_PGK2_with_inhibitor",
    count_ntc_col: str = "count_NTC",
    historic_hits_col: str = "historic_hits",
) -> np.ndarray:
    """
    BCM's Recommended Orthosteric Binders Criteria.
    
    Filters compounds based on raw read count thresholds:
      - count_PGK2 >= 3
      - count_PGK2_with_inhibitor < 0.1 * count_PGK2
      - count_NTC == 0
      - historic_hits < 5
      
    Returns a binary numpy array (1.0 for actives, 0.0 for inactives).
    """
    count_pgk2 = df[count_pgk2_col] if count_pgk2_col in df.columns else 0
    count_inh = df[count_inh_col] if count_inh_col in df.columns else 0
    count_ntc = df[count_ntc_col] if count_ntc_col in df.columns else 0
    historic_hits = df[historic_hits_col] if historic_hits_col in df.columns else 0

    # Apply boolean conditions
    cond1 = count_pgk2 >= 3
    cond2 = count_inh < 0.1 * count_pgk2
    cond3 = count_ntc == 0
    cond4 = historic_hits < 5

    # Compute binary intersection
    bcm_active = cond1 & cond2 & cond3 & cond4
    return bcm_active.astype(float).to_numpy()


def score_bcm_hybrid(
    df: pd.DataFrame,
    z_pgk2_col: str = "zscore_PGK2",
    z_inh_col: str = "zscore_PGK2_with_inhibitor",
    z_ntc_col: str = "zscore_NTC",
    temperature: float = 1.5,
    bias: float = 1.5,
    count_pgk2_col: str = "count_PGK2",
    count_inh_col: str = "count_PGK2_with_inhibitor",
    count_ntc_col: str = "count_NTC",
    historic_hits_col: str = "historic_hits",
) -> np.ndarray:
    """
    Hybrid BCM-Sigmoid Score.
    
    Computes the continuous soft-sigmoid specificity score:
      S = Z_PGK2 - max(Z_NTC, Z_inh)
      Score_continuous = sigmoid(S)
      
    And multiplies/masks it with BCM's recommended binary selection criteria:
      - count_PGK2 >= 3
      - count_PGK2_with_inhibitor < 0.1 * count_PGK2
      - count_NTC == 0
      - historic_hits < 5
      
    This eliminates noisy background records while retaining continuous relative 
    affinity scores for high-confidence pocket binders!
    """
    continuous_score = score_tier2_soft_sigmoid(
        df,
        z_pgk2_col=z_pgk2_col,
        z_inh_col=z_inh_col,
        z_ntc_col=z_ntc_col,
        temperature=temperature,
        bias=bias,
    )
    bcm_mask = score_bcm_recommended(
        df,
        count_pgk2_col=count_pgk2_col,
        count_inh_col=count_inh_col,
        count_ntc_col=count_ntc_col,
        historic_hits_col=historic_hits_col,
    )
    return continuous_score * bcm_mask


def map_scores(
    df: pd.DataFrame,
    scheme: str = "tier2",
    **kwargs,
) -> np.ndarray:
    """
    Modular entry point to map raw experimental values to a 0-1 target value.
    
    This is where model performance is decided. Experimenting with different sigmoid 
    temperatures/biases or adding weights to controls is highly encouraged!
    """
    scheme_clean = scheme.lower().replace("_", "")
    if scheme_clean in ("tier2", "softsigmoid"):
        return score_tier2_soft_sigmoid(df, **kwargs)
    elif scheme_clean in ("bcm", "bcmbinary", "bcmrecommended", "bcm_binary"):
        return score_bcm_recommended(df, **kwargs)
    elif scheme_clean in ("hybrid", "bcmhybrid", "hybridsigmoid", "bcm_hybrid"):
        return score_bcm_hybrid(df, **kwargs)
    else:
        raise ValueError(
            f"Unknown scoring scheme: '{scheme}'. "
            f"Supported schemes: 'tier2', 'bcm_binary', 'bcm_hybrid'."
        )

