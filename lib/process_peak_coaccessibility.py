"""
Peak Coaccessibility Analysis using Cosine Similarity

This module provides functionality to analyze peak coaccessibility using cosine similarity
with parallel processing for efficiency.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
from typing import Tuple, List, Union
import anndata as ad


def find_local_connections(X_chr: np.ndarray, peaks_chr: pd.DataFrame, 
                          threshold: float, window_ext: int) -> List[Tuple[str, str, float]]:
    """
    Find local connections between peaks using cosine similarity.
    
    Args:
        X_chr: ATAC-seq data matrix for a chromosome (peaks x cells)
        peaks_chr: DataFrame with peak information (must have 'start', 'end' columns)
        threshold: Minimum cosine similarity threshold for connections
        window_ext: Window extension size for local analysis
        
    Returns:
        List of tuples containing (peak1, peak2, similarity_score)
    """
    starts = peaks_chr["start"].to_numpy()
    ends = peaks_chr["end"].to_numpy()
    names = peaks_chr.index.to_numpy()
    out = []

    for i in range(len(peaks_chr)):
        # find local window
        left = np.searchsorted(starts, starts[i] - window_ext, side='left')
        right = np.searchsorted(starts, ends[i] + window_ext, side='right')

        # compute similarities only in that slice
        if right - left <= 1: 
            continue
        sim_block = cosine_similarity(X_chr[i:i+1], X_chr[left:right]).ravel()

        for j, s in zip(range(left, right), sim_block):
            if j <= i: 
                continue
            # check overlap and threshold
            if (starts[i] - window_ext < ends[j]) and (starts[j] - window_ext < ends[i]):
                if abs(s) >= threshold:
                    out.append((names[i], names[j], s))
    return out


def compute_peak_coaccessibility(atac_data: ad.Anndata, 
                                threshold: float = 0.1, 
                                window_ext: int = 250_000,
                                n_jobs: int = -1,
                                verbose: int = 5) -> pd.DataFrame:
    """
    Compute peak coaccessibility using cosine similarity with parallel processing.
    
    Args:
        atac_data: AnnData object containing ATAC-seq data with peak annotations
        threshold: Minimum cosine similarity threshold for connections (default: 0.1)
        window_ext: Window extension size for local analysis in base pairs (default: 250,000)
        n_jobs: Number of parallel jobs (-1 for all cores, default: -1)
        verbose: Verbosity level for parallel processing (default: 5)
        
    Returns:
        DataFrame with columns ['Peak1', 'Peak2', 'coaccess'] containing peak pairs
        and their coaccessibility scores
        
    Example:
        >>> import scanpy as sc
        >>> adata = sc.read_h5ad('atac_data.h5ad')
        >>> coaccess_df = compute_peak_coaccessibility(adata, threshold=0.2, window_ext=500000)
    """
    # Process each chromosome in parallel
    connections = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(find_local_connections)(
            atac_data[:, atac_data.var["chr"] == chrom].X.T,
            atac_data.var[atac_data.var["chr"] == chrom],
            threshold,
            window_ext
        )
        for chrom in atac_data.var["chr"].unique()
    )
    
    # Flatten results and create DataFrame
    df = pd.DataFrame([x for sub in connections for x in sub],
                      columns=["Peak1", "Peak2", "coaccess"])
    
    return df


# Example usage (commented out to avoid execution)
# if __name__ == "__main__":
#     # Assuming 'atac' is your AnnData object
#     coaccess_df = compute_peak_coaccessibility(
#         atac_data=atac,
#         threshold=0.1,
#         window_ext=250_000,
#         n_jobs=-1,
#         verbose=5
#     )
#     print(coaccess_df)