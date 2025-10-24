"""
Pytest configuration and fixtures for CellOracle tests

This module provides shared fixtures and configuration for all test modules.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
import yaml
import scanpy as sc
from pathlib import Path
from unittest.mock import Mock, MagicMock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration dictionary"""
    return {
        'rna_h5ad': os.path.join(temp_dir, 'test_rna.h5ad'),
        'peak_names_file': os.path.join(temp_dir, 'peaks.txt'),
        'peak_coaccess_path': os.path.join(temp_dir, 'coaccess.tsv'),
        'TG2TF_json_path': os.path.join(temp_dir, 'tg2tf.json'),
        'output_dir': os.path.join(temp_dir, 'output'),
        'run_name': 'test_run',
        'genome_dir': os.path.join(temp_dir, 'genomes'),
        'reference_dir': 'hg38',
        'cluster_column': 'leiden',
        'embedding': 'X_umap',
        'raw_counts': True,
        'tf_binding_frp': 0.02,
        'motif_filtering_method': 'cumulative_score',
        'motif_threshold': 10,
        'TF_evidence_direct': False,
        'grn_edge_p_threshold': 0.001,
        'verbose': False
    }


@pytest.fixture
def test_rna_adata(temp_dir):
    """Create test RNA AnnData object"""
    # Create synthetic RNA data
    n_cells, n_genes = 100, 1000
    X = np.random.poisson(5, (n_cells, n_genes))
    
    adata = sc.AnnData(X=X)
    adata.obs['leiden'] = np.random.choice(['0', '1', '2'], n_cells)
    adata.obsm['X_umap'] = np.random.randn(n_cells, 2)
    adata.layers['raw_count'] = X.copy()
    
    # Add gene names
    adata.var_names = [f'GENE_{i}' for i in range(n_genes)]
    adata.obs_names = [f'CELL_{i}' for i in range(n_cells)]
    
    return adata


@pytest.fixture
def test_peak_data(temp_dir):
    """Create test peak data files"""
    # Peak names file
    peak_names = [
        "chr1_1000_2000",
        "chr2_3000_4000", 
        "chr3_5000_6000",
        "chr4_7000_8000"
    ]
    
    peak_file = os.path.join(temp_dir, 'peaks.txt')
    with open(peak_file, 'w') as f:
        for peak in peak_names:
            f.write(f"{peak}\n")
    
    # Coaccessibility data
    coaccess_df = pd.DataFrame({
        'Peak1': ['chr1_1000_2000', 'chr2_3000_4000', 'chr3_5000_6000'],
        'Peak2': ['chr2_3000_4000', 'chr3_5000_6000', 'chr4_7000_8000'],
        'coaccess': [0.8, 0.9, 0.7]
    })
    
    coaccess_file = os.path.join(temp_dir, 'coaccess.tsv')
    coaccess_df.to_csv(coaccess_file, sep='\t', index=False)
    
    return {
        'peak_names': peak_names,
        'peak_file': peak_file,
        'coaccess_df': coaccess_df,
        'coaccess_file': coaccess_file
    }


@pytest.fixture
def test_tg2tf_mapping(temp_dir):
    """Create test TG2TF mapping"""
    tg2tf_dict = {
        'GENE_1': ['TF1', 'TF2'],
        'GENE_2': ['TF3', 'TF4'],
        'GENE_3': ['TF1', 'TF5'],
        'GENE_4': ['TF2', 'TF6']
    }
    
    tg2tf_file = os.path.join(temp_dir, 'tg2tf.json')
    with open(tg2tf_file, 'w') as f:
        json.dump(tg2tf_dict, f)
    
    return {
        'mapping': tg2tf_dict,
        'file': tg2tf_file
    }


@pytest.fixture
def test_tss_data():
    """Create test TSS annotation data"""
    return pd.DataFrame({
        'chr': ['chr1', 'chr2', 'chr3', 'chr4'],
        'start': [1000, 3000, 5000, 7000],
        'end': [2000, 4000, 6000, 8000],
        'gene_short_name': ['GENE_1', 'GENE_2', 'GENE_3', 'GENE_4'],
        'strand': ['+', '-', '+', '-']
    })


@pytest.fixture
def test_integrated_peaks():
    """Create test integrated peak-gene data"""
    return pd.DataFrame({
        'peak_id': ['chr1_1000_2000', 'chr2_3000_4000', 'chr3_5000_6000'],
        'gene_short_name': ['GENE_1', 'GENE_2', 'GENE_3'],
        'coaccess': [0.8, 0.9, 0.7]
    })


@pytest.fixture
def test_base_grn():
    """Create test base GRN data"""
    return pd.DataFrame({
        'peak_id': ['chr1_1000_2000', 'chr2_3000_4000', 'chr3_5000_6000'],
        'gene_short_name': ['GENE_1', 'GENE_2', 'GENE_3'],
        'TF1': [1, 0, 1],
        'TF2': [0, 1, 0],
        'TF3': [1, 1, 0],
        'TF4': [0, 0, 1]
    })


@pytest.fixture
def test_network_data():
    """Create test network data"""
    return {
        '0': pd.DataFrame({
            'source': ['TF1', 'TF2', 'TF3'],
            'target': ['GENE_1', 'GENE_2', 'GENE_3'],
            'coef_mean': [0.5, -0.3, 0.8],
            'coef_abs': [0.5, 0.3, 0.8],
            'p': [0.001, 0.01, 0.0001],
            '-logp': [3.0, 2.0, 4.0]
        }),
        '1': pd.DataFrame({
            'source': ['TF2', 'TF4'],
            'target': ['GENE_1', 'GENE_2'],
            'coef_mean': [0.2, -0.6],
            'coef_abs': [0.2, 0.6],
            'p': [0.05, 0.001],
            '-logp': [1.3, 3.0]
        })
    }


@pytest.fixture
def mock_celloracle_pipeline(temp_dir):
    """Create a mocked CellOraclePipeline instance"""
    from lib.process_celloracle import CellOraclePipeline
    
    pipeline = CellOraclePipeline(
        genome_dir=os.path.join(temp_dir, 'genomes'),
        output_dir=os.path.join(temp_dir, 'output'),
        n_cpu=2,
        ref_genome='hg38',
        verbose=False
    )
    
    return pipeline


@pytest.fixture
def mock_oracle():
    """Create a mocked Oracle object"""
    oracle = Mock()
    oracle.adata = Mock()
    oracle.adata.shape = (100, 1000)
    oracle.pca = Mock()
    oracle.pca.explained_variance_ratio_ = np.array([0.3, 0.2, 0.1, 0.05, 0.02, 0.01])
    return oracle


@pytest.fixture
def mock_tfinfo():
    """Create a mocked TFinfo object"""
    tfinfo = Mock()
    tfinfo.to_dataframe.return_value = pd.DataFrame({
        'TF1': [1, 0], 'TF2': [0, 1]
    })
    return tfinfo


@pytest.fixture
def mock_network():
    """Create a mocked network object"""
    network = Mock()
    network.links_dict = {
        '0': pd.DataFrame({
            'source': ['TF1'], 'target': ['GENE_1'],
            'coef_mean': [0.5], 'coef_abs': [0.5],
            'p': [0.001], '-logp': [3.0]
        })
    }
    return network


@pytest.fixture
def test_config_file(temp_dir, test_config):
    """Create a test configuration file"""
    config_file = os.path.join(temp_dir, 'test_config.yaml')
    with open(config_file, 'w') as f:
        yaml.dump(test_config, f)
    return config_file


@pytest.fixture
def sample_motif_data():
    """Create sample motif data for testing"""
    return {
        'motifs': ['MOTIF1', 'MOTIF2', 'MOTIF3'],
        'scores': [15.2, 8.7, 12.1],
        'threshold': 10.0
    }


@pytest.fixture
def sample_simulation_data():
    """Create sample simulation data"""
    return {
        'conditions': {'GENE_1': 0.0, 'GENE_2': 0.5},
        'n_propagation': 3,
        'n_neighbors': 200,
        'sampled_fraction': 0.3
    }


@pytest.fixture
def sample_plot_data():
    """Create sample plotting data"""
    return {
        'cluster_names': ['0', '1', '2'],
        'graph_stats': [
            'degree_centrality_all',
            'degree_centrality_in', 
            'degree_centrality_out',
            'betweenness_centrality',
            'eigenvector_centrality'
        ],
        'top_n_genes': 30,
        'percentile': 99
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names"""
    for item in items:
        if "test_celloracle_pipeline" in item.name:
            item.add_marker(pytest.mark.unit)
        elif "test_run_celloracle" in item.name:
            item.add_marker(pytest.mark.integration)
        elif "test_integration" in item.name:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)
