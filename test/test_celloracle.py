"""
Unit and Functional Tests for CellOracle Pipeline

This module contains comprehensive tests for the CellOracle GRN inference pipeline,
including unit tests for individual methods and functional tests for end-to-end workflows.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
import yaml
from unittest.mock import Mock, patch, MagicMock
import scanpy as sc
from pathlib import Path

# Import the modules to test
from lib.process_celloracle import CellOraclePipeline, NotInitializedError
from run_celloracle_inference import execute
from lib.utils import makedir, save_yaml, load_config, load_json, get_peak_names_from_file


class TestCellOraclePipeline:
    """Unit tests for CellOraclePipeline class"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.temp_dir = tempfile.mkdtemp()
        self.genome_dir = os.path.join(self.temp_dir, "genomes")
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.genome_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create test pipeline instance
        self.pipeline = CellOraclePipeline(
            genome_dir=self.genome_dir,
            output_dir=self.output_dir,
            n_cpu=2,
            ref_genome="hg38",
            verbose=False
        )
    
    def teardown_method(self):
        """Clean up after each test method"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test CellOraclePipeline initialization"""
        assert self.pipeline.genome_dir == self.genome_dir
        assert self.pipeline.output_dir == self.output_dir
        assert self.pipeline.n_cpu == 2
        assert self.pipeline.ref_genome == "hg38"
        assert self.pipeline.verbose == False
        assert self.pipeline.oracle is None
        assert self.pipeline.base_grn is None
        assert self.pipeline.network is None
        assert self.pipeline.tf_info is None
    
    def test_check_genome_installation(self):
        """Test genome installation check"""
        with patch('celloracle.motif_analysis.is_genome_installed') as mock_check:
            mock_check.return_value = True
            result = self.pipeline.check_genome_installation()
            assert result == True
            mock_check.assert_called_once_with(
                ref_genome="hg38",
                genomes_dir=self.genome_dir
            )
    
    def test_install_genome_success(self):
        """Test successful genome installation"""
        with patch('genomepy.install_genome') as mock_install:
            result = self.pipeline.install_genome()
            assert result == True
            mock_install.assert_called_once_with(
                name="hg38",
                provider="UCSC",
                genomes_dir=self.genome_dir
            )
    
    def test_install_genome_failure(self):
        """Test genome installation failure"""
        with patch('genomepy.install_genome') as mock_install:
            mock_install.side_effect = Exception("Installation failed")
            result = self.pipeline.install_genome()
            assert result == False
    
    def test_initialize_celloracle(self):
        """Test oracle initialization"""
        with patch('celloracle.Oracle') as mock_oracle_class:
            mock_oracle = Mock()
            mock_oracle_class.return_value = mock_oracle
            self.pipeline.initialize_celloracle()
            assert self.pipeline.oracle == mock_oracle
    
    def test_select_PCA_components(self):
        """Test PCA component selection"""
        # Mock oracle with PCA attributes
        mock_oracle = Mock()
        # Use a variance ratio array that will produce a valid result
        mock_oracle.pca.explained_variance_ratio_ = np.array([0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01])
        self.pipeline.oracle = mock_oracle
        
        n_comps = self.pipeline.select_PCA_components()
        assert isinstance(n_comps, int)
        assert n_comps > 0
        
        # Test edge case where no components meet threshold
        mock_oracle.pca.explained_variance_ratio_ = np.array([0.001, 0.001, 0.001, 0.001])
        n_comps_default = self.pipeline.select_PCA_components()
        assert isinstance(n_comps_default, int)
        assert n_comps_default > 0
    
    def test_impute(self):
        """Test KNN imputation"""
        mock_oracle = Mock()
        self.pipeline.oracle = mock_oracle
        
        self.pipeline.impute(n_comps=10, k=50, balanced=True)
        mock_oracle.knn_imputation.assert_called_once_with(
            n_pca_dims=10,
            k=50,
            balanced=True,
            b_sight=400,  # k*8
            b_maxl=200,   # k*4
            n_jobs=2
        )
    
    def test_get_tss_annotations(self):
        """Test TSS annotation retrieval"""
        peak_names = ["chr1_1000_2000", "chr2_3000_4000"]
        
        with patch('celloracle.motif_analysis.get_tss_info') as mock_get_tss:
            mock_df = pd.DataFrame({
                'chr': ['chr1', 'chr2'],
                'start': [1000, 3000],
                'end': [2000, 4000],
                'gene_short_name': ['GENE1', 'GENE2'],
                'strand': ['+', '-']
            })
            mock_get_tss.return_value = mock_df
            
            result = self.pipeline.get_tss_annotations(peak_names)
            mock_get_tss.assert_called_once()
            pd.testing.assert_frame_equal(result, mock_df)
    
    def test_integrate_tss_coaccessible_peaks(self):
        """Test TSS and coaccessibility integration"""
        tss_df = pd.DataFrame({
            'chr': ['chr1', 'chr1'],
            'start': [1000, 2000],
            'end': [1500, 2500],
            'gene_short_name': ['GENE1', 'GENE2'],
            'strand': ['+', '-']
        })
        
        coaccess_df = pd.DataFrame({
            'Peak1': ['chr1_1000_1500', 'chr1_2000_2500'],
            'Peak2': ['chr1_2000_2500', 'chr1_1000_1500'],
            'coaccess': [0.8, 0.9]
        })
        
        with patch('celloracle.motif_analysis.integrate_tss_peak_with_cicero') as mock_integrate:
            expected_result = pd.DataFrame({
                'peak_id': ['chr1_1000_1500', 'chr1_2000_2500'],
                'gene_short_name': ['GENE1', 'GENE2'],
                'coaccess': [0.8, 0.9]
            })
            mock_integrate.return_value = expected_result
            
            result = CellOraclePipeline.integrate_tss_coaccessible_peaks(tss_df, coaccess_df)
            mock_integrate.assert_called_once_with(
                tss_peak=tss_df,
                cicero_connections=coaccess_df
            )
            pd.testing.assert_frame_equal(result, expected_result)
    
    def test_get_peak2gene_conns(self):
        """Test peak-to-gene connection filtering"""
        tss_integrated_df = pd.DataFrame({
            'peak_id': ['peak1', 'peak2', 'peak3'],
            'gene_short_name': ['GENE1', 'GENE2', 'GENE3'],
            'coaccess': [0.9, 0.7, 0.5]  # Only first one should pass threshold
        })
        
        with patch('celloracle.motif_analysis.check_peak_format') as mock_check:
            # Only return the first row since only peak1 has coaccess >= 0.8
            mock_check.return_value = tss_integrated_df.iloc[:1]
            
            result = self.pipeline.get_peak2gene_conns(tss_integrated_df, coaccess_threshold=0.8)
            
            # Should filter to only peaks with coaccess >= 0.8
            assert len(result) == 1
            assert result.iloc[0]['peak_id'] == 'peak1'
    
    def test_initialize_TFinfo(self):
        """Test TFinfo initialization"""
        peak_df = pd.DataFrame({
            'peak_id': ['peak1', 'peak2'],
            'gene_short_name': ['GENE1', 'GENE2']
        })
        
        with patch('celloracle.motif_analysis.TFinfo') as mock_tfinfo_class:
            mock_tfinfo = Mock()
            mock_tfinfo_class.return_value = mock_tfinfo
            
            self.pipeline.initialize_TFinfo(peak_df)
            
            mock_tfinfo_class.assert_called_once_with(
                peak_data_frame=peak_df,
                ref_genome="hg38",
                genomes_dir=self.genome_dir
            )
            assert self.pipeline.tf_info == mock_tfinfo
    
    def test_scan_motifs_not_initialized(self):
        """Test scan_motifs raises error when TFinfo not initialized"""
        with pytest.raises(NotInitializedError):
            self.pipeline.scan_motifs()
    
    def test_scan_motifs_success(self):
        """Test successful motif scanning"""
        mock_tfinfo = Mock()
        self.pipeline.tf_info = mock_tfinfo
        
        self.pipeline.scan_motifs(fpr=0.02, motifs=None, TF_evidence_direct_only=False)
        
        mock_tfinfo.scan.assert_called_once_with(
            background_length=200,
            fpr=0.02,
            n_cpus=2,
            verbose=False,
            motifs=None,
            TF_evidence_level="direct_and_indirect",
            TF_formatting="auto",
            batch_size=None,
            divide=100000
        )
        mock_tfinfo.to_hdf5.assert_called_once()
    
    def test_filter_motifs_not_initialized(self):
        """Test filter_motifs raises error when TFinfo not initialized"""
        with pytest.raises(ValueError):
            self.pipeline.filter_motifs()
    
    def test_filter_motifs_success(self):
        """Test successful motif filtering"""
        mock_tfinfo = Mock()
        self.pipeline.tf_info = mock_tfinfo
        
        self.pipeline.filter_motifs(threshold=10, method="cumulative_score")
        
        mock_tfinfo.reset_filtering.assert_called_once()
        mock_tfinfo.filter_motifs_by_score.assert_called_once_with(
            threshold=10,
            method="cumulative_score"
        )
    
    def test_get_base_grn_not_initialized(self):
        """Test get_base_grn raises error when TFinfo not initialized"""
        with pytest.raises(NotInitializedError):
            self.pipeline.get_base_grn()
    
    def test_get_base_grn_success(self):
        """Test successful base GRN generation"""
        mock_tfinfo = Mock()
        mock_grn_df = pd.DataFrame({'TF1': [1, 0], 'TF2': [0, 1]})
        mock_tfinfo.to_dataframe.return_value = mock_grn_df
        self.pipeline.tf_info = mock_tfinfo
        
        result = self.pipeline.get_base_grn()
        
        mock_tfinfo.make_TFinfo_dataframe_and_dictionary.assert_called_once_with(verbose=False)
        mock_tfinfo.to_dataframe.assert_called_once()
        pd.testing.assert_frame_equal(result, mock_grn_df)
        assert self.pipeline.base_grn is not None
    
    def test_get_cluster_networks(self):
        """Test cluster network generation"""
        mock_oracle = Mock()
        mock_network = Mock()
        mock_oracle.get_links.return_value = mock_network
        self.pipeline.oracle = mock_oracle
        
        result = self.pipeline.get_cluster_networks("leiden")
        
        mock_oracle.get_links.assert_called_once_with(
            cluster_name_for_GRN_unit="leiden",
            alpha=10,
            verbose_level=0
        )
        assert self.pipeline.network == mock_network
        assert result == mock_network
    
    def test_save_network(self):
        """Test network saving"""
        mock_network = Mock()
        self.pipeline.network = mock_network
        
        self.pipeline.save_network("leiden")
        mock_network.to_hdf5.assert_called_once_with(
            file_path=f"{self.output_dir}/fleiden.celloracle.links"
        )
    
    def test_filter_egdes(self):
        """Test edge filtering"""
        mock_network = Mock()
        self.pipeline.network = mock_network
        
        self.pipeline.filter_egdes(p_value=0.001, weight="weight")
        mock_network.filter_links.assert_called_once_with(
            p=0.001,
            weight="weight",
            threshold_number=20000,
            genelist_source=None,
            genelist_target=None
        )
    
    def test_get_network_score(self):
        """Test network score calculation"""
        mock_network = Mock()
        self.pipeline.network = mock_network
        
        self.pipeline.get_network_score()
        mock_network.get_network_score.assert_called_once()
    
    def test_plot_node_degree_distributions(self):
        """Test node degree distribution plotting"""
        mock_network = Mock()
        self.pipeline.network = mock_network
        
        self.pipeline.plot_node_degree_distibutions()
        mock_network.plot_degree_distributions.assert_called_once_with(
            plot_model=False,
            save=self.output_dir
        )
    
    def test_plot_scores_rankings(self):
        """Test score ranking plots"""
        mock_network = Mock()
        self.pipeline.network = mock_network
        
        self.pipeline.plot_scores_rankings("cluster1", top_n_gene=30)
        mock_network.plot_scores_as_rank.assert_called_once_with(
            cluster="cluster1",
            n_gene=30,
            save=f"{self.output_dir}/ranked_score/cluster1"
        )
    
    def test_plot_score_comparisons(self):
        """Test score comparison plots"""
        mock_network = Mock()
        self.pipeline.network = mock_network
        
        self.pipeline.plot_score_comparisons("degree_centrality", "cluster1", "cluster2")
        mock_network.plot_score_comparison_2D.assert_called_once_with(
            value="degree_centrality",
            cluster1="cluster1",
            cluster2="cluster2",
            percentile=99,
            save=f"{self.output_dir}/score_comparison/"
        )
    
    def test_plot_score_distributions(self):
        """Test score distribution plots"""
        mock_network = Mock()
        self.pipeline.network = mock_network
        
        values = ["degree_centrality", "betweenness_centrality"]
        self.pipeline.plot_score_distributions(values, method="boxplot")
        mock_network.plot_score_discributions.assert_called_once_with(
            values=values,
            method="boxplot",
            save=self.output_dir
        )
    
    def test_plot_network_entropy_distribution(self):
        """Test network entropy distribution plotting"""
        mock_network = Mock()
        self.pipeline.network = mock_network
        
        self.pipeline.plot_network_entropy_distribution()
        mock_network.plot_network_entropy_distributions.assert_called_once_with(
            save=self.output_dir
        )
    
    def test_get_cluster_network(self):
        """Test cluster network retrieval"""
        mock_network = Mock()
        mock_links_dict = {"cluster1": pd.DataFrame({
            'source': ['TF1', 'TF2'],
            'target': ['GENE1', 'GENE2'],
            'coef_mean': [0.5, -0.3],
            'coef_abs': [0.5, 0.3],
            'p': [0.001, 0.01],
            '-logp': [3.0, 2.0]
        })}
        mock_network.links_dict = mock_links_dict
        self.pipeline.network = mock_network
        
        result = self.pipeline.get_cluster_network("cluster1")
        pd.testing.assert_frame_equal(result, mock_links_dict["cluster1"])
    
    def test_initialize_similation_model(self):
        """Test simulation model initialization"""
        mock_network = Mock()
        mock_oracle = Mock()
        self.pipeline.network = mock_network
        self.pipeline.oracle = mock_oracle
        
        self.pipeline.initialize_similation_model(use_cluster_specific_TFdict=True)
        
        mock_network.filter_links.assert_called_once()
        mock_oracle.get_cluster_specific_TFdict_from_Links.assert_called_once_with(links=mock_network)
        mock_oracle.fit_GRN_for_simulation.assert_called_once_with(
            alpha=10,
            use_cluster_specific_TFdict=True
        )
        mock_oracle.to_hdf5.assert_called_once()
    
    def test_calc_similate_shift(self):
        """Test simulation shift calculation"""
        mock_oracle = Mock()
        self.pipeline.oracle = mock_oracle
        
        conditions = {"GENE1": 0.0, "GENE2": 0.5}
        self.pipeline.calc_similate_shift(conditions, n_propagation=3)
        
        mock_oracle.simulate_shift.assert_called_once_with(
            perturb_condition=conditions,
            n_propagation=3
        )
    
    def test_transition_prob_estimation(self):
        """Test transition probability estimation"""
        mock_oracle = Mock()
        self.pipeline.oracle = mock_oracle
        
        self.pipeline.transition_prob_estimation(
            n_neighbors=200,
            knn_random=True,
            sampled_fraction=0.3,
            sampling_probs=(0.1, 0.05),
            calculate_randomized=True,
            sigma_corr=0.05
        )
        
        mock_oracle.estimate_transition_prob.assert_called_once_with(
            n_neighbors=200,
            knn_random=True,
            sampled_fraction=0.3,
            sampling_probs=(0.1, 0.05),
            calculate_randomized=True,
            n_jobs=2,
            threads=2
        )
        mock_oracle.calculate_embedding_shift.assert_called_once_with(sigma_corr=0.05)
    
    def test_calculate_p_mass(self):
        """Test probability mass calculation"""
        mock_oracle = Mock()
        self.pipeline.oracle = mock_oracle
        
        self.pipeline.calculate_p_mass(smooth=0.8, n_grid=40, n_neighbors=200)
        mock_oracle.calculate_p_mass.assert_called_once_with(
            smooth=0.8,
            n_grid=40,
            n_neighbors=200
        )
    
    def test_suggest_and_apply_mass_filter(self):
        """Test mass filter suggestion and application"""
        mock_oracle = Mock()
        self.pipeline.oracle = mock_oracle
        
        self.pipeline.suggest_and_apply_mass_filter(n_suggestion=12, min_mass=0.01)
        
        mock_oracle.suggest_mass_thresholds.assert_called_once_with(n_suggestion=12)
        mock_oracle.calculate_mass_filter.assert_called_once_with(min_mass=0.01, plot=False)
    
    def test_plot_simulation_comparison(self):
        """Test simulation comparison plotting"""
        mock_oracle = Mock()
        self.pipeline.oracle = mock_oracle
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = Mock()
            mock_ax = [Mock(), Mock()]
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            self.pipeline.plot_simulation_comparison("GENE1", scale_simulation=0.5)
            
            mock_subplots.assert_called_once_with(1, 2, figsize=(13, 6))
            mock_oracle.plot_simulation_flow_on_grid.assert_called_once_with(
                scale=0.5, ax=mock_ax[0]
            )
            mock_oracle.plot_simulation_flow_random_on_grid.assert_called_once_with(
                scale=0.5, ax=mock_ax[1]
            )
    
    def test_plot_vector_field_with_clusters(self):
        """Test vector field plotting with clusters"""
        mock_oracle = Mock()
        self.pipeline.oracle = mock_oracle
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            self.pipeline.plot_vector_field_with_clusters(scale_simulation=0.5, point_size=10)
            
            mock_subplots.assert_called_once_with(figsize=(8, 8))
            mock_oracle.plot_cluster_whole.assert_called_once_with(ax=mock_ax, s=10)
            mock_oracle.plot_simulation_flow_on_grid.assert_called_once_with(
                scale=0.5, ax=mock_ax, show_background=False
            )


class TestRunCellOracle:
    """Functional tests for run_celloracle_inference.py"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create test configuration
        self.test_config = {
            'rna_h5ad': os.path.join(self.temp_dir, 'test_rna.h5ad'),
            'peak_names_file': os.path.join(self.temp_dir, 'peaks.txt'),
            'peak_coaccess_path': os.path.join(self.temp_dir, 'coaccess.tsv'),
            'TG2TF_json_path': os.path.join(self.temp_dir, 'tg2tf.json'),
            'output_dir': self.output_dir,
            'run_name': 'test_run',
            'genome_dir': os.path.join(self.temp_dir, 'genomes'),
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
        
        # Save config to file
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def teardown_method(self):
        """Clean up after each test"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_data(self):
        """Create test data files"""
        # Create test RNA h5ad file
        adata = sc.AnnData(X=np.random.poisson(5, (100, 1000)))
        adata.obs['leiden'] = np.random.choice(['0', '1', '2'], 100)
        adata.obsm['X_umap'] = np.random.randn(100, 2)
        adata.layers['raw_count'] = adata.X.copy()
        adata.write_h5ad(self.test_config['rna_h5ad'])
        
        # Create peak names file
        with open(self.test_config['peak_names_file'], 'w') as f:
            f.write("chr1_1000_2000\nchr2_3000_4000\nchr3_5000_6000\n")
        
        # Create coaccessibility file
        coaccess_df = pd.DataFrame({
            'Peak1': ['chr1_1000_2000', 'chr2_3000_4000'],
            'Peak2': ['chr2_3000_4000', 'chr3_5000_6000'],
            'coaccess': [0.8, 0.9]
        })
        coaccess_df.to_csv(self.test_config['peak_coaccess_path'], sep='\t', index=False)
        
        # Create TG2TF mapping
        tg2tf_dict = {
            'GENE1': ['TF1', 'TF2'],
            'GENE2': ['TF3', 'TF4']
        }
        with open(self.test_config['TG2TF_json_path'], 'w') as f:
            json.dump(tg2tf_dict, f)
    
    def test_execute_success(self):
        """Test successful execution of the main pipeline"""
        self.create_test_data()
        
        # Mock the CellOraclePipeline methods
        with patch('run_celloracle_inference.CellOraclePipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            
            # Create mock args
            args = Mock()
            args.config = self.config_path
            args.n_cpu = 4
            
            # Execute the pipeline
            execute(args)
            
            # Verify pipeline was instantiated correctly
            mock_pipeline_class.assert_called_once_with(
                genome_dir=self.test_config['genome_dir'],
                ref_genome=self.test_config['reference_dir'],
                output_dir=os.path.join(self.output_dir, 'test_run'),
                verbose=False,
                n_cpu=4
            )
            
            # Verify all major methods were called
            mock_pipeline.process_base_grn_from_motif_Tfs.assert_called_once()
            mock_pipeline.process_initialize_celloracle.assert_called_once()
            mock_pipeline.process_grn_inference.assert_called_once()
    
    def test_execute_with_raw_counts_layer(self):
        """Test execution with raw counts layer"""
        self.create_test_data()
        
        with patch('run_celloracle_inference.CellOraclePipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            
            args = Mock()
            args.config = self.config_path
            args.n_cpu = 4
            
            execute(args)
            
            # Verify that raw counts layer was used
            mock_pipeline.process_initialize_celloracle.assert_called_once()
            call_args = mock_pipeline.process_initialize_celloracle.call_args
            assert call_args[1]['raw_counts'] == True
    
    def test_execute_without_raw_counts_layer(self):
        """Test execution without raw counts layer"""
        # Create adata without raw_count layer
        adata = sc.AnnData(X=np.random.poisson(5, (100, 1000)))
        adata.obs['leiden'] = np.random.choice(['0', '1', '2'], 100)
        adata.obsm['X_umap'] = np.random.randn(100, 2)
        adata.write_h5ad(self.test_config['rna_h5ad'])
        
        # Create other test files
        with open(self.test_config['peak_names_file'], 'w') as f:
            f.write("chr1_1000_2000\n")
        
        coaccess_df = pd.DataFrame({
            'Peak1': ['chr1_1000_2000'],
            'Peak2': ['chr1_1000_2000'],
            'coaccess': [1.0]
        })
        coaccess_df.to_csv(self.test_config['peak_coaccess_path'], sep='\t', index=False)
        
        with open(self.test_config['TG2TF_json_path'], 'w') as f:
            json.dump({'GENE1': ['TF1']}, f)
        
        with patch('run_celloracle_inference.CellOraclePipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            
            args = Mock()
            args.config = self.config_path
            args.n_cpu = 4
            
            execute(args)
            
            # Should still work without raw_count layer
            mock_pipeline.process_initialize_celloracle.assert_called_once()


class TestUtils:
    """Unit tests for utility functions"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up after each test"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_makedir(self):
        """Test directory creation"""
        test_dir = os.path.join(self.temp_dir, "test_dir")
        result = makedir(test_dir)
        assert os.path.exists(test_dir)
        assert result == str(Path(test_dir).resolve())
    
    def test_makedir_existing(self):
        """Test directory creation when directory already exists"""
        test_dir = os.path.join(self.temp_dir, "existing_dir")
        os.makedirs(test_dir)
        result = makedir(test_dir)
        assert os.path.exists(test_dir)
        assert result == str(Path(test_dir).resolve())
    
    def test_save_yaml(self):
        """Test YAML saving"""
        test_data = {'key1': 'value1', 'key2': 42}
        yaml_path = os.path.join(self.temp_dir, 'test.yaml')
        save_yaml(test_data, yaml_path)
        
        assert os.path.exists(yaml_path)
        with open(yaml_path, 'r') as f:
            loaded_data = yaml.safe_load(f)
        assert loaded_data == test_data
    
    def test_load_config(self):
        """Test YAML config loading"""
        test_config = {'param1': 'value1', 'param2': 123}
        config_path = os.path.join(self.temp_dir, 'config.yaml')
        
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        loaded_config = load_config(config_path)
        assert loaded_config == test_config
    
    def test_load_json(self):
        """Test JSON loading"""
        test_data = {'gene1': ['tf1', 'tf2'], 'gene2': ['tf3']}
        json_path = os.path.join(self.temp_dir, 'test.json')
        
        with open(json_path, 'w') as f:
            json.dump(test_data, f)
        
        loaded_data = load_json(json_path)
        assert loaded_data == test_data
    
    def test_get_peak_names_from_file(self):
        """Test peak names file reading"""
        peak_file = os.path.join(self.temp_dir, 'peaks.txt')
        with open(peak_file, 'w') as f:
            f.write("chr1:1000-2000\nchr2:3000-4000\nchr3:5000-6000\n")
        
        peak_names = get_peak_names_from_file(peak_file)
        expected = ["chr1_1000_2000", "chr2_3000_4000", "chr3_5000_6000"]
        assert peak_names == expected


class TestIntegration:
    """Integration tests for the complete pipeline"""
    
    def setup_method(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "integration_config.yaml")
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def teardown_method(self):
        """Clean up after integration tests"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_pipeline_mock(self):
        """Test full pipeline execution with mocked dependencies"""
        # Create comprehensive test data
        adata = sc.AnnData(X=np.random.poisson(5, (50, 500)))
        adata.obs['leiden'] = np.random.choice(['0', '1'], 50)
        adata.obsm['X_umap'] = np.random.randn(50, 2)
        adata.layers['raw_count'] = adata.X.copy()
        adata.write_h5ad(os.path.join(self.temp_dir, 'test_rna.h5ad'))
        
        # Create test files
        with open(os.path.join(self.temp_dir, 'peaks.txt'), 'w') as f:
            f.write("chr1_1000_2000\nchr2_3000_4000\n")
        
        coaccess_df = pd.DataFrame({
            'Peak1': ['chr1_1000_2000', 'chr2_3000_4000'],
            'Peak2': ['chr2_3000_4000', 'chr1_1000_2000'],
            'coaccess': [0.8, 0.9]
        })
        coaccess_df.to_csv(os.path.join(self.temp_dir, 'coaccess.tsv'), sep='\t', index=False)
        
        with open(os.path.join(self.temp_dir, 'tg2tf.json'), 'w') as f:
            json.dump({'GENE1': ['TF1'], 'GENE2': ['TF2']}, f)
        
        # Create config
        config = {
            'rna_h5ad': os.path.join(self.temp_dir, 'test_rna.h5ad'),
            'peak_names_file': os.path.join(self.temp_dir, 'peaks.txt'),
            'peak_coaccess_path': os.path.join(self.temp_dir, 'coaccess.tsv'),
            'TG2TF_json_path': os.path.join(self.temp_dir, 'tg2tf.json'),
            'output_dir': self.output_dir,
            'run_name': 'integration_test',
            'genome_dir': os.path.join(self.temp_dir, 'genomes'),
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
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Mock all external dependencies
        with patch('run_celloracle_inference.CellOraclePipeline') as mock_pipeline_class, \
             patch('celloracle.motif_analysis.get_tss_info') as mock_get_tss, \
             patch('celloracle.motif_analysis.integrate_tss_peak_with_cicero') as mock_integrate, \
             patch('celloracle.motif_analysis.check_peak_format') as mock_check_peak, \
             patch('celloracle.motif_analysis.TFinfo') as mock_tfinfo_class:
            
            # Set up mocks
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            
            # Mock TSS data
            tss_df = pd.DataFrame({
                'chr': ['chr1', 'chr2'],
                'start': [1000, 3000],
                'end': [2000, 4000],
                'gene_short_name': ['GENE1', 'GENE2'],
                'strand': ['+', '-']
            })
            mock_get_tss.return_value = tss_df
            
            # Mock integration result
            integrated_df = pd.DataFrame({
                'peak_id': ['chr1_1000_2000', 'chr2_3000_4000'],
                'gene_short_name': ['GENE1', 'GENE2'],
                'coaccess': [0.8, 0.9]
            })
            mock_integrate.return_value = integrated_df
            mock_check_peak.return_value = integrated_df
            
            # Mock TFinfo
            mock_tfinfo = Mock()
            mock_tfinfo_class.return_value = mock_tfinfo
            mock_tfinfo.to_dataframe.return_value = pd.DataFrame({
                'TF1': [1, 0], 'TF2': [0, 1]
            })
            
            # Mock oracle
            mock_oracle = Mock()
            mock_pipeline.oracle = mock_oracle
            mock_pipeline.base_grn = pd.DataFrame({'TF1': [1], 'TF2': [0]})
            
            # Mock network
            mock_network = Mock()
            mock_network.links_dict = {'0': pd.DataFrame({
                'source': ['TF1'], 'target': ['GENE1'],
                'coef_mean': [0.5], 'coef_abs': [0.5],
                'p': [0.001], '-logp': [3.0]
            })}
            mock_pipeline.network = mock_network
            
            # Execute pipeline
            args = Mock()
            args.config = self.config_path
            args.n_cpu = 2
            
            execute(args)
            
            # Verify all major steps were called
            mock_pipeline.process_base_grn_from_motif_Tfs.assert_called_once()
            mock_pipeline.process_initialize_celloracle.assert_called_once()
            mock_pipeline.process_grn_inference.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
