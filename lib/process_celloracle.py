"""
CellOracle Gene Regulatory Network Inference Pipeline

A modular class-based implementation of the CellOracle pipeline for GRN inference
and in-silico perturbation analysis.

Author: Berk
"""

class NotInitializedError(ValueError):
    """
    Raised when an excepted attribute is not initialized
    """

import celloracle as co
from celloracle import motif_analysis as ma
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import genomepy
from typing import Literal, Optional, Dict, List, Tuple, Union
import os
from gimmemotifs.motif import Motif


class CellOraclePipeline:
    """
    Base class for CellOracle pipeline with common functionality.
    """
    
    def __init__(self, 
                 genome_dir: str, 
                 output_dir:str,
                 n_cpu=4,
                 ref_genome: str = "hg38",
                 verbose=False):
        """
        Initialize the CellOracle pipeline.
        
        Args:
            genomes_dir: Directory for genome files
            ref_genome: Reference genome (default: hg38)
        """
        self.genome_dir = genome_dir
        self.ref_genome = ref_genome
        self.output_dir = output_dir
        self.n_cpu = n_cpu
        self.verbose = verbose

        self.oracle = None
        self.base_grn = None
        self.network = None ### Cell Oracle inferred GRN
        self.tf_info = None

        
    def check_genome_installation(self) -> bool:
        """Check if reference genome is installed."""
        genome_installation = ma.is_genome_installed(
            ref_genome=self.ref_genome,
            genomes_dir=self.genome_dir
        )
        print(f"{self.ref_genome} installation: {genome_installation}")
        return genome_installation
    
    def install_genome(self) -> bool:
        """Install reference genome if not available."""
        try:
            print("Installing ref genome...")
            genomepy.install_genome(
                name=self.ref_genome, 
                provider="UCSC", 
                genomes_dir=self.genome_dir
            )
            return True
        except Exception as e:
            print(f"Couldn't install genome: {e}")
            return False
    
    ### START ORACLE OBJ ###
    def process_initialize_celloracle(self, rna_adata, cluster_column_name:str, embedding_name:str, raw_counts:bool, TG_to_TF_dictionary:dict):
        self.initialize_celloracle()
        self.import_anndata(rna_adata, cluster_column_name, embedding_name, raw_counts)
        self.import_base_GRN()
        self.import_TG2TF_dict(TG_to_TF_dictionary)
        self.process_co_knn_imputation()

        self.oracle.to_hdf5(f"{self.output_dir}/initialized.celloracle.oracle")


    def initialize_celloracle(self):
        self.oracle = co.Oracle()

    
    def import_anndata(self, rna_adata, cluster_column_name, embedding_name, raw_counts=False):
        self.oracle.import_anndata_as_raw_count(adata=rna_adata,
                                    cluster_column_name=cluster_column_name,
                                    embedding_name=embedding_name) if raw_counts \
                                        else \
        self.oracle.import_anndata_as_normalized_count(adata=rna_adata,
                                    cluster_column_name=cluster_column_name,
                                    embedding_name=embedding_name)
    
    def import_base_GRN(self, base_grn:pd.DataFrame=None):
        """
        base_grn:
        peak_id	gene_short_name	9430076C15RIK	AC002126.6	AC012531.1	AC226150.2	AFP	AHR	AHRR	AIRE	...	ZNF784	ZNF8	ZNF816	ZNF85	ZSCAN10	ZSCAN16	ZSCAN22	ZSCAN26	ZSCAN31	ZSCAN4
        0	chr10_100346395_100346895	SCD	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
        1	chr10_100346926_100347426	SCD	0	0	0	0	0	1	1	0	...	0	0	0	0	0	0	0	0	0	0
        2	chr10_100735409_100735909	PAX2	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
        3	chr10_102225966_102226466	ELOVL3	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	1	0	0	0
        4	chr10_102394090_102394590	NFKB2	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
        """
        if base_grn is None:
            base_grn = self.base_grn
        self.oracle.import_TF_data(TF_info_matrix=base_grn)


    def import_TG2TF_dict(self, TG_to_TF_dictionary):
        """
        TG_to_TF_dictionary:
            {'gene_name' : ['TF1', 'TF2' ...]}
        """
        # Add TF information
        self.oracle.addTFinfo_dictionary(TG_to_TF_dictionary)


    def process_co_knn_imputation(self):
        # Perform PCA
        self.oracle.perform_PCA()
        n_comps = self.select_PCA_components()
        n_cell = self.oracle.adata.shape[0]
        self.impute(n_comps=n_comps, k=min(200, int(0.02*n_cell)))


    def select_PCA_components(self, min_variance_threshold=0.002, max_components=50):
        variance_ratios = self.oracle.pca.explained_variance_ratio_
        diff_result = np.diff(np.diff(np.cumsum(variance_ratios))) > min_variance_threshold
        indices = np.where(diff_result)[0]
        
        if len(indices) == 0:
            # If no components meet the threshold, return a default value
            n_comps = min(len(variance_ratios), max_components)
        else:
            n_comps = indices[0]
            
        return min(n_comps, max_components)

    def impute(self, n_comps, k, balanced=True):
        self.oracle.knn_imputation(
            n_pca_dims=n_comps, 
            k=k, 
            balanced=balanced, 
            b_sight=k*8,
            b_maxl=k*4, 
            n_jobs=self.n_cpu
            )

    ####### Motif - TF -> Base_GRN processes ###
    def get_tss_annotations(self, peak_names:List[str])->pd.DataFrame:
        """
        peak_names: list of peaks in the "chr_start_end" format  'chrX_9999_10000'
        returns:
            chr	start	end	gene_short_name	strand
        0	chr7	130668558	130669058	COPG2	-
        1	chr10	96043176	96043676	CCNJ	+
        2	chr1	161226384	161226884	MIR5187	+
        3	chr1	161225713	161226213	MIR5187	+
        4	chr2	227871323	227871823	DAW1	+
        """
        peak_names = list(map(lambda x : x.replace(":", "_").replace("-", "_"), peak_names))#atac.var_names)) ### This line goes to outer function call
        return ma.get_tss_info(peak_str_list=peak_names, ref_genome=self.ref_genome)

    @staticmethod
    def integrate_tss_coaccessible_peaks(annotated_tss_peak:pd.DataFrame, co_accessible_peaks_df:pd.DataFrame):
        """
        Integrates cicero-like peak-peak co accessibility data to the TSS info

        annotated_tss_peak: pd.DataFrame
            chr	start	end	gene_short_name	strand
        0	chr7	130668558	130669058	COPG2	-
        1	chr10	96043176	96043676	CCNJ	+
        2	chr1	161226384	161226884	MIR5187	+
        3	chr1	161225713	161226213	MIR5187	+
        4	chr2	227871323	227871823	DAW1	+


        co_accessible_peaks_df: pd.DataFrame ### Comes from cicero pipeline
        Peak1	Peak2	coaccess
        0	chr6_6725033_6725533	chr6_6686988_6687488	0.121002
        1	chr6_159126547_159127047	chr6_158999093_158999593	0.124089
        2	chr6_159126547_159127047	chr6_159125820_159126320	0.272069
        3	chr6_159126547_159127047	chr6_159134625_159135125	0.101309
        4	chr6_158819133_158819633	chr6_158817174_158817674	0.153805



        return: pd.DataFrame
        peak_id	gene_short_name	coaccess
        0	chr10_100267093_100267593	CWF19L1	1.0
        1	chr10_100267687_100268187	CWF19L1	1.0
        2	chr10_100285934_100286434	BLOC1S2	1.0
        3	chr10_100286440_100286940	BLOC1S2	1.0
        4	chr10_100286993_100287493	BLOC1S2	1.0

        
        """
        return ma.integrate_tss_peak_with_cicero(
            tss_peak=annotated_tss_peak,
            cicero_connections=co_accessible_peaks_df
            )

    def get_peak2gene_conns(self, tss_peak_integrated_df:pd.DataFrame, coaccess_threshold=0.8):
        peak_conns = tss_peak_integrated_df[tss_peak_integrated_df["coaccess"] >= coaccess_threshold]
        peak_conns = peak_conns[["peak_id", "gene_short_name"]].reset_index(drop=True)
        peak_conns = ma.check_peak_format(peak_conns, self.ref_genome, genomes_dir=self.genome_dir)
        peak_conns.to_csv(f"{self.output_dir}/celloralce_peak_gene.tsv", index=False, sep="\t")
        return peak_conns

    def process_base_grn_from_motif_Tfs(self, 
    peak_names:List[str], peak_coaccess_df:pd.DataFrame, 
    tf_binding_fpr=0.02, motifs:List[Motif]=None, TF_evidence_direct_only=False,
    motif_threshold=10, motif_filtering_method:Literal["cumulative_score", "individual_score"]="cumulative_score"
    ):
            ## Instantiate TFinfo object   
        tss_df = self.get_tss_annotations(peak_names)
        tss_peak_integrated_df = self.integrate_tss_coaccessible_peaks(annotated_tss_peak=tss_df, co_accessible_peaks_df=peak_coaccess_df)
        peak_conns = self.get_peak2gene_conns(tss_peak_integrated_df)

        self.initialize_TFinfo(peak_df=peak_conns)
        self.scan_motifs(fpr=tf_binding_fpr, motifs=motifs, TF_evidence_direct_only=TF_evidence_direct_only)

        self.filter_motifs(threshold=motif_threshold, method=motif_filtering_method)

        self.get_base_grn()

    def initialize_TFinfo(self, peak_df):
        """
        peak_df: pd.DataFrame

        peak_id	gene_short_name
        5	chr10_100346395_100346895	SCD
        6	chr10_100346926_100347426	SCD
        9	chr10_100735409_100735909	PAX2
        12	chr10_102225966_102226466	ELOVL3
        15	chr10_102394090_102394590	NFKB2
        
        """
        self.tf_info = ma.TFinfo(peak_data_frame=peak_df,
                    ref_genome=self.ref_genome,
                    genomes_dir=self.genome_dir)
    
    def scan_motifs(self, fpr: float = 0.02, 
                   motifs: Optional[List[Motif]] = None, 
                   TF_evidence_direct_only=False) -> None:
        """
        Scan DNA sequences searching for TF binding motifs.

        Args:
           background_length (int): background length. This is used for the calculation of the binding score.

           fpr (float): False positive rate for motif identification.

           n_cpus (int): number of CPUs for parallel calculation.

           verbose (bool): Whether to show a progress bar.

           motifs (list): a list of gimmemotifs motifs, will revert to default_motifs() if None

           TF_evidence_level (str): Please select one from ["direct", "direct_and_indirect"]. If "direct" is selected, TFs that have a binding evidence were used.
               If "direct_and_indirect" is selected, TFs with binding evidence and inferred TFs are used.
               For more information, please read explanation of Motif class in gimmemotifs documentation (https://gimmemotifs.readthedocs.io/en/master/index.html)

        """
        if self.tf_info is None:
            raise NotInitializedError("TFinfo object not initialized. Call get_tf_info() first.")
            
        # Scan motifs
        self.tf_info.scan(
            background_length=200,
            fpr=fpr, 
            n_cpus=self.n_cpu, 
            verbose=self.verbose, 
            motifs=motifs, 
            TF_evidence_level="direct" if TF_evidence_direct_only else "direct_and_indirect", 
            TF_formatting="auto", 
            batch_size=None, 
            divide=100000
            )
        
        # Save TFinfo object
        self.tf_info.to_hdf5(file_path=f"{self.output_dir}/scanned.celloracle.tfinfo")
    
    def filter_motifs(self, threshold: float = 10, 
                     method: str = "cumulative_score") -> None:
        """
        Filter motifs by binding score.
        
        Args:
            threshold: Score threshold for filtering
            method: Filtering method ("individual_score" or "cumulative_score")
        """
        if self.tf_info is None:
            raise ValueError("TFinfo object not initialized.")
            
        # Reset filtering
        self.tf_info.reset_filtering()
        
        # Apply filtering
        self.tf_info.filter_motifs_by_score(threshold=threshold, method=method)
    
    def get_base_grn(self):
        """
        Convert TFinfo results to DataFrame format.
        
        Args:
            verbose: Verbose output
            
        Returns:
            DataFrame with TF binding information
        """
        if self.tf_info is None:
            raise NotInitializedError("TFinfo object not initialized.")
            
        # Format results
        self.tf_info.make_TFinfo_dataframe_and_dictionary(verbose=self.verbose)
        
        # Get base GRN
        self.base_grn = self.tf_info.to_dataframe()
        self.base_grn.to_parquet(f"{self.output_dir}/heart_data_base_GRN_dataframe.parquet")
        return self.base_grn
    

    #### GRN inference ###
    def process_grn_inference(self, cluster_colum, egde_p_value=0.001, weights_name="weight", genelist_source=None, genelist_target=None):
        if any([obj is None for obj in (self.oracle, self.base_grn)]):
            raise NotInitializedError("oracle or base_grn objects are not initialized. Call get_base_grn() or process_initialize_celloracle() methods first")

        ## initialize cluster specific network
        self.get_cluster_networks(cluster_colum)
        self.filter_edges(egde_p_value, weight=weights_name, genelist_source=genelist_source, genelist_target=genelist_target)
        self.save_network(cluster_colum)

        ## GRN summary statistics
        self.get_network_score()
        self.plot_node_degree_distibutions()
        self.save_graph_summary_stats(cluster_colum)

        graph_stats= ["degree_centrality_all", "degree_centrality_in", "degree_centrality_out", "betweenness_centrality", "eigenvector_centrality"]
        self.plot_score_distributions(values=graph_stats)
        self.plot_network_entropy_distribution()

        cluster_keys = list(self.network.links_dict.keys())
        

        for idx, cluster_key in enumerate(cluster_keys):
            self.plot_scores_rankings(cluster_key)
            self.save_cluster_speficic_network(cluster_key)

            if idx+1 == len(cluster_keys): break
            for cluster_2_key in cluster_keys[idx+1:]:
                map(lambda x: self.plot_score_comparisons(
                                comparison_value=x,
                                cluster_1=cluster_key,
                                cluster_2=cluster_2_key
                ),graph_stats
                )


    def get_cluster_networks(self, cluster_column):
        self.network = self.oracle.get_links(
            cluster_name_for_GRN_unit=cluster_column, 
            alpha=10,
            verbose_level=10 if self.verbose else 0
            )
        return self.network
    
    def save_cluster_speficic_network(self, cluster_key):
        self.get_cluster_speficic_network(cluster_key).to_csv(f"{self.output_dir}/{cluster_key}_source_target_network.tsv", index=False, sep="\t")
    
    def get_cluster_speficic_network(self, cluster_key):
        return self.network.links_dict[cluster_key]

    def save_network(self, cluster_column):
        self.network.to_hdf5(file_path=f"{self.output_dir}/f{cluster_column}.celloracle.links")

    def save_graph_summary_stats(self, cluster_column):
        self.network.merged_score.to_csv(f"{self.output_dir}/{cluster_column}_gene_graph_summary_stats.tsv", index=True, sep="\t")

    def filter_edges(self, p_value=0.001, weight="coef_abs", genelist_source=None, genelist_target=None, threshold_number=20000):
        """
        p (float): threshold for p-value of the network edge.
        weight (str): Please select network weight name for the filtering
        genelist_source (list of str): gene list to remain in regulatory gene nodes. Default is None.
        genelist_target (list of str): gene list to remain in target gene nodes. Default is None.
        """
        self.network.filter_links(p=p_value, weight="coef_abs", threshold_number=threshold_number, genelist_source=genelist_source, genelist_target=genelist_target)

    def plot_node_degree_distibutions(self):
        self.network.plot_degree_distributions(plot_model=False, save=self.output_dir)


    def get_network_score(self):
        self.network.get_network_score()


    def plot_scores_rankings(self, cluster_name, top_n_gene=30):
        self.network.plot_scores_as_rank(
                cluster=cluster_name, 
                n_gene=top_n_gene, 
                save=f"{self.output_dir}/ranked_score/{cluster_name}"
        )

    def plot_score_comparisons(self, comparison_value, cluster_1, cluster_2, percentile=99):
        # Compare GRN score between two clusters
        self.network.plot_score_comparison_2D(
                                value=comparison_value,
                                cluster1=cluster_1, cluster2=cluster_2,
                                percentile=percentile,
                                save=f"{self.output_dir}/score_comparison/"
                                )

    
    def plot_score_distributions(self, values:List, method="boxplot"):
        self.network.plot_score_discributions(values=values,
                               method=method,
                               save=self.output_dir,
                              )
    
    def plot_network_entropy_distribution(self):
        self.network.plot_network_entropy_distributions(
            save=self.output_dir
    )
    
    def get_cluster_network(self, cluster_name)->pd.DataFrame:
        """
        returns: 
        	source	target	coef_mean	coef_abs	p	-logp
        0	FOXA1	A2M	0.006124	0.006124	4.347631e-03	2.361747
        1	EPAS1	A2M	0.115925	0.115925	1.222158e-09	8.912873
        2	SOX2	A2M	0.018543	0.018543	4.646367e-07	6.332887
        3	NR1D1	A2M	0.015789	0.015789	9.760397e-04	3.010533
        4	TCF7	A2M	-0.036418	0.036418	2.028253e-08	7.692878
        """
        return self.network.links_dict[cluster_name]
    

    ##### In-silico Perturbations ###
    def initialize_similation_model(self, use_cluster_specific_TFdict=True):
        ## Make predictive models for simulation ###
        self.network.filter_links()
        self.oracle.get_cluster_specific_TFdict_from_Links(links=self.network)
        self.oracle.fit_GRN_for_simulation(alpha=10,
                                    use_cluster_specific_TFdict=use_cluster_specific_TFdict)

        self.oracle.to_hdf5(f"{self.output_dir}/similation.celloracle.oracle")
    
    def calc_similate_shift(self, conditions:Dict, n_propagation=3):
        """
        conditions: {"gene_name" : float}  if float == 0.0 --> gene_knockdown
        """
        self.oracle.simulate_shift(
                      perturb_condition=conditions,
                      n_propagation=n_propagation
                      )

    def transition_prob_estimation(self, 
            n_neighbors, 
            knn_random=True,
            sampled_fraction=0.3, 
            sampling_probs=(0.1, 0.05), 
            calculate_randomized=True,
            sigma_corr=0.05
            ):
        ## Need to make it analyse per cluster separately, else it's exceeds memory
        self.oracle.estimate_transition_prob(
            n_neighbors=n_neighbors,              # Much smaller than default
            knn_random=knn_random,
            sampled_fraction=sampled_fraction,       # Reduced from 0.3
            sampling_probs=sampling_probs,  # Sample fewer distant neighbors
            calculate_randomized=calculate_randomized,  # Skip negative control
            n_jobs=self.n_cpu,
            threads=self.n_cpu                 # Match your CPU cores
        )

        self.oracle.calculate_embedding_shift(sigma_corr=sigma_corr)


    def calculate_p_mass(self, smooth=0.8, n_grid=40, n_neighbors=200):
        """
        Calculate probability mass for oracle object.
        
        Parameters:
        -----------
        oracle : Oracle object
            The oracle object to perform calculations on
        smooth : float, default=0.8
            Smoothing parameter for mass calculation
        n_grid : int, default=40
            Number of grid points
        n_neighbors : int, default=200
            Number of neighbors to consider
            
        Returns:
        --------
        oracle : Oracle object
            Oracle object with calculated p_mass
        """
        self.oracle.calculate_p_mass(smooth=smooth, n_grid=n_grid, n_neighbors=n_neighbors)


    def suggest_and_apply_mass_filter(self, n_suggestion=12, min_mass=0.01):
        """
        Suggest mass thresholds and apply mass filter.
        
        Parameters:
        -----------
        oracle : Oracle object
            The oracle object to perform calculations on
        n_suggestion : int, default=12
            Number of threshold suggestions to generate
        min_mass : float, default=0.01
            Minimum mass threshold to apply
            
        Returns:
        --------
        oracle : Oracle object
            Oracle object with applied mass filter
        """
        self.oracle.suggest_mass_thresholds(n_suggestion=n_suggestion)
        self.oracle.calculate_mass_filter(min_mass=min_mass, plot=False)


    def plot_simulation_comparison(self, goi, scale_simulation=0.5):
        """
        Plot comparison of simulated and randomized flow vectors.
        
        Parameters:
        -----------
        oracle : Oracle object
            The oracle object containing simulation data
        goi : str
            Gene of interest name for plot title
        scale_simulation : float, default=0.5
            Scale factor for vector arrows
        figsize : tuple, default=(13, 6)
            Figure size as (width, height)
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure object
        ax : array of matplotlib.axes.Axes
            Array of axes objects
        """
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        
        # Show quiver plot
        self.oracle.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax[0])
        ax[0].set_title(f"Simulated cell identity shift vector: {goi} KO")
        
        # Show quiver plot that was calculated with randomized graph
        self.oracle.plot_simulation_flow_random_on_grid(scale=scale_simulation, ax=ax[1])
        ax[1].set_title(f"Randomized simulation vector")
        
        ###plt.save_plot()


    def plot_vector_field_with_clusters(self, scale_simulation=0.5, 
                                        point_size=10):
        """
        Plot vector field overlaid on cell clusters.
        
        Parameters:
        -----------
        scale_simulation : float, default=0.5
            Scale factor for vector arrows
        point_size : int, default=10
            Size of scatter points for cells
            
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        self.oracle.plot_cluster_whole(ax=ax, s=point_size)
        self.oracle.plot_simulation_flow_on_grid(scale=scale_simulation, ax=ax, 
                                        show_background=False)
        
        ## save the plot !! plt.show()
    
    def _get_motif_list_from_file():
        ...