

def execute(args):
    import scanpy as sc
    import pandas as pd
    from lib.process_celloracle import CellOraclePipeline
    from lib.utils import makedir, save_yaml, load_config, load_json, get_peak_names_from_file

    cfg = load_config(args.config)

    # paths and run naming
    rna_path = cfg['rna_h5ad']
    ###atac_path = cfg.get('atac_h5ad', None) no need to load atac adata, we just neead a file of peak_names

    outdir = makedir(os.path.join(cfg['output_dir'], cfg.get('run_name', datetime.date.today().isoformat())))

    save_yaml(cfg, os.path.join(outdir, "used_config.yaml"))

    # read AnnData inputs
    print(f"Loading RNA AnnData from {rna_path} ...")
    adata_rna = sc.read_h5ad(rna_path)


    ##chek .obs column if cluster key (cell_type) key exists
    if not cfg["cluster_column"] in adata_rna.obs:
        raise ValueError(f"{cfg['cluster_column']}  key not found in adata.obs!")
    
    ### check peak coeff tabl column
    peak_coeff_df = pd.read_csv(cfg["peak_coaccess_path"], sep="\t")
    if len(peak_coeff_df.columns) != 3:
        raise ValueError(f"Expected 3 columns: peak1, peak2, weight.\nBut columns are:\n {peak_coeff_df.columns}")
    #weights_column = peak_coeff_df.columns[-1]


    # optionally set adata.X to raw_count layer (the tutorial uses raw counts for oracle input)
    if cfg.get('raw_counts', True):
        if 'raw_count' in adata_rna.layers:
            print("Setting adata.X = adata.layers['raw_count'] (raw counts) for Oracle import.")
            adata_rna.X = adata_rna.layers['raw_count'].copy()
        else:
            print("No 'raw_count' layer found in RNA AnnData; leaving adata.X as-is.")

    # instantiate Oracle pipe

    co_pipe = CellOraclePipeline(
        genome_dir=cfg["genome_dir"],
        ref_genome=cfg["reference_dir"],
        output_dir=outdir,
        verbose=cfg.get("verbose", False),
        n_cpu=args.n_cpu
    )

    ### First calculate base grn
    co_pipe.process_base_grn_from_motif_Tfs(
        peak_names=get_peak_names_from_file(cfg["peak_names_file"]),
        peak_coaccess_df=peak_coeff_df,
        tf_binding_fpr=cfg["tf_binding_frp"],
        motifs=None, ## cfg.get("motifs_file", None),
        TF_evidence_direct_only=cfg.get("TF_evidence_direct", False),
        motif_filtering_method=cfg["motif_filtering_method"],
        motif_threshold=cfg["motif_threshold"],  
    )

    ### Initiate oracle
    co_pipe.process_initialize_celloracle(
        rna_adata=adata_rna,
        cluster_column_name=cfg["cluster_column"],
        embedding_name=cfg["embedding"],
        raw_counts=cfg["raw_counts"],
        TG_to_TF_dictionary=load_json(cfg["TG2TF_json_path"]),
    )

    ### GRN inference
    ## get_gene_lists_from_file(cfg["source_target_genelist"])
    co_pipe.process_grn_inference(
        cluster_colum=cfg["cluster_column"],
        egde_p_value=cfg["grn_edge_p_threshold"],
        #weights_name=weights_column,
        genelist_source=None,
        genelist_target=None,
    )

    print("Pipeline finished. Outputs in:", outdir)

if __name__ == "__main__":
    import argparse, os, sys, datetime

    parser = argparse.ArgumentParser(description="Run CellOracle GRN pipeline from YAML config")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--n_cpu", required=False, help="number of available cpus", default=4, type=int)
    args = parser.parse_args()
    execute(args)
