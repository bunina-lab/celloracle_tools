"""
Supported Species and reference genomes
Human: ['hg38', 'hg19']
Mouse: ['mm39', 'mm10', 'mm9']
S.cerevisiae: ["sacCer2", "sacCer3"]
Zebrafish: ["danRer7", "danRer10", "danRer11"]
Xenopus tropicalis: ["xenTro2", "xenTro3"]
Xenopus laevis: ["Xenopus_laevis_v10.1"]
Rat: ["rn4", "rn5", "rn6"]
Drosophila: ["dm3", "dm6"]
C.elegans: ["ce6", "ce10"]
Arabidopsis: ["TAIR10"]
Chicken: ["galGal4", "galGal5", "galGal6"]
Guinea Pig: ["Cavpor3.0"]
Pig: ["Sscrofa11.1"]

"""

def execute(args):
    from lib.process_celloracle import CellOraclePipeline

    co_pipe = CellOraclePipeline(
        genome_dir=args.genome_dir,
        ref_genome=args.genome_name,
        output_dir=None
    )

    genome_installed = co_pipe.check_genome_installation()
    if genome_installed:
        print(f"Genome {args.genome_name} is already installed in {args.genome_dir}.\nExiting")
        sys.exit()

    genome_installed = co_pipe.install_genome()
    
    print(f"Genome {args.genome_name} installed: {genome_installed}")




if __name__ == "__main__":
    import argparse, os, sys, datetime

    parser = argparse.ArgumentParser(description="Checks and installs supported genome when needed")
    parser.add_argument("--genome_dir", required=True, help="Path to genome installation directory")
    parser.add_argument(
        "--genome_name",
        required=True,
        help="One of the supported genomes",
        choices=[
            "hg38", "hg19",                   # Human
            "mm39", "mm10", "mm9",            # Mouse
            "sacCer2", "sacCer3",             # S. cerevisiae
            "danRer7", "danRer10", "danRer11",# Zebrafish
            "xenTro2", "xenTro3",             # Xenopus tropicalis
            "Xenopus_laevis_v10.1",           # Xenopus laevis
            "rn4", "rn5", "rn6",              # Rat
            "dm3", "dm6",                     # Drosophila
            "ce6", "ce10",                    # C. elegans
            "TAIR10",                         # Arabidopsis
            "galGal4", "galGal5", "galGal6",  # Chicken
            "Cavpor3.0",                      # Guinea Pig
            "Sscrofa11.1"                     # Pig
        ]
    )
    
    args = parser.parse_args()
    execute(args)
