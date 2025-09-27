"""
Run AlphaFold 3 for antibody structure prediction.

This script is based on the AlphaFold 3  from the AlphaFold 3 repository:
https://github.com/deepmind/alphafold/

Attribution:
If you use this code, the AlphaFold 3 model, or outputs produced by it in your research, please cite:

Abramson, J., Adler, J., Dunger, J., Evans, R., Green, T., Pritzel, A., Ronneberger, O., Willmore, L., Ballard, A. J., Bambrick, J., et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature, 630(8016), 493–500. https://doi.org/10.1038/s41586-024-07487-w

Copyright 2024 DeepMind Technologies Limited.

License:
- The AlphaFold 3 source code is licensed under the Creative Commons Attribution-Non-Commercial ShareAlike International License, Version 4.0 (CC-BY-NC-SA 4.0). See: https://github.com/google-deepmind/alphafold3/blob/main/LICENSE
- The AlphaFold 3 model parameters are subject to the AlphaFold 3 Model Parameters Terms of Use: https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

You may not use this file or the model parameters except in compliance with these terms.
"""

import subprocess
import argparse
import tempfile
import os
import json
import shutil
import numpy as np
import pandas as pd

from typing import Union, List
from colabfold.colabfold import run_mmseqs2
from Bio import PDB
from concurrent.futures import ProcessPoolExecutor, TimeoutError


def create_input_dict(
    binder_seq, target_seq, binder_chain, target_chains, design_name, seed
):
    """
    Create input JSON data for AlphaFold3 inference.

    Args:
        binder_seq (str): Amino acid sequence of the binder protein.
        target_seq (str): Amino acid sequence of the target protein.
        binder_chain (str): Chain ID for the binder protein.
        target_chains (Union[str, List[str]]): Chain ID(s) for the target protein.
        design_name (str): Name identifier for this design.
        seed (Union[int, List[int]]): Random seed(s) for model inference.

    Returns:
        dict: AF3-compatible input JSON structure.
    """
    if isinstance(target_chains, str):
        target_chains = [target_chains]
    if isinstance(seed, int):
        seed = [seed]

    input_json_data = {
        "name": design_name,
        "modelSeeds": seed,
        "dialect": "alphafold3",
        "version": 2,
    }

    sequences = []
    for chain_id in [binder_chain] + target_chains:
        sequences.append(
            {
                "protein": {
                    "id": [chain_id],
                    "sequence": binder_seq if chain_id == binder_chain else target_seq,
                }
            }
        )
    input_json_data["sequences"] = sequences
    return input_json_data


def extract_structure_and_scores(output_dir, design_name):
    """
    Extract predicted structure and confidence scores from AF3 output.

    Processes AF3 results by converting the output CIF file to PDB format,
    extracting confidence metrics from JSON files, and cleaning up temporary
    files to save disk space.

    Args:
        output_dir (str): Directory containing AF3 output folder.
        design_name (str): Name identifier for the design.

    Returns:
        tuple: (pdb_path, scores_dict) where:
            - pdb_path (str): Path to converted PDB structure file
            - scores_dict (dict): Confidence metrics including pLDDT, PAE, pTM, iPTM
    """

    af3_results_folder = os.path.join(output_dir, design_name)
    # Convert mmCIF structure file to PDB format for compatibility
    af3_structure = os.path.join(af3_results_folder, f"{design_name}_model.cif")
    pdb_path = os.path.join(output_dir, f"{design_name}_af3.pdb")
    parser = PDB.MMCIFParser(QUIET=True)
    io = PDB.PDBIO()
    structure = parser.get_structure("structure", af3_structure)
    io.set_structure(structure)
    io.save(pdb_path)
    # Extract confidence scores from AF3 JSON output files
    summary_confidences = os.path.join(
        af3_results_folder, f"{design_name}_summary_confidences.json"
    )
    full_confidences = os.path.join(
        af3_results_folder, f"{design_name}_confidences.json"
    )
    af3_scores = {}
    with open(summary_confidences, "r") as f:
        summary_metrics = json.load(f)
    with open(full_confidences, "r") as f:
        full_metrics = json.load(f)
    af3_scores["plddt"] = np.mean(full_metrics["atom_plddts"]) / 100
    pae_matrix = np.array(full_metrics["pae"])
    af3_scores["pae_matrix"] = pae_matrix
    af3_scores["pae"] = np.mean(pae_matrix)
    af3_scores["ptm"] = [summary_metrics["ptm"]]
    af3_scores["iptm"] = [summary_metrics["iptm"]]
    af3_scores["aggregate_score"] = [summary_metrics["ranking_score"]]
    # Clean up temporary AF3 job folder to save disk space
    shutil.rmtree(af3_results_folder)

    return pdb_path, af3_scores


def _run_af3(
    input_json: dict,
    output_dir: str,
    run_settings: dict,
) -> tuple:
    """
    Execute AlphaFold3 structure prediction via Singularity container.

    Runs AF3 inference using the provided input JSON and configuration settings.
    The function handles MSA generation, container execution, and result extraction.

    Args:
        input_json (dict): AF3-compatible input JSON with sequence information.
        output_dir (str): Directory to save prediction outputs.
        binder_chain (str): Chain identifier for the binder protein.
        msa_mode (str): MSA generation mode ("none", "local", "colabfold", "target").
        run_settings (dict): Configuration containing all AF3 paths and settings.

    Returns:
        tuple: (pdb_path, scores_dict) where:
            - pdb_path (str): Path to predicted structure in PDB format
            - scores_dict (dict): Confidence metrics and scores
    """
    # Verify output directory exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print(f"Directory created at {output_dir}.")

    # Process input. if a path, load the dict, and get the dir name. otherwise, use a temp dir.
    input_dir = os.path.join(output_dir, "af3_inputs")
    os.makedirs(input_dir, exist_ok=True)
    input_path = os.path.join(input_dir, f"{input_json['name']}.json")

    # Write updated JSON for AF3
    with open(input_path, "w") as f:
        json.dump(input_json, f)

    af3_repo_path = run_settings["af3_repo_path"]

    run_cmds = [
        "python",
        os.path.join(af3_repo_path, "run_alphafold.py"),
        "--json_path",
        input_path,
        "--output_dir",
        output_dir,
    ]

    popen = subprocess.Popen(
        run_cmds,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        universal_newlines=True,
    )  # stderr=subprocess.DEVNULL
    for line in popen.stdout:
        continue

    popen.stdout.close()
    return_code = popen.wait()

    if return_code:
        raise subprocess.CalledProcessError(return_code)

    pdb_path, scores = extract_structure_and_scores(output_dir, input_json["name"])

    return pdb_path, scores


def run_af3(
    binder_seq: str,
    target_seq: str,
    target_chains: Union[List[str], str],
    output_dir: str,
    design_name: str,
    seed: Union[int, List[int]],
    run_settings: dict,
    binder_chain: str = "B",
    msa_mode: str = "none",
):
    """
    Run AlphaFold3 structure prediction for antibody-target complex.

    High-level interface for AF3 structure prediction. Creates input JSON,
    generates MSAs if requested, runs AF3 inference, and returns the predicted
    structure with confidence scores.

    Args:
        binder_seq (str): Amino acid sequence of the antibody binder.
        target_seq (str): Amino acid sequence of the target protein.
        target_chains (Union[List[str], str]): Chain identifier(s) for target protein.
        output_dir (str): Directory to save prediction outputs.
        design_name (str): Unique identifier for this design.
        seed (Union[int, List[int]]): Random seed(s) for reproducible predictions.
        run_settings (dict): Configuration containing AF3 paths and MSA settings:
            - af3_repo_path: Path to AlphaFold3 repository
            - af3_sif_path: Path to AF3 Singularity image
            - af3_model_dir: Path to AF3 model weights
            - af3_db_dir: Path to AF3 public databases
            - msa_db_dir: Path to ColabFold MSA databases
            - use_metagenomic_db: Whether to use metagenomic databases
        binder_chain (str, optional): Chain ID for binder protein. Defaults to 'B'.
        msa_mode (str, optional): MSA generation method:
            - 'none': No MSA generation
            - 'local': Use local ColabFold databases
            - 'colabfold': Use ColabFold remote API
            - 'target': Generate MSA only for target protein
            Defaults to 'none'.

    Returns:
        tuple: (pdb_path, scores_dict) where:
            - pdb_path (str): Path to predicted complex structure (PDB format)
            - scores_dict (dict): Confidence metrics including pLDDT, PAE, pTM, iPTM
    """

    input_json_data = create_input_dict(
        binder_seq, target_seq, binder_chain, target_chains, design_name, seed
    )
    return _run_af3(
        input_json_data,
        output_dir,
        run_settings=run_settings,
    )


def main(
    input_json: str,
    output_dir: str,
    af3_repo_path: str,
):
    print("Running AF3...")

    results = pd.DataFrame(columns=["name", "plddt", "iptm"])
    with open(input_json) as f:
        input_json_data = json.load(f)
        if isinstance(input_json_data, list):
            print(
                f"Detected list input with {len(input_json_data)} items. Running AF3 for each."
            )
            for i, input_json_dict in enumerate(input_json_data):
                run_settings = {
                    "af3_repo_path": af3_repo_path,
                }
                pdb_path, scores = _run_af3(
                    input_json_dict,
                    output_dir,
                    run_settings=run_settings,
                )
                results = results.append(
                    {
                        "name": input_json_dict["name"],
                        "plddt": scores["plddt"],
                        "iptm": scores["iptm"],
                    },
                    ignore_index=True,
                )
                # save pae matrix
                pae_matrix_path = os.path.join(
                    output_dir, f"{input_json_dict['name']}_pae_matrix.npy"
                )
                np.save(pae_matrix_path, scores["pae_matrix"])

                print(
                    f"Folded {i + 1}/{len(input_json_data)} AF3 structures: design {input_json_dict['name']}, iptm {scores['iptm']}, plddt {scores['plddt']}"
                )
        else:
            run_settings = {
                "af3_repo_path": af3_repo_path,
            }
            pdb_path, scores = _run_af3(
                input_json_data,
                output_dir,
                run_settings=run_settings,
            )
            results = results.append(
                {
                    "name": input_json_data["name"],
                    "plddt": scores["plddt"],
                    "iptm": scores["iptm"],
                },
                ignore_index=True,
            )
            # save pae matrix
            pae_matrix_path = os.path.join(
                output_dir, f"{input_json_data['name']}_pae_matrix.npy"
            )
            np.save(pae_matrix_path, scores["pae_matrix"])

    results.to_csv(os.path.join(output_dir, "results.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        "-i",
        required=True,
        help="Path to input json file. Must be absolute",
    )
    parser.add_argument(
        "--output_dir", "-o", required=True, help="Output directory for AF3 outputs."
    )
    parser.add_argument(
        "--af3_repo_path", required=False, help="Path to local AlphaFold3 repo to bind."
    )
    args = parser.parse_args()

    main(
        input_json=args.input_json,
        output_dir=args.output_dir,
        af3_repo_path=args.af3_repo_path,
    )
