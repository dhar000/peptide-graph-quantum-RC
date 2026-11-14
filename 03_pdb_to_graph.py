"""
03_pdb_to_graph.py
------------------
Parses 3D peptide PDB files into residue contact graphs (C-alpha Euclidean
distance cutoff = 7.0 Angstroms) using BioPython and NetworkX, extracts
topological graph metrics, and converts structures into PyTorch Geometric Data tensors.
"""

import os
import networkx as nx
import numpy as np
import pandas as pd
import torch
from Bio import PDB
from torch_geometric.data import Data

AA_TO_INT = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4, "GLN": 5, "GLU": 6,
    "GLY": 7, "HIS": 8, "ILE": 9, "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13,
    "PRO": 14, "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
}


def encode_pdb_to_graph(pdb_path: str, distance_threshold: float = 7.0):
    """Parses a PDB structure and builds a C-alpha contact graph."""
    if not os.path.exists(pdb_path):
        return None

    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("peptide", pdb_path)
    except Exception as exc:
        print(f"Error parsing {pdb_path}: {exc}")
        return None

    chains = list(structure.get_chains())
    if not chains:
        return None

    residues = [r for r in chains[0].get_residues() if PDB.is_aa(r)]
    graph = nx.Graph()

    for res in residues:
        if "CA" in res:
            res_name = res.get_resname()
            res_id = res.get_id()
            node_label = f"{res_name}{res_id}"
            graph.add_node(
                node_label,
                coord=res["CA"].get_coord(),
                res_name=res_name,
                res_id=res_id,
            )

    nodes = list(graph.nodes(data=True))
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node_i, data_i = nodes[i]
            node_j, data_j = nodes[j]
            dist = float(np.linalg.norm(data_i["coord"] - data_j["coord"]))
            if dist <= distance_threshold:
                graph.add_edge(node_i, node_j, distance=dist)

    return graph


def process_structural_graphs(
    csv_path: str = "data/processed/quantum_peptide_dataset.csv",
    pdb_dir: str = "data/raw/structures",
    output_csv: str = "data/processed/quantum_peptide_graph_features.csv",
    distance_threshold: float = 7.0,
):
    """Builds contact graphs for all peptides and exports graph topology metrics."""
    df = pd.read_csv(csv_path)
    graph_objects = {}
    metrics = []

    for idx, row in df.iterrows():
        pdb_file = os.path.join(pdb_dir, f"{str(row['Peptide_ID']).lower()}.pdb")
        G = encode_pdb_to_graph(pdb_file, distance_threshold=distance_threshold)

        if G is not None and G.number_of_nodes() > 0:
            graph_objects[idx] = G
            metrics.append({
                "Node_Count": G.number_of_nodes(),
                "Edge_Count": G.number_of_edges(),
                "Graph_Density": nx.density(G),
            })
        else:
            metrics.append({"Node_Count": 0, "Edge_Count": 0, "Graph_Density": 0.0})

    enriched_df = pd.concat([df, pd.DataFrame(metrics)], axis=1)
    enriched_df.to_csv(output_csv, index=False)
    print(f"Saved topological graph features for {len(enriched_df)} peptides to '{output_csv}'.")
    return graph_objects, enriched_df


def convert_nx_to_pyg_tensors(graphs_dict: dict, dataset_df: pd.DataFrame) -> list:
    """Converts NetworkX contact graphs into PyTorch Geometric Data objects."""
    pyg_dataset = []

    for idx, row in dataset_df.iterrows():
        G = graphs_dict.get(idx)
        if G is None or G.number_of_nodes() == 0:
            continue

        node_features = []
        node_mapping = {}
        for i, (node_name, data) in enumerate(G.nodes(data=True)):
            node_mapping[node_name] = i
            aa_id = AA_TO_INT.get(data.get("res_name", "GLY"), 7)
            coords = data.get("coord", np.zeros(3))
            node_features.append([aa_id] + list(coords))

        x_tensor = torch.tensor(node_features, dtype=torch.float)

        edge_indices = []
        for u, v in G.edges():
            src, dst = node_mapping[u], node_mapping[v]
            edge_indices.append([src, dst])
            edge_indices.append([dst, src])

        edge_index_tensor = (
            torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            if edge_indices
            else torch.empty((2, 0), dtype=torch.long)
        )

        y_tensor = torch.tensor([row["Label_ID"]], dtype=torch.long)
        data_obj = Data(x=x_tensor, edge_index=edge_index_tensor, y=y_tensor)
        data_obj.peptide_id = row["Peptide_ID"]
        pyg_dataset.append(data_obj)

    return pyg_dataset


if __name__ == "__main__":
    graphs, df_features = process_structural_graphs()
    tensors = convert_nx_to_pyg_tensors(graphs, df_features)
    print(f"Prepared {len(tensors)} PyTorch Geometric graph tensors.")
