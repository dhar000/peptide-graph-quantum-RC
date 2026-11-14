# Peptide-Graph-QRC: 3D Structural Graph & Quantum Reservoir Encoding Pipeline

A modular bioinformatics and quantum machine learning data pipeline for converting therapeutic peptide sequences into 3D residue contact graphs, low-dimensional structural embeddings, and Quantum Reservoir Computing (QRC) feature representations using PennyLane.

## Dataset Summary
The curated dataset (`data/processed/quantum_peptide_dataset.csv`) contains **292 non-overlapping peptide sequences** across three functional classes:
* **AMP (Antimicrobial Peptides, Label `0`):** 97 sequences
* **ACP (Anticancer Peptides, Label `1`):** 97 sequences
* **ADP (Antidiabetic Peptides, Label `2`):** 98 sequences

## Repository Structure

```text
peptide-graph-qrc/
├── data/
│   ├── raw/                                  # Target folder for downloaded ESMFold .pdb structures
│   └── processed/
│       ├── quantum_peptide_dataset.csv       # Master sequence index & class labels (292 peptides)
│       ├── quantum_peptide_graph_features.csv # Topological contact graph metrics (nodes, edges, density)
│       ├── peptide_gat_4d_embeddings.csv     # 4D structural graph embeddings
│       ├── peptide_gat_5d_embeddings.csv     # 5D structural graph embeddings
│       ├── quantum_angle_encoded_states.csv  # 4-qubit angle-encoded statevectors (32 real/imag features)
│       └── quantum_reservoir_25d_output.csv  # 5-qubit QRC observables (15 single-qubit + 10 ZZ correlations)
├── src/
│   ├── 01_build_dataset.py                   # Sequence curation & cross-class duplicate validation
│   ├── 02_fetch_pdb_structures.py            # Automated 3D structure retrieval via ESMFold API
│   ├── 03_pdb_to_graph.py                    # C-alpha contact graph (7.0 A) & PyG tensor builder
│   └── 04_quantum_encoding.py                # PennyLane 4-qubit angle & 5-qubit reservoir circuits
├── .gitignore
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/peptide-graph-qrc.git
cd peptide-graph-qrc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run pipeline stages sequentially
python src/01_build_dataset.py
python src/02_fetch_pdb_structures.py
python src/03_pdb_to_graph.py
python src/04_quantum_encoding.py
```

## Roadmap
- [x] Non-overlapping AMP / ACP / ADP sequence curation (`292` peptides)
- [x] Automated ESMFold 3D `.pdb` structure retrieval
- [x] $C_\alpha$ residue contact graph (`7.0 Å`) & PyTorch Geometric tensor conversion
- [x] PennyLane 4-qubit angle encoding & 5-qubit quantum reservoir feature extraction
- [ ] Supervised GNN pre-training with one-hot + physicochemical node features
- [ ] Data re-uploading QRC circuit optimization & homology-clustered classification benchmarks
