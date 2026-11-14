"""
04_quantum_encoding.py
----------------------
Generates quantum feature representations from low-dimensional structural
embeddings using PennyLane:
1. 4-Qubit Angle Embedding -> 16-dimensional complex quantum statevector (32 real/imag features).
2. 5-Qubit Quantum Reservoir -> 25-dimensional expectation value readout
   (15 single-qubit Pauli-X/Y/Z + 10 two-qubit Pauli-ZZ correlations).
"""

import itertools
import os
import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler


def generate_angle_encoded_states(
    input_csv: str = "data/processed/peptide_gat_4d_embeddings.csv",
    output_csv: str = "data/processed/quantum_angle_encoded_states.csv",
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    feature_cols = ["GAT_Dim_1", "GAT_Dim_2", "GAT_Dim_3", "GAT_Dim_4"]
    X = df[feature_cols].to_numpy()

    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_scaled = scaler.fit_transform(X)

    num_qubits = 4
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def angle_circuit(features):
        qml.AngleEmbedding(features, wires=range(num_qubits), rotation="X")
        return qml.state()

    states = angle_circuit(X_scaled)
    data_dict = {
        "Peptide_ID": df["Peptide_ID"],
        "Class_Label": df["Class_Label"],
    }
    for i in range(states.shape[1]):
        data_dict[f"QS_Real_{i + 1}"] = states[:, i].real
        data_dict[f"QS_Imag_{i + 1}"] = states[:, i].imag

    df_out = pd.DataFrame(data_dict)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"Saved {len(df_out)} angle-encoded statevectors to '{output_csv}'.")
    return df_out


def generate_quantum_reservoir_features(
    input_csv: str = "data/processed/peptide_gat_5d_embeddings.csv",
    output_csv: str = "data/processed/quantum_reservoir_25d_output.csv",
    seed: int = 42,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    feature_cols = ["GAT_Dim_1", "GAT_Dim_2", "GAT_Dim_3", "GAT_Dim_4", "GAT_Dim_5"]
    X = df[feature_cols].to_numpy()

    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_scaled = scaler.fit_transform(X)

    num_qubits = 5
    dev = qml.device("default.qubit", wires=num_qubits)
    qubit_pairs = list(itertools.combinations(range(num_qubits), 2))

    np.random.seed(seed)
    fixed_weights = np.random.uniform(0, 2 * np.pi, size=(2, num_qubits))

    @qml.qnode(dev)
    def reservoir_circuit(features, weights):
        qml.AngleEmbedding(features, wires=range(num_qubits), rotation="Y")
        for layer in weights:
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[num_qubits - 1, 0])
            for qubit in range(num_qubits):
                qml.RX(layer[qubit], wires=qubit)

        outputs = []
        for i in range(num_qubits):
            outputs.append(qml.expval(qml.PauliX(wires=i)))
            outputs.append(qml.expval(qml.PauliY(wires=i)))
            outputs.append(qml.expval(qml.PauliZ(wires=i)))
        for pair in qubit_pairs:
            outputs.append(qml.expval(qml.PauliZ(pair[0]) @ qml.PauliZ(pair[1])))
        return outputs

    expanded = np.array(reservoir_circuit(X_scaled, fixed_weights)).T

    col_names = []
    for i in range(num_qubits):
        col_names.extend([f"Q{i}_X", f"Q{i}_Y", f"Q{i}_Z"])
    for pair in qubit_pairs:
        col_names.append(f"Joint_Z{pair[0]}_Z{pair[1]}")

    df_out = pd.DataFrame(expanded, columns=col_names)
    df_out.insert(0, "Peptide_ID", df["Peptide_ID"].values)
    df_out["Class_Label"] = df["Class_Label"].values

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"Saved {len(df_out)} 25D quantum reservoir feature rows to '{output_csv}'.")
    return df_out


if __name__ == "__main__":
    generate_angle_encoded_states()
    generate_quantum_reservoir_features()
