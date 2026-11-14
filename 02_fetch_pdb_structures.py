"""
02_fetch_pdb_structures.py
--------------------------
Fetches predicted 3D protein structures (.pdb) for all sequences in
`data/processed/quantum_peptide_dataset.csv` using the ESMFold REST API.
Supports automatic resume and rate-limit handling.
"""

import os
import time
import pandas as pd
import requests
from tqdm import tqdm

ESMFOLD_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
RATE_LIMIT_DELAY = 1.5
MAX_RETRIES = 3


def fetch_pdb_structures(
    dataset_csv: str = "data/processed/quantum_peptide_dataset.csv",
    output_dir: str = "data/raw/structures",
) -> None:
    df = pd.read_csv(dataset_csv)
    os.makedirs(output_dir, exist_ok=True)

    success_count = 0
    failed_ids = []

    print(f"Fetching 3D PDB structures for {len(df)} peptides into '{output_dir}'...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="ESMFold API"):
        peptide_id = str(row["Peptide_ID"])
        clean_seq = "".join(str(row["Sequence"]).split()).upper()
        file_path = os.path.join(output_dir, f"{peptide_id.lower()}.pdb")

        if os.path.exists(file_path):
            success_count += 1
            continue

        for _ in range(MAX_RETRIES):
            try:
                response = requests.post(
                    url=ESMFOLD_API_URL,
                    data=clean_seq,
                    headers={"Content-Type": "text/plain"},
                    timeout=45,
                )
                if response.status_code == 200:
                    with open(file_path, "w", encoding="utf-8") as pdb_file:
                        pdb_file.write(response.text)
                    success_count += 1
                    break
                elif response.status_code == 429:
                    time.sleep(10)
            except requests.exceptions.RequestException:
                time.sleep(2)
        else:
            failed_ids.append(peptide_id)

        time.sleep(RATE_LIMIT_DELAY)

    print(f"Completed: {success_count}/{len(df)} PDB structures available.")
    if failed_ids:
        print(f"Failed sequences ({len(failed_ids)}): {', '.join(failed_ids)}")


if __name__ == "__main__":
    fetch_pdb_structures()
