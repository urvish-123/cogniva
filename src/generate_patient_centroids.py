"""
Build the patient index (`patient_people` table) ahead of time.

Photos: dataset/Patient/<PatientID>/<Person>/*.jpg
        e.g. dataset/Patient/P001/Son_of_amitab/1.jpg

Optional: the API indexes every patient at startup and re-indexes changed
person folders on the next request, embedding only what changed. Use
--rebuild to embed everything again (e.g. after replacing photos in place).

python -m src.generate_patient_centroids                    # every patient
python -m src.generate_patient_centroids --patient P001    # one patient
python -m src.generate_patient_centroids --rebuild          # from scratch
"""

import argparse

from .config import PATIENT_DATASET_DIR
from .patient import (
    patient_folder,
    patient_key,
    load_all_patients,
    sync_patient,
)
from .database import delete_patient_people, init_database


def main(patient=None, rebuild=False):
    init_database()

    if patient:
        folder = patient_folder(patient)
        if folder is None:
            print(f"No folder for patient {patient} in {PATIENT_DATASET_DIR}")
            return
        key = patient_key(patient)
        if rebuild:
            delete_patient_people(key)
        total = len(sync_patient(key, folder, verbose=True))
    else:
        if rebuild:
            delete_patient_people()  # also drops patients whose folder is gone
        total = load_all_patients(verbose=True)

    print(f"Patient index ready for {total} people.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default=None,
                        help="Index only this patient ID (e.g. P001).")
    parser.add_argument("--rebuild", action="store_true",
                        help="Embed every photo again instead of only changes.")
    args = parser.parse_args()
    main(args.patient, args.rebuild)
