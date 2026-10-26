# Patient dataset

One folder per patient ID, one subfolder per person, at least 1 clear
single-face photo each (3-10 recommended for reliable matching):

```text
dataset/Patient/
├── P001/
│   ├── Son_of_amitab/    1.jpg, 2.jpg, ...
│   └── Wife_of_amitab/   1.jpg, 2.jpg, ...
└── P002/
    └── ...
```

The patient folder name (`P001`) is the `patient_id` (case-insensitive:
`p001` finds `P001`). The person folder name is the name shown, with
underscores as spaces (`Son_of_amitab` -> "Son of amitab").

`POST /recognize?patient_id=P001` compares the face with
P001's people **only**: the best match at or above
`PATIENT_SIMILARITY_THRESHOLD` (0.45) is returned as `type: patient`,
otherwise `type: unknown`. Celebrities and objects are never searched.
An unknown patient ID returns HTTP 404. Without `patient_id` the normal
celebrity/object flow runs and patients are never searched.

Indexing is automatic and incremental. The API indexes every patient at
startup and keeps it in memory. When photos are added to, removed from, or
renamed in a person folder, the next request re-embeds only that person.
Replacing a photo in place under the same file name is not detected: save it
under a new name, or rebuild:

```bash
python -m src.generate_patient_centroids                  # all patients
python -m src.generate_patient_centroids --patient P001  # one patient
python -m src.generate_patient_centroids --rebuild        # embed everything again
```
