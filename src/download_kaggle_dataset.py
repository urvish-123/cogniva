import kagglehub

DATASET_ID = "cybersimar08/face-recognition-dataset"


def main():
    path = kagglehub.dataset_download(DATASET_ID)
    print(f"Dataset downloaded to: {path}")
    print("Run:")
    print(f'python -m src.prepare_kaggle_dataset --source "{path}"')


if __name__ == "__main__":
    main()
