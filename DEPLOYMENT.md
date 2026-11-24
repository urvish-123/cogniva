# Deploy Cogniva on Render

This repository can be deployed as one Docker web service. FastAPI serves the React
application, API, and health endpoint on the same HTTPS URL.

## Requirements

- A GitHub, GitLab, or Bitbucket repository containing this project
- A Render account
- A paid Render web-service plan with at least 4 GB RAM
- A 10 GB Render persistent disk

The ML models, ONNX export, and SQLite database use more resources than Render's
free web-service tier normally provides. The included `render.yaml` uses the
Starter web-service plan and a persistent disk.

## 1. Push the project to Git

From the project folder:

```powershell
git add .
git commit -m "Add Render deployment configuration"
git push origin main
```

Use the branch name that is configured as the default branch, if it is not `main`.

## 2. Create the Render Blueprint

1. Sign in to Render.
2. Open **New > Blueprint**.
3. Connect the Cogniva Git repository.
4. Select the repository and branch.
5. Render will read `render.yaml`.
6. Review the service and disk, then choose **Apply**.

If Blueprint deployment is not used, create a Docker web service manually:

1. Open **New > Web Service**.
2. Connect the repository.
3. Select **Docker** as the runtime.
4. Set the service name and region.
5. Select the Starter or a larger plan.
6. Set the health-check path to `/health`.
7. Add a persistent disk mounted at `/var/data`, sized at least 10 GB.

## 3. Configure environment variables

The Blueprint configures these automatically:

| Variable | Value |
|---|---|
| `CELEBRITY_DB_PATH` | `/var/data/database/celebrity_embeddings.db` |
| `INSIGHTFACE_HOME` | `/var/data/.insightface` |
| `HF_HOME` | `/var/data/.cache/huggingface` |
| `TRANSFORMERS_CACHE` | `/var/data/.cache/huggingface` |
| `PYTHONUNBUFFERED` | `1` |

For a custom frontend domain, set:

```text
CORS_ALLOWED_ORIGINS=https://app.example.com
```

The same Render HTTPS origin does not need to be added because FastAPI serves
the React frontend and API together.

## 4. Deploy

Use **Deploy latest commit** or push another commit to the configured branch.
The Docker build will:

1. Build the React/Vite frontend.
2. Install the Python dependencies.
3. Copy the frontend build into the image.
4. Start FastAPI on Render's assigned port.

At the first application start, FastAPI downloads or prepares:

- InsightFace/ArcFace models
- the CLIP processor and model
- the CLIP ONNX image model

The downloads are stored on the persistent disk. The first startup can take
several minutes.

## 5. Verify deployment

Open the Render URL shown in the dashboard.

Verify the API health endpoint:

```text
https://YOUR-SERVICE.onrender.com/health
```

The response should contain status `ok`.

The following routes are available from the same origin:

```text
/                         React application
/health                   Health check
/docs                     FastAPI API documentation
/recognize                Image recognition endpoint
```

Test recognition:

```bash
curl -X POST \
  -F "file=@test_images/example.jpg" \
  "https://YOUR-SERVICE.onrender.com/recognize"
```

## Persistence

The SQLite database is copied from
`database/celebrity_embeddings.db` to the persistent disk only when
`/var/data/database/celebrity_embeddings.db` does not exist.

After deployment:

- Database changes and generated patient embeddings remain on the disk.
- Rebuilding or replacing the Docker image does not erase the disk.
- Deleting or disconnecting the disk removes that persistent data.
- The database can be updated locally and then re-deployed using a database
  migration or replacement process.

## Updating the application

Commit and push changes to the configured branch:

```powershell
git add .
git commit -m "Update Cogniva"
git push origin main
```

Render deploys the latest commit automatically.

## Troubleshooting

### Service has no response or sleeps

A free or lower-cost service may sleep or have insufficient memory. For
production use, select the smallest Starter-compatible plan that provides at
least 4 GB RAM and keep the persistent disk attached.

### Out of memory during startup

InsightFace and CLIP load in the application lifespan. Use at least 4 GB RAM,
and use a larger plan if startup or recognition remains unstable.

### Model download repeats after deployment

Confirm `/var/data/.insightface` and `/var/data/.cache/huggingface` are on
the attached persistent disk and that `HF_HOME` and `INSIGHTFACE_HOME` are set.

### Database is empty

Confirm `CELEBRITY_DB_PATH` is set to `/var/data/database/celebrity_embeddings.db`.
The Docker image includes the bundled database and copies it to the data disk
on the first startup.

### Frontend displays but uploads fail

Check `/health` and `/docs`, review the Render service logs, and confirm that
the deployment instance has sufficient memory.

## Local Docker test

Before pushing, build and run the complete application locally:

```powershell
docker build -t cogniva .
docker run --rm -p 10000:10000 -e PORT=10000 cogniva
```

Open `http://localhost:10000`.
