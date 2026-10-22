# MovieLens 25M NCF training in Google Colab

This handoff trains an isolated candidate using the project's NeuMF model,
streaming shards, early-stopping trainer, and `ModelStore` artifact format.
It does **not** promote the result or replace `collaborative_ncf`. Full training
starts only after changing the notebook's explicit opt-in and entering the
confirmation phrase.

## Safe export on Windows

Run these commands from the repository root in PowerShell. They copy only the
candidate shard directory out of the Docker volume; the production model,
database, `.env`, and the rest of the volume are not exported.

### Option A: From a running Docker API container
```powershell
$stage = Join-Path $env:TEMP ("CineMind-NCF-Colab-{0}" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
New-Item -ItemType Directory -Path $stage -Force | Out-Null
docker cp cinemind-api:/app/data/ncf_stream_25m_candidate_20261011 "$stage\candidate"
if ($LASTEXITCODE -ne 0) { throw "Could not copy candidate shards from the API container" }
$export = Join-Path $stage "cinemind_ncf_candidate_20261011.zip"
& .\backend\.venv\Scripts\python.exe training\colab_ncf\export_candidate.py `
  --candidate-dir "$stage\candidate" `
  --source-root backend `
  --output $export
```

### Option B: Pre-verified candidate archive
The candidate dataset (101 train shards, 101 validation shards, 24,997,664 interaction rows, fingerprint `f946c4a9ffcc65c63ed0546223dd726eef5035523ec3a2d23685f852913f08f5`) has already been validated and packaged to:
`$env:TEMP\cinemind_ncf_candidate_20261011.zip`
(Size: ~290.3 MB / 304,383,440 bytes).

The exporter uses a strict allow-list: the candidate `metadata.json`, numbered
train/validation NPZ shards, the NCF model/data/streaming/evaluation and
`ModelStore` source modules, plus the candidate-training helper. It validates
the mappings, preprocessing metadata, shard arrays, row totals, file hashes,
and shard numbering. It refuses to overwrite an existing archive. It does not
read `.env`, app/database data, production weights, `.venv`, `node_modules`, or
other directories. Keep the ZIP in a private location; it contains user/item
index mappings and training interactions.

## Colab steps

1. Create a new Colab notebook by uploading `training/colab_ncf/CineMind_NCF_Colab.ipynb`
   to [Google Colab](https://colab.research.google.com/) and select
   **Runtime → Change runtime type → T4 GPU** (or another CUDA GPU).
2. **Bundle placement (choose one):**
   - **Mode A (Drive-free, recommended if Drive credentials fail):** Leave `USE_DRIVE = False` in Cell 1. Upload `cinemind_ncf_candidate_20261011.zip` directly to the `/content/` directory using Colab's left-side file browser. This avoids Google Drive permission prompts and eliminates the `MessageError: credential propagation was unsuccessful` error.
   - **Mode B (Google Drive):** Set `USE_DRIVE = True` in Cell 1. Copy `cinemind_ncf_candidate_20261011.zip` to your Google Drive under `MyDrive/CineMind-NCF/`.
3. Run cells in order. The GPU check stops immediately if
   `torch.cuda.is_available()` is false and prints the selected device when it
   succeeds. Bundle CRC and SHA-256 checks run before extraction.
4. Run the short benchmark cell. It times real train and validation batches from the
   candidate shards and reports estimated per-epoch, configured-run, and
   earliest-early-stop runtimes. Treat this as an estimate, not a completion
   promise; Colab GPU assignment and I/O vary.
5. Review the benchmark and storage availability. Set `RUN_FULL_TRAINING = True`
   in the guarded cell and execute it. The cell additionally requires typing
   `TRAIN CANDIDATE`. Leave it false to benchmark/inspect without training.
6. **Resuming after an interruption:**
   - **Crucial note:** Local storage under `/content` is ephemeral and is completely erased whenever Colab disconnects, idles out, or resets the runtime environment. It does not persist on its own.
   - Checkpoints are saved atomically at every completed epoch boundary to `RUN_DIR/checkpoints/candidate_epoch.pt`.
   - **Mode A (Drive-free):** After each completed epoch, use the provided helper function `download_checkpoint()` to download `candidate_epoch.pt` to your local computer before the runtime disconnects. To resume in a new session after a disconnect:
     1. Connect to a fresh T4 GPU runtime.
     2. Re-upload `cinemind_ncf_candidate_20261011.zip` to `/content/`.
     3. Create `/content/cinemind_run/checkpoints/` and upload your saved `candidate_epoch.pt` there.
     4. Set `RESUME = True` in the training cell and ensure identical settings/bundle.
   - **Mode B (Google Drive):** Checkpoints are saved directly to Google Drive and survive session disconnections automatically.
   - When `RESUME = True`, the trainer verifies the dataset fingerprint, model, and optimizer configurations, and resumes from `start_epoch + 1`. If a checkpoint exists and `RESUME = False`, training raises `FileExistsError` to prevent accidentally overwriting or restarting from epoch 1.
7. The script writes only
   `models/collaborative_ncf_candidate_20261011/`, verifies that it loads using
   the existing `ModelStore` implementation, checks mapping sizes, and performs
   a finite inference probe. It refuses to overwrite an existing artifact or
   report. Download the `training_report.json` and candidate directory using
   `download_candidate_results()` for review; do not copy it over the production artifact.

## Training and evaluation details

- Model defaults match the current backend configuration: NeuMF dimensions
  `GMF=64`, MLP `[128, 64, 32, 16]`, dropout `0.2`, explicit normalized labels,
  Adam learning rate `0.001`, weight decay `1e-5`, batch size `1024`, maximum
  20 epochs, early-stop patience 3, seed 42.
- The project streaming trainer evaluates held-out rows each epoch and records
  validation RMSE, MAE, train loss, and cumulative elapsed seconds. The
  report selects the best validation-RMSE epoch. These are point-wise rating
  metrics, not ranking metrics.
- This training uses the candidate's existing 90/10 split and trained
  CineMind-movie mappings. It does not rebuild or modify the CSV, source shards,
  database, or production artifact.
- Creating an artifact and a successful inference probe do not by themselves
  establish production readiness. Review validation metrics, mapping coverage,
  recommendation quality, and API compatibility before any separately
  authorized promotion.

No full training has been started by preparing this handoff.
