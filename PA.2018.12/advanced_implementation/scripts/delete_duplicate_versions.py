from mlflow.tracking import MlflowClient

MODEL_NAME = "mine_safety_xgboost"
# List of (version, metrics) tuples for deduplication
# We'll keep the first occurrence of each unique (test_rmse, test_mae, test_ll)

client = MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}'")

seen = set()
to_delete = []
for v in versions:
    metrics = (v.run_id, v.current_stage, v.status)
    # Use metrics for deduplication
    run = client.get_run(v.run_id)
    test_rmse = run.data.metrics.get("test_rmse")
    test_mae = run.data.metrics.get("test_mae")
    test_ll = run.data.metrics.get("test_ll")
    key = (test_rmse, test_mae, test_ll)
    if key in seen:
        to_delete.append(v.version)
    else:
        seen.add(key)

for version in to_delete:
    print(f"Deleting duplicate model version: {version}")
    client.delete_model_version(name=MODEL_NAME, version=str(version))
print("Done.")
