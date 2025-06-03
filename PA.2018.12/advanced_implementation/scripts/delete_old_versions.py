from mlflow.tracking import MlflowClient

MODEL_NAME = "mine_safety_xgboost"
VERSIONS_TO_DELETE = ["1", "2", "3"]

client = MlflowClient()

for version in VERSIONS_TO_DELETE:
    try:
        print(f"Deleting model version: {MODEL_NAME} v{version}")
        client.delete_model_version(name=MODEL_NAME, version=version)
    except Exception as e:
        print(f"Failed to delete version {version}: {e}")
print("Done.")
