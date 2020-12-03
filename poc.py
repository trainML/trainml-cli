import requests
import time
import json

from cognito_auth import get_tokens

tokens = get_tokens()


headers = {"Authorization": tokens.get("id_token")}

url = "https://api.trainml.dev/" "dataset/pub"
payload = dict(
    name="Test CLI Dataset",
    source_type="aws",
    source_uri="s3://trainml-examples/data/cifar10",
)
r = requests.post(url, headers=headers, data=json.dumps(payload))
print(r.status_code)

dataset = r.json()
print(dataset)

while dataset.get("status") != "ready":
    time.sleep(5)
    url = "https://api.trainml.dev/" f"dataset/pub/{dataset.get('dataset_uuid')}"
    r = requests.get(url, headers=headers)
    dataset = r.json()

print(dataset)

url = "https://api.trainml.dev/" "job"
payload = dict(
    name="Test CLI Training Job",
    type="headless",
    resources=dict(
        gpu_type_id="db18d391-dce8-44f2-9988-29d80685d250", gpu_count=1, disk_size=10
    ),
    environment=dict(
        type="DEEPLEARNING_PY37",
        env=[],
        worker_key_types=[],
    ),
    data=dict(
        datasets=[dict(dataset_uuid=dataset.get("dataset_uuid"), type="existing")],
        output_uri="s3://trainml-examples/output/resnet_cifar10",
        output_type="aws",
    ),
    model=dict(git_uri="git@github.com:trainML/test-private.git"),
    worker_count=1,
    worker_commands=[
        "PYTHONPATH=$PYTHONPATH:$TRAINML_MODEL_PATH python -m official.vision.image_classification.resnet_cifar_main --num_gpus=1 --data_dir=$TRAINML_DATA_PATH --model_dir=$TRAINML_OUTPUT_PATH --enable_checkpoint_and_export=True --train_epochs=10 --batch_size=1024"
    ],
    vpn=dict(net_prefix_type_id=1),
)
r = requests.post(url, headers=headers, data=json.dumps(payload))
print(r.status_code)

job = r.json()
print(job)