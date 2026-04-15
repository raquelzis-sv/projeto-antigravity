train:
	mlflow run . -e train -P config=configs/baseline.yaml

exp2:
	mlflow run . -e train -P config=configs/experimento_hidden16.yaml

test:
	pytest tests/test_pipeline.py

ui:
	mlflow ui

clean:
	rm -rf mlruns artifacts __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +