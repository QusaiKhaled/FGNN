# Data Processing
python main.py create -p parameters/preprocessing/feature_distance.yaml

# Training
python main.py grid --parameters parameters/GAE.yaml
python main.py grid --parameters parameters/GNN.yaml