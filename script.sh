# Data Processing
python main.py create -p parameters/preprocessing/feature_distance.yaml
python main.py create -p parameters/preprocessing/feature_distance_1819.yaml

# Training preliminary models
python main.py grid --parameters parameters/GAE.yaml
python main.py grid --parameters parameters/GNN_2018.yaml

# Training on 2018-2019 data
python main.py grid --parameters parameters/GNN.yaml