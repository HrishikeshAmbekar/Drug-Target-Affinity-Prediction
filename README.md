# Drug Target Affinity (DTA) Prediction

**Training**:
run command of following type:
python train.py 0 0 1000 0 0
where the parameters are:
1. dataset :-
    0: Davis
    1: KIBA  
2. GPU selection
3. Number of Epochs
4. Architecture Selection :-
     0: 2 GCN Layers
     1: 2 GAT Layers
     2: 3 GCN Layers
     3: 3 GAT layers
     4: 1 GCN and 1 GAT Layer
5. Type of embedding to be used:
     0: esm1b
     1: esm2
     2: protT5


**Testing**:
run command of following type:
python test.py 0 0 1000 0 0
where the parameters are:
1. dataset :-
    0: Davis
    1: KIBA  
2. GPU selection
3. Number of Epochs
4. Architecture Selection :-
     0: 2 GCN Layers
     1: 2 GAT Layers
     2: 3 GCN Layers
     3: 3 GAT layers
     4: 1 GCN and 1 GAT Layer
5. Type of embedding to be used:
     0: esm1b
     1: esm2
     2: protT5


