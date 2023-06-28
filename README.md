# Drug Target Affinity (DTA) Prediction

**Training**: <br/>
run command of following type: <br/>
python train.py 0 0 1000 0 0 <br/>
where the parameters are:
1. dataset :- <br/>
      0: Davis <br/>
      1: KIBA <br/>
2. GPU selection
3. Number of Epochs
4. Architecture Selection :- <br/>
     0: 2 GCN Layers <br/>
     1: 2 GAT Layers <br/>
     2: 3 GCN Layers <br/>
     3: 3 GAT layers <br/>
     4: 1 GCN and 1 GAT Layer
5. Type of embedding to be used: <br/>
     0: esm1b <br/>
     1: esm2 <br/>
     2: protT5 <br/>


**Testing**: <br/>
run command of following type: <br/>
python test.py 0 0 1000 0 0 <br/>
where the parameters are:<br/>
1. dataset :- <br/>
    0: Davis <br/>
    1: KIBA  
2. GPU selection
3. Number of Epochs
4. Architecture Selection :- <br/>
     0: 2 GCN Layers <br/>
     1: 2 GAT Layers <br/>
     2: 3 GCN Layers <br/>
     3: 3 GAT layers <br/>
     4: 1 GCN and 1 GAT Layer <br/>
5. Type of embedding to be used: <br/>
     0: esm1b <br/>
     1: esm2 <br/>
     2: protT5 <br/>


