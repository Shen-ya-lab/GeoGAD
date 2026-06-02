# Curriculum Learning Enhanced Dual-Geometric Graph Anomaly Representation Learning

This is the official code for GeoGAD. 

## Environment Install

```
pip install -r requirements.txt
```

## Example: run AIDS dataset
```
python main_final.py --data aids --anom_type 0 --diff 0.0 --dim 32 --n_layers 3 --lr 1e-3 --gamma 1.0 --tau 0.8