#!/bin/bash
for p in 20
do
  python main.py experiments/stochastic_weights_kp/50-items/penalty-$p/comboptnet.yaml
done
for p in 5 10 20
do
  python main.py experiments/stochastic_capacity_kp/50-items/penalty-$p/comboptnet.yaml
done