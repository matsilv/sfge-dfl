SET PYTHONPATH=.
for %%p in (5, 10, 20) do (
    python data/generation_scripts/generate_stochastic_weights_kp_data.py --input-dim 5 --output-dim 50 --relative-capacity 0.2 --degree 5 --num-instances 1000 --multiplicative-noise 0.1 --additive-noise 0.03 --correlate-values-weights 1 --penalty %%p --seed 0 1 2 3 4
)
