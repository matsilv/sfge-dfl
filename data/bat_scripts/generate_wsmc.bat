SET PYTHONPATH=.
for %%p in (1, 5, 10) do (
    python data/generation_scripts/generate_wsmc_data.py --input-dim 5 --output-dim 5 --num-sets 25 --num-products 5 --penalty %%p --degree 5 --num-instances 1000 --multiplicative-noise 0.01 --additive-noise 0.03 --seeds 0 1 2 3 4
    python data/generation_scripts/generate_wsmc_data.py --input-dim 5 --output-dim 10 --num-sets 50 --num-products 10 --penalty %%p --degree 5 --num-instances 1000 --multiplicative-noise 0.01 --additive-noise 0.03 --seeds 0 1 2 3 4
)
