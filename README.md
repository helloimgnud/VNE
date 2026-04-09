how to run :

pip install -r requirements.txt

python scripts/generate_datasets.py --experiments fig6
python src/scripts/run_experiments.py --experiments fig6
python src/scripts/run_experiments.py --plot-only --experiments fig6

================ NEW NEW NEW ================

python -m src.scripts.run_experiments --experiments fig6

python -m src.scripts.generate_datasets --experiments fig6 --num-replicas 5 --substrate-nodes "50,100" --num-vnrs "150,300" --force 

python -m src.agents.train_scheduler.py --episodes 500 --K_max 20 --save_path model.pt