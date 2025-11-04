import pandas as pd
from tqdm import tqdm
import os,json

os.makedirs('./data/Covid/Covid_design',exist_ok=True)

df = pd.read_csv('./data/Covid/Covid_design.csv')

RBD = 'RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNK'

for i,(hseq,lseq) in tqdm(enumerate(df.values)):
    sequences = [{"proteinChain": {
                    "sequence": RBD,
                    "count": 1
                }},
                {"proteinChain": {
                    "sequence": hseq,
                    "count": 1
                }},
                {"proteinChain": {
                    "sequence": lseq,
                    "count": 1
                }}]
    info = [{'sequences':sequences,'name':f'idx_{i}'}]
    with open(f'./data/Covid/Covid_design/idx_{i}.json','w') as f:
        json.dump(info,f)