## 🛠 Installation

```bash
conda create --name protenix python=3.11
pip3 install protenix
```

## 🚀 Inference

```bash
wget "https://zenodo.org/records/17541122/files/Covid-design-10.pt?download=1" -O outputs/Covid-design-10.pt
conda activate protenix
bash predict.py
```