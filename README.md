# A Transformer Model for Symbolic regression towards Scientific Discovery

This GitHub repository provides additional details and the code for our paper accepted at the 2023 NeurIPS Workshop on AI for Scientific Discovey: From Theory to Practice.  
The paper can be accessed here:  
[A Transformer Model for Symbolic Regression towards Scientific Discovery (OpenReview)](https://openreview.net/forum?id=AIfqWNHKjo)

## Contents of this repository

- `best_model_weights/` directory with the weights of our pretained models
- `datasets/` directory code to generate the training datasets
- `model/` directory with the Symbolic Transformer ANN architecture
- `.gitignore`
- `README.md` this file
- `evaluate_model.py` Python script used when testing our best Symbolic Transformer using the [SRSD datasets](https://huggingface.co/papers/2206.10540)
- `requirements.txt` Dependencies
- `train_symbolic_transformer.py` Python script used (on GPUs) for training using our synthetic datasets
- `tutorial.ipynb` Jupyter Notebook for a demonstration using our best Symbolic Transformer. **You might want to start from here!**

## Setup

Install dependencies (we used Python 3.11.3)  
``
pip install -r requirements.txt
``

## Citation

```
@inproceedings{lalande2023,
  title = {A Transformer Model for Symbolic Regression towards Scientific Discovery},
  author = {Florian Lalande and Yoshitomo Matsubara and Naoya Chiba and Tatsunori Taniai and Ryo Igarashi and Yoshitaka Ushiku},
  booktitle = {NeurIPS 2023 AI for Science Workshop},
  year = {2023},
  url = {https://openreview.net/forum?id=AIfqWNHKjo},
}
```
