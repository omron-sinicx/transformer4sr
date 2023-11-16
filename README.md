# The Symbolic Transformer

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
- `train_symbolic_transformer.py` Python script used (on GPUs) for training using our synthetic datasets
- `tutorial.ipynb` Jupyter Notebook for a demonstration using our best Symbolic Transformer. **You might want to start from here!**

## Citation

```
@article{lalande2023,
  title={A Transformer Model for Symbolic Regression towards Scientific Discovery},
  author={Lalande, Florian and Matsubara, Yoshimoto and Chiba, Naoya and Taniai, Tatsunori and Igarashi, Ryo and Ushiku, Yoshitaka},
  journal={arXiv preprint arXiv:BLABLABLABLABLABLA},
  year={2023}
}
```
