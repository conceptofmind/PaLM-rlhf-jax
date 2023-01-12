<img src="./palm.gif" width="450px"></img>

## PaLM + RLHF - JAX (WIP)
Implementation of the Transformer architecture from <a href="https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html">PaLM - Scaling Language Modeling with Pathways</a> - in Jax with RLHF (Reinforcement Learning with Human Feedback). Using the [Flax](https://github.com/google/flax) framework.

## Acknowledgement:
Built in collaboration with [Phil 'Lucidrains' Wang](https://github.com/lucidrains). Please check out his [open-source implementations](https://github.com/lucidrains) of multiple different transformer architectures and [support](https://github.com/sponsors/lucidrains) his work.

### Looking for Data Engineers familiar with JAX and training on TPUs

## FAQ

- Does this contain a model for inference?

There is no trained model. I am part of the TRC program so I will be looking into training a small model.

## Community

<a href="https://carper.ai/">CarperAI</a> had been working on <a href="https://github.com/CarperAI/trlx">an RLHF framework</a> for large language models for many months prior to the release of ChatGPT.

## Usage

The model does not require `vmap` to train.

```python
import jax
import numpy as np
key = jax.random.PRNGKey(0)
seq = jax.random.randint(key, (1, 1024), 0, 20000)
model = PaLM(
    num_tokens = 20000,
    dim = 512,
    depth = 12,
    heads = 8,
    dim_head = 64
)
init_rngs = {'params': jax.random.PRNGKey(1), 
            'dropout': jax.random.PRNGKey(2)}
params = model.init(init_rngs, seq)
output = model.apply(params, seq)
print(output.shape) # (1, 1024, 20000)
n_params_flax = sum(
    jax.tree_leaves(jax.tree_map(lambda x: np.prod(x.shape), params))
)
print(f"Number of parameters in Flax model: {n_params_flax}") # 55073280
```

## TODO:
- [X] Finish building base model architecture
- [ ] Add XPOS
- [ ] Add LoRA
- [ ] Add Encoder-Decoder
- [ ] Add Trainer
- [ ] Add Enwiki8


## Author:
- Enrico Shippole

## Citations:
```bibtex
@inproceedings{Chowdhery2022PaLMSL,
  title   = {PaLM: Scaling Language Modeling with Pathways},
  author  = {Aakanksha Chowdhery and Sharan Narang and Jacob Devlin and Maarten Bosma and Gaurav Mishra and Adam Roberts and Paul Barham and Hyung Won Chung and Charles Sutton and Sebastian Gehrmann and Parker Schuh and Kensen Shi and Sasha Tsvyashchenko and Joshua Maynez and Abhishek Rao and Parker Barnes and Yi Tay and Noam M. Shazeer and Vinodkumar Prabhakaran and Emily Reif and Nan Du and Benton C. Hutchinson and Reiner Pope and James Bradbury and Jacob Austin and Michael Isard and Guy Gur-Ari and Pengcheng Yin and Toju Duke and Anselm Levskaya and Sanjay Ghemawat and Sunipa Dev and Henryk Michalewski and Xavier Garc{\'i}a and Vedant Misra and Kevin Robinson and Liam Fedus and Denny Zhou and Daphne Ippolito and David Luan and Hyeontaek Lim and Barret Zoph and Alexander Spiridonov and Ryan Sepassi and David Dohan and Shivani Agrawal and Mark Omernick and Andrew M. Dai and Thanumalayan Sankaranarayana Pillai and Marie Pellat and Aitor Lewkowycz and Erica Oliveira Moreira and Rewon Child and Oleksandr Polozov and Katherine Lee and Zongwei Zhou and Xuezhi Wang and Brennan Saeta and Mark Diaz and Orhan Firat and Michele Catasta and Jason Wei and Kathleen S. Meier-Hellstern and Douglas Eck and Jeff Dean and Slav Petrov and Noah Fiedel},
  year    = {2022}
}
```

```bibtex
@software{flax2020github,
  author = {Jonathan Heek and Anselm Levskaya and Avital Oliver and Marvin Ritter and Bertrand Rondepierre and Andreas Steiner and Marc van {Z}ee},
  title = {{F}lax: A neural network library and ecosystem for {JAX}},
  url = {http://github.com/google/flax},
  version = {0.5.0},
  year = {2020},
}
```

```bibtex
@article{Stiennon2020LearningTS,
    title   = {Learning to summarize from human feedback},
    author  = {Nisan Stiennon and Long Ouyang and Jeff Wu and Daniel M. Ziegler and Ryan J. Lowe and Chelsea Voss and Alec Radford and Dario Amodei and Paul Christiano},
    journal = {ArXiv},
    year    = {2020},
    volume  = {abs/2009.01325}
}
```

```bibtex
@article{Hu2021LoRALA,
    title   = {LoRA: Low-Rank Adaptation of Large Language Models},
    author  = {Edward J. Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Weizhu Chen},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2106.09685}
}
```

```bibtex
@inproceedings{Sun2022ALT,
  title     = {A Length-Extrapolatable Transformer},
  author    = {Yutao Sun and Li Dong and Barun Patra and Shuming Ma and Shaohan Huang and Alon Benhaim and Vishrav Chaudhary and Xia Song and Furu Wei},
  year      = {2022}
}
```