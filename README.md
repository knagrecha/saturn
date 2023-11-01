[![Documentation Status](https://readthedocs.org/projects/saturn/badge/?version=latest)](https://saturn.readthedocs.io/en/latest/?badge=latest)

# Saturn: Optimized Training of Multiple Large Deep Learning Models
Saturn is a novel system for multi-model deep learning training that automatically optimizes jobs for highly efficient training.
It automatically selects parallelization techniques, determines optimized resource allocations, and constructs execution schedules
for submitted jobs. Applying Saturn for hyperparameter optimization or model selection requires only a few lines of code.

![Hydra_Summary_Figure (1)](https://github.com/knagrecha/saturn/assets/32966638/ecd6742e-1f33-4d76-a9da-e7c57bb9ad1f)

Saturn is designed to support extensibility, allowing users to specify new execution procedures that can be
included in its optimization plan and search space. In this way, you can keep up with the latest
advances in model execution optimizations without having to wait for library updates & changes.

![Hydra Figures (1)](https://github.com/knagrecha/saturn/assets/32966638/ef1f5787-0eb6-482b-849c-1d778b8c7488)


### Install Saturn

To install Saturn, please read the [instructions](INSTALL.md). We're always excited to hear about new use cases and details of your experience with Saturn, so feel free
to contact us at knagrech@ucsd.edu if you want to share news.

### Framework Support

We currently prioritize PyTorch support, but Saturn's general techniques are framework-independent. 
We would welcome contributions for TensorFlow & Jax.

## Contributing
We welcome contributions to Saturn. Areas of particular interest are an alternative solver (e.g. using reinforcement learning),
new interfaces, dashboards, and ways to support online job submissions. Please let us know if you encounter any bugs
or have any suggestions by submitting an issue.

You can join the Slack here: https://join.slack.com/t/saturn-dl/shared_invite/zt-267mfi3s4-ifUYLiJUtaVeGFcYe9vbxA or by scanning this QR code:

<img width="290" alt="slack" src="https://github.com/ollie-robin/saturn/assets/105469320/bcc1421c-1aec-486f-8520-cfccffb9f3da">



## Documentation
[You can find the docs for Saturn here](https://saturn.readthedocs.io/en/latest/).

## How to Cite this Work
If you use this system in an academic work, please cite our [tech report](https://adalabucsd.github.io/papers/TR_2023_Saturn.pdf) as follows.
```
@article{nagrechasaturn,
  title={Saturn: An Optimized Data System for Multi-Large-Model Deep Learning Workloads (Information System Architectures)},
  author={Nagrecha, Kabir and Kumar, Arun}
}
```
### The Team
Saturn is currently developed and maintained by Kabir Nagrecha at UCSD.

### License
Saturn uses Apache License 2.0.


