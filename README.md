[![Documentation Status](https://readthedocs.org/projects/saturn/badge/?version=latest)](https://saturn.readthedocs.io/en/latest/?badge=latest)

# Saturn: Optimized Multi-Model Deep Learning
Saturn is a novel system for multi-model deep learning training that automatically optimizes jobs for highly efficient training.
It automatically selects parallelization techniques, determines optimized resource allocations, and constructs execution schedules
for submitted jobs. Applying Saturn for hyperparameter optimization or model selection requires only a few lines of code.

Saturn is designed to support extensibility, allowing users to specify new execution procedures that can be
included in its optimization plan and search space. In this way, you can keep up with the latest
advances in model execution optimizations without having to wait for library updates & changes.

### Install Saturn

To install Saturn, please read the [instructions](INSTALL.md). We are currently working on building pre-built Docker
packages for quick start scenarios. We're always excited to hear about new use cases and details of your experience with Saturn, so feel free
to contact us at knagrech@ucsd.edu.

### Framework Support

We currently prioritize PyTorch support, but Saturn's general techniques are framework-independent. 
We would welcome contributions for TensorFlow & Jax.

## Contributing
We welcome contributions to Saturn. Areas of particular interest are an alternative solver (e.g. using reinforcement learning),
new interfaces, dashboards, and ways to support online job submissions. Please let us know if you encounter any bugs
or have any suggestions by submitting an issue.


### Documentation
[You can find the docs for Saturn here](https://saturn.readthedocs.io/en/latest/).

### Citations
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
Saturn uses Apache License 2.0


