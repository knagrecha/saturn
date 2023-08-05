# Fine-Tuning HuggingFace Models on WikiText Using Saturn
## Prerequisites and Setup

- Your machine must have at least one GPU.
- We assume PyTorch version 2.0.

All other dependencies & requirements for running the WikiText example can be found in 
the [requirements](./requirements.txt) file.

## Summary
In this example, we'll run hyperparameter optimization on a batch of GPT-J models
for the WikiText-2 dataset.

To run the fine-tuning script, launch ``python3 WikiText2.py``.


## Walkthrough

### Specifications

In the first section of the code, we specify the training tasks.

We define the tokenizers using the HuggingFace Transformers library.

We also define model-loading functions, also from the HuggingFace library.
These can be found in our [models directory](./models/).

We also define functions to produce PyTorch dataloaders in the 
[dataloaders directory](./dataloaders).

We also use some "hints" which can be used by the user-defined executors to
guide execution. For example, we specify that the submitted jobs are
Transformer architectures. That allows us to guide the FSDP executor to use
the [Transformer auto-wrap policy](https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html).

Our implementations of some user-defined executors --- [FSDP](https://engineering.fb.com/2021/07/15/open-source/fsdp/), 
[TorchGPipe](https://github.com/kakaobrain/torchgpipe), and 
[offloading](https://github.com/facebookresearch/fairscale) are defined in the 
[executors directory](./executors). We register these implemnetations in our example code.

### Launching

The HPO sweep is defined by assigning different learning rates and batch sizes for each task.
We invoke Saturn's "search" function to generate performance estimates for each job.

To save time, we avoid repeated searches for different learning rates (since it doesn't affect performance).
Instead, we just copy over the tasks after the search and change their identifiers.

Finally, we invoke the "orchestrate" function to launch the jobs and being training. 
The models will be saved into the ".models" directory for further use and evaluation.
