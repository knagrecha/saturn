# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from examples.wikitext103.models.GPTJ import get_model as getGPTJ, GPTJBlock, GPTJMLP, pretraining_loss

from examples.wikitext103.dataloaders import get_loader
from saturn.core.representations import Task, HParams
from transformers import AutoTokenizer

from examples.wikitext103.executors.FSDP import FSDPExecutor
from examples.wikitext103.executors.Pipeline import PipelineExecutor
from examples.wikitext103.executors.Spilled import SpilledExecutor
import unittest
from saturn.library import register

import torch
from functools import partial
import copy
from saturn.trial_runner.PerformanceEvaluator import search
from saturn import orchestrate



def setup_tasks():
	"""
    	Configuring hints & tokenizers for the GPT-2 & GPT-J jobs
	"""
		
	gptJTokenizer = AutoTokenizer.from_pretrained(
		"EleutherAI/gpt-j-6B")  # gpt2-xl-medium
	if gptJTokenizer.pad_token is None:
		gptJTokenizer.add_special_tokens({'pad_token': '[PAD]'})
	
	"""
		Registering execution procedures to the library
	"""
	
	classes = [FSDPExecutor, PipelineExecutor, SpilledExecutor]
	class_names = ["FSDPExecutor", "PipelineExecutor", "SpilledExecutor"]


	for c_name, c in zip(self.class_names, self.classes):
		register(c_name, c)
	
	test_tasks = []
	
	"""
		Creating the HPO sweep
	"""
	
	for lr in [1e-5]:
		test_tasks += [ Task(
							getGPTJ, # model loading function
							partial(get_loader, # dataloading function
								16 // (x + 1), split="train", tokenizer=gptJTokenizer, tok_name='gpt-j', full_data=True),
							pretraining_loss, # loss function
							hparams=HParams(lr=lr, epochs=1, optimizer_cls=torch.optim.SGD), # hparams
							hints={"is_transformer": True, "transformer_cls": {GPTJBlock, GPTJMLP}}, # arch-specific hints
							) for x in range(2)
						]
	return test_tasks

def test(test_tasks):
	search(test_tasks, log=True)
	
	for i in test_tasks:
		for gpu_count, strat in i.strategies.items():
			print("Task: {}".format(i.name))
			print("GPUs: {}".format(gpu_count))
			if strat.executor is not None:
				print("Executor: {}".format(strat.executor.name))
			else:
				print("Executor: {}".format(None))
	
	
	# learning rate does not affect perf, so skip re-searching on those
	batch_2 = copy.deepcopy(test_tasks)
	for b in batch_2:
		b.change_name()
		b.hparams.lr = 1e-3
	
	batch_3 = copy.deepcopy(test_tasks)
	for b in batch_3:
		b.change_name()
		b.hparams.lr = 3e-3
	
	test_tasks += batch_2
	test_tasks += batch_3

	# execute
	orchestrate(self.test_tasks, log=True, interval=1000)

if __name__ == '__main__':
	test_tasks = setup_tasks()
	test(test_tasks)
