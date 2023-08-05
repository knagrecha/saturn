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

from examples.wikitext2.models.GPTJ import get_model as getGPTJ, GPTJBlock, GPTJMLP, pretraining_loss

from examples.wikitext2.dataloaders import get_loader
from saturn.core.representations import Task, HParams
from transformers import AutoTokenizer

from examples.wikitext2.executors.FSDP import FSDPExecutor
from examples.wikitext2.executors.Pipeline import PipelineExecutor
from examples.wikitext2.executors.Spilled import SpilledExecutor
import unittest
from saturn.library import register

import torch
from functools import partial
import copy
from saturn.trial_runner.PerformanceEvaluator import search
from saturn import orchestrate


class TestPerformanceEvaluator(unittest.TestCase):
	def setUp(self):
		
		"""
            Configuring hints & tokenizers for the GPT-2 & GPT-J jobs
        """

		
		gptjhints = {"is_transformer": True,
		             "transformer_cls": {GPTJBlock, GPTJMLP}}
		
		
		gptJTokenizer = AutoTokenizer.from_pretrained(
			"EleutherAI/gpt-j-6B")  # gpt2-xl-medium
		if gptJTokenizer.pad_token is None:
			gptJTokenizer.add_special_tokens({'pad_token': '[PAD]'})
		
		
		
		self.test_tasks = []
		
		"""
			Creating the HPO sweep
		"""
		
		for lr in [1e-5]:
			hparams = HParams(lr=lr, batch_count=100, optimizer_cls=torch.optim.SGD)
			self.test_tasks += [
				Task(
					getGPTJ,
					partial(get_loader,
					        32 // (x + 1),
					        split="train",
					        tokenizer=gptJTokenizer,
					        tok_name='gpt2-xl'),
					pretraining_loss,
					hparams=copy.deepcopy(hparams),
					hints=copy.deepcopy(gptjhints),
					gpu_range=[4, 8]
				) for x in range(1)
			]
		
		"""
			Registering execution procedures to the library
		"""
		
		self.classes = [FSDPExecutor, PipelineExecutor, SpilledExecutor]
		self.class_names = ["FSDPExecutor",
		                    "PipelineExecutor", "SpilledExecutor"]
		
		for c_name, c in zip(self.class_names, self.classes):
			register(c_name, c)
	
	def test(self):
		
		# Launch performance evaluation search
		search(self.test_tasks, log=True)
		
		# Copy the job searches
		batch_2 = copy.deepcopy(self.test_tasks)
		
		for b in batch_2:
			b.hparams.lr = 1e-3
			b.change_name()
		
		batch_3 = copy.deepcopy(self.test_tasks)
		for b in batch_3:
			b.hparams.lr = 3e-3
			b.change_name()
		
		self.test_tasks += batch_2
		self.test_tasks += batch_3
		
		# invoke execution
		orchestrate(self.test_tasks, log=True)


if __name__ == '__main__':
	unittest.main()
