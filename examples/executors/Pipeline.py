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

from torchgpipe import GPipe
import torch
from timeit import default_timer as timer
from saturn.utilities import processify
from torchgpipe.balance import balance_by_size, balance_by_time
from saturn.core.executors.Technique import BaseTechnique


class PipelineExecutor(BaseTechnique):

    def execute(task, gpu, tid, override_batch_count=None):
        """ Runs a task using pipelinig with a list of GPUs"""

        gpus = [torch.device('cuda', g) for g in gpu]
        model, loss_fn = task.get_model(fresh=True), task.loss_function
        if task.has_ckpt():
            model_ckpt = task.get_model(fresh=False)
            model.load_state_dict(model_ckpt)
            
        microbatch_count = task.selected_strategy.parameters["microbatch_count"]
        balance = task.selected_strategy.parameters["balance"]

        gpipe_model = GPipe(model, balance=balance, chunks=microbatch_count)
        iterator = task.get_iterator()
        lr, optimizer_cls = task.hparams.lr, task.hparams.optimizer_cls

        # TODO 1: allow user-specified optimizers
        optimizer = optimizer_cls(gpipe_model.parameters(), lr=lr)
        ctr = 0
        while True:
            try:
                batch, label = next(iterator)
            except:
                iterator = task.get_fresh_iterator()
                batch, label = next(iterator)
            if override_batch_count is not None and ctr == override_batch_count:
                break
            ctr += 1
            batch = batch.to(gpus[0], non_blocking=True)
            label = label.to(gpus[-1], non_blocking=True)
            out = gpipe_model(batch)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            gpipe_model.zero_grad()
            optimizer.zero_grad()
        task.save(model.state_dict())

    def search(task, gpu, tid):
        if len(gpu) < 2:
            return None, None
        try:
            model, iterator, loss_fn = task.get_model(
            ), task.get_fresh_iterator(), task.loss_function
            batch, label = next(iterator)
            lr, optimizer_cls = task.hparams.lr, task.hparams.optimizer_cls

            gpus = [torch.device('cuda', g) for g in gpu]

            """
                Rough estimates of how param counts are duplicated under different optimizers
            """
            param_scale = 2  # used to estimate CUDA consumption
            if optimizer_cls == torch.optim.SGD:
                param_scale = 2
            elif optimizer_cls == torch.optim.Adam:  # TODO: Add ADAM variants
                param_scale = 5
            elif optimizer_cls == torch.optim.Adadelta:
                param_scale = 4
            elif optimizer_cls == torch.optim.Adagrad:
                param_scale = 3
            elif optimizer_cls == torch.optim.RMSprop:
                param_scale = 4

            @processify
            def run_test(model, proposed_microbatch_count, batch, label, loss_fn, gpus, param_scale):

                balance = balance_by_time(
                    len(gpu),
                    model,
                    # Same size with mini-batch to train
                    batch,
                    # Number of micro-batches to train with GPipe
                    # chunks=proposed_microbatch_count,
                    # 4 for Adam
                    # param_scale=param_scale,
                )
                print("Balance of {}".format(balance))
                try:
                    gpipe_model = GPipe(
                        model, balance, chunks=proposed_microbatch_count)
                    optimizer = optimizer_cls(
                        gpipe_model.parameters(), lr=lr)
                    # test twice to be sure!
                    time_taken = -1
                    for _ in range(2):
                        batch = batch.to(gpus[0], non_blocking=True)
                        label = label.to(gpus[-1], non_blocking=True)
                        if _ == 1:
                            st = timer()
                        # run a test pass
                        out = gpipe_model(batch)
                        loss = loss_fn(out, label)
                        loss.backward()
                        optimizer.step()
                        if _ == 1:
                            time_taken = timer() - st
                        gpipe_model.zero_grad()
                        optimizer.zero_grad()

                except Exception as e:
                    if "memory" in str(e) or "Memory" in str(e):
                        return None, None
                    else:
                        raise e

                return time_taken, balance

            """
                Automatically choose how many chunks to use.
            """

            # microbatch size == 1
            curr_microbatch_count, proposed_microbatch_count = None, batch.shape[0]
            curr_balance, proposed_balance = None, None
            pipeline_latency, new_pipeline_latency = None, None

            while True:
                print("Evaluating pipelining on {} gpus with {} microbatches".format(
                    len(gpus), proposed_microbatch_count))

                new_pipeline_latency, proposed_balance = run_test(model, proposed_microbatch_count, batch,
                                                                  label, loss_fn, gpus, param_scale)

                if new_pipeline_latency == None:  # OOM'd. Remember that memory will increase with each cycle
                    break                         # as we reduce microbatch count

                if pipeline_latency is not None and new_pipeline_latency >= pipeline_latency:
                    break

                pipeline_latency = new_pipeline_latency
                print("{} microbatches achieved {}s latency".format(
                    proposed_microbatch_count, pipeline_latency))
                curr_microbatch_count = proposed_microbatch_count
                curr_balance = proposed_balance
                # halve the number of microbatches
                proposed_microbatch_count = proposed_microbatch_count // 2

            if curr_microbatch_count is None:
                return None, None

            else:
                return {"microbatch_count": curr_microbatch_count, "balance": curr_balance}, pipeline_latency
        except Exception as e:
            print(e)
            raise e
