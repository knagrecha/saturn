import torch
from typing import List
from saturn.core.representations import Task
from fairscale.experimental.nn.offload import OffloadModel
from timeit import default_timer as timer
import numpy as np
import time
import copy
from saturn.core.executors.Technique import BaseTechnique
from saturn.utilities import processify
import traceback


class SpilledExecutor(BaseTechnique):
    def search(task, gpu, tid):
        if len(gpu) > 1:
            return None, None
        try:
            max_trials = 100

            g = gpu[0]
            model, iterator, loss_fn = task.get_model(
            ), task.get_fresh_iterator(), task.loss_function
            

            lr, optimizer_cls = task.hparams.lr, task.hparams.optimizer_cls
            batch, label = next(iterator)
            g = torch.device('cuda', g)

            # we subprocess it because PyTorch CUDA alloc is messy, and doesn't cleanup fully on dels.
            # this is slow! Which makes it all the more critical we address
            # todo item 2.
            @processify
            def run_test(model, gpu, partition_count, test_batch, test_label):
                try:
                    # TODO: optimize microbatch count selection
                    test_model = OffloadModel(
                        model=copy.deepcopy(model), device=gpu, offload_device=torch.device("cpu"), num_slices=partition_count, checkpoint_activation=True, num_microbatches=1)

                    optimizer = optimizer_cls(test_model.parameters(), lr=lr)
                    # Test memory bounds
                    oom = False
                    try:
                        # test twice to be sure!
                        time_taken = -1
                        for _ in range(2):
                            inner_batch = test_batch.to(gpu)
                            inner_label = test_label.to(gpu)
                            if _ == 1:
                                st = timer()
                            out = test_model(inner_batch)
                            loss = loss_fn(out, inner_label)
                            loss.backward()
                            optimizer.step()
                            test_model.zero_grad()
                            optimizer.zero_grad()
                            if _ == 1:
                                time_taken = timer() - st
                    except Exception as e:
                        if "memory" in str(e):
                            print("OOM!")
                            oom = True
                            model.cpu()
                        else:
                            raise e
                    if oom:
                        del test_model
                        try:
                            del out
                        except:
                            pass
                        try:
                            del loss
                        except:
                            pass
                        del inner_batch
                        del inner_label
                        torch.cuda.empty_cache()
                    return not oom, time_taken
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    raise e

            total_layers = len(list(model.children()))

            def factorize(x):
                return [i for i in range(1, x+1) if x % i == 0]

            test_partitions = factorize(total_layers)

            #step = max(1, total_layers // max_trials)
            # test_partitions = np.arange(
            #    1, total_layers + step, step)
            test_model = copy.deepcopy(model)
            for trial in test_partitions:
                success, exec_time = run_test(model=test_model, gpu=g,
                                              partition_count=trial, test_batch=batch, test_label=label)
                if success:
                    return {"partition_count": trial}, exec_time

            return None, None
        except Exception as e:
            print(e)
            raise e

    @processify
    def execute(task, gpu, tid, override_batch_count=None):
        try:
            task.setup()
            """ Runs a task using spilling with a list of GPUs"""

            gpu = torch.device('cuda', gpu[0])
            model, loss_fn = task.get_model(fresh=True), task.loss_function
            if task.has_ckpt():
                model_ckpt = task.get_model(fresh=False)
                model.load_state_dict(model_ckpt)

            partition_count = task.selected_strategy.parameters["partition_count"]
            offloadmodel = OffloadModel(
                model=model, device=gpu, offload_device=torch.device("cpu"), num_slices=partition_count, checkpoint_activation=True, num_microbatches=1)

            lr, optimizer_cls = task.hparams.lr, task.hparams.optimizer_cls

            optimizer = optimizer_cls(model.parameters(), lr=lr)
            iterator = task.get_iterator()

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
                out = offloadmodel(batch.to(gpu))
                loss = loss_fn(out, label.to(gpu))
                loss.backward()
                optimizer.step()
                offloadmodel.zero_grad()
                optimizer.zero_grad()

        except Exception as e:
            print(e)
            raise e
        task.save(model)
