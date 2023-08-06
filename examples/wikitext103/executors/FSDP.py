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

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload, FullStateDictConfig, StateDictType
)
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    offload_wrapper,
    checkpoint_wrapper,
    apply_activation_checkpointing)

import copy
from timeit import default_timer as timer
import saturn.core.executors.multiprocessing.my_multiprocessing as mp
import multiprocessing as true_mp
import os
import functools
import torch.distributed as dist
from torch.distributed.fsdp.wrap import (
    always_wrap_policy,
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from saturn.core.representations import Task, Strategy, Techniques
from saturn.core.executors.Technique import BaseTechnique
from typing import List
import ray
import warnings


def setup(rank, world_size, port):
    warnings.filterwarnings("ignore")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(12000 + port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class FSDPExecutor(BaseTechnique):
    name = "FSDP"
    trial_batch_count = 2


    def execute(task: Task, gpu: List[int], tid, override_batch_count=None):
        WORLD_SIZE = len(gpu)
        mp.spawn(FSDPExecutor.inner_fsdp_execution, args=(
            WORLD_SIZE, task, gpu, tid, override_batch_count), nprocs=WORLD_SIZE, join=True)

    def search(task: Task, gpu: List[int], tid):
        if len(gpu) < 2:
            return None, None
        try:
            warnings.filterwarnings("ignore")
            parameters_list = [
                # {"checkpoint": False, "offload": False},
                {"checkpoint": True, "offload": False},
                {"checkpoint": False, "offload": True},
                {"checkpoint": True, "offload": True}

            ]

            ctx = true_mp.get_context("spawn")
            q = ctx.Queue()

            WORLD_SIZE = len(gpu)
            m = task.get_model()
            for param in parameters_list:
                oom = False
                try:
                    mp.spawn(FSDPExecutor.trial, args=(
                        WORLD_SIZE, copy.deepcopy(m), task, gpu, param, q, tid), nprocs=WORLD_SIZE, join=True)

                    rt = q.get()
                except Exception as e:
                    oom = True

                if not oom:
                    return param, rt

            return None, None
        except Exception as e:
            raise e

    def trial(rank, world_size, model, task, gpu_list, execution_parameters, q, tid):
        try:
            setup(rank, world_size, tid)
            torch.cuda.set_device(gpu_list[rank])
            loss_fn = task.loss_function
            dataloader = FSDPExecutor.gen_dataloader(
                rank, world_size, task.internal_dl())

            # if a transformer, then need to use special wrapping policy
            if task.hints is not None and task.hints.get("is_transformer", False):
                wrap_policy = functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=task.hints.get(
                        "transformer_cls", None)
                )
            else:
                wrap_policy = always_wrap_policy

            if execution_parameters["offload"]:
                fsdp_model = FSDP(model, auto_wrap_policy=wrap_policy, cpu_offload=CPUOffload(
                    offload_params=True), device_id=torch.cuda.current_device())
            else:
                fsdp_model = FSDP(model, auto_wrap_policy=wrap_policy,
                                  device_id=torch.cuda.current_device())

            if execution_parameters["checkpoint"]:
                apply_activation_checkpointing(
                    fsdp_model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=lambda l: isinstance(l, FSDP))

            lr = task.hparams.lr

            optimizer = task.hparams.optimizer_cls(
                fsdp_model.parameters(), lr=lr)

            time_taken = 0
            for iter, (batch, label) in enumerate(dataloader):
                if iter == FSDPExecutor.trial_batch_count:
                    break
                elif iter == FSDPExecutor.trial_batch_count - 1:
                    st = timer()
                out = fsdp_model(batch.to(torch.cuda.current_device()))
                loss = loss_fn(out, label.to(torch.cuda.current_device()))
                loss.backward()
                optimizer.step()
                fsdp_model.zero_grad()
                optimizer.zero_grad()
                if iter == FSDPExecutor.trial_batch_count - 1:
                    time_taken = timer() - st
            if rank == 0:
                q.put(time_taken)
        except Exception as e:
            raise e

    def gen_dataloader(rank, world_size, old):
        sampler = torch.utils.data.DistributedSampler(
            old.dataset,
            world_size,
            rank,
            shuffle=False,  # currently do not support shuffling @TODO, fix this
            drop_last=old.drop_last
        )
        return iter(
            torch.utils.data.DataLoader(
                old.dataset,
                batch_size=old.batch_size // world_size,
                shuffle=None,  # can't specify along with sampler
                sampler=sampler,
                batch_sampler=None,
                num_workers=old.num_workers,
                collate_fn=old.collate_fn,
                pin_memory=old.pin_memory,
                drop_last=old.drop_last,
                timeout=old.timeout,
                worker_init_fn=old.worker_init_fn,
                multiprocessing_context=old.multiprocessing_context,
                generator=old.generator,
                prefetch_factor=old.prefetch_factor,
                persistent_workers=old.persistent_workers,
                pin_memory_device=old.pin_memory_device
            )
        )

    def inner_fsdp_execution(rank: int, world_size: int, task: Task, gpu_list: List[int], tid, override_batch_count=None):
        setup(rank, world_size, tid)
        torch.cuda.set_device(gpu_list[rank])

        model, loss_fn = task.get_model(fresh=True), task.loss_function
        if task.has_ckpt() and rank == 0:
            model_ckpt = task.get_model(fresh=False)
            model.load_state_dict(model_ckpt)
        
        
        execution_parameters = task.selected_strategy.parameters
        iterator = task.get_iterator(FSDPExecutor.gen_dataloader(
            rank, world_size, task.internal_dl()))

        # if a transformer, then need to use special wrapping policy
        if task.hints is not None and task.hints.get("is_transformer", False):
            wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=task.hints.get("transformer_cls", None)
            )
        else:
            wrap_policy = always_wrap_policy

        if execution_parameters["offload"]:
            fsdp_model = FSDP(model, auto_wrap_policy=wrap_policy, cpu_offload=CPUOffload(
                offload_params=True), device_id=torch.cuda.current_device())
        else:
            fsdp_model = FSDP(model, auto_wrap_policy=wrap_policy,
                              device_id=torch.cuda.current_device())

        # Apply Checkpointing BEFORE offload/spilling
        if execution_parameters["checkpoint"]:
            apply_activation_checkpointing(
                fsdp_model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=lambda l: isinstance(l, FSDP))

        lr = task.hparams.lr
        optimizer = task.hparams.optimizer_cls(fsdp_model.parameters(), lr=lr)
        torch.distributed.barrier()
        ctr = 0
        while True:
            try:
                batch, label = next(iterator)
            except:
                iterator = task.get_iterator(FSDPExecutor.gen_dataloader(rank, world_size, task.internal_dl()))
                batch, label = next(iterator)
            if override_batch_count is not None and ctr == override_batch_count:
                break
            ctr += 1
            out = fsdp_model(batch.to(torch.cuda.current_device()))
            loss = loss_fn(out, label.to(torch.cuda.current_device()))
            loss.backward()
            optimizer.step()
            fsdp_model.zero_grad()
            optimizer.zero_grad()
        
        dist.barrier()
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state = fsdp_model.state_dict()
        if rank == 0:
            task.save(cpu_state)
        
