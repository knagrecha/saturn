from torch.nn.parallel import DistributedDataParallel as DDP
import sys
import torch
import copy
from timeit import default_timer as timer
import saturn.core.executors.multiprocessing.my_multiprocessing as mp
import multiprocessing as true_mp
import os
import functools
import torch.distributed as dist
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


class DDPExecutor(BaseTechnique):

    trial_batch_count = 2

    def __init__(self):
        super().__init__()
        self.name = "DDPExecutor"

    def execute(task: Task, gpu: List[int], tid, override_batch_count=None):
        WORLD_SIZE = len(gpu)
        mp.spawn(DDPExecutor.inner_ddp_execution, args=(
            WORLD_SIZE, task, gpu, tid, override_batch_count), nprocs=WORLD_SIZE, join=True)

    def search(task: Task, gpu: List[int], tid):
        if len(gpu) < 2:
            return None, None
        try:
            warnings.filterwarnings("ignore")
            ctx = true_mp.get_context("spawn")
            q = ctx.Queue()
            WORLD_SIZE = len(gpu)
            m = task.get_model()
            oom = False
            try:
                mp.spawn(DDPExecutor.trial, args=(
                    WORLD_SIZE, copy.deepcopy(m), task, gpu, q, tid), nprocs=WORLD_SIZE, join=True)

                rt = q.get()
                print(rt)

            except Exception as e:
                oom = True

            if not oom:
                return None, rt

            return None, None
          
        except Exception as e:
            print(e)
            raise e

    def trial(rank, world_size, model, task, gpu_list, q, tid):
        try:
            setup(rank, world_size, tid)
            torch.cuda.set_device(gpu_list[rank])
            loss_fn = task.loss_function
            dataloader = DDPExecutor.gen_dataloader(
                rank, world_size, task.internal_dl())

            

            
            ddp_model = DDP(model, device_ids=[torch.cuda.current_device()])

            
            lr = task.hparams.lr

            optimizer = task.hparams.optimizer_cls(
                ddp_model.parameters(), lr=lr)

            time_taken = 0
            for iter, (batch, label) in enumerate(dataloader):
                if (rank == 0):
                    print(iter)
                if iter == DDPExecutor.trial_batch_count:
                    break
                elif iter == DDPExecutor.trial_batch_count - 1:
                    st = timer()
                out = ddp_model(batch.to(torch.cuda.current_device()))
                loss = loss_fn(out, label.to(torch.cuda.current_device()))
                loss.backward()
                optimizer.step()
                ddp_model.zero_grad()
                optimizer.zero_grad()
                if iter == DDPExecutor.trial_batch_count - 1:
                    time_taken = timer() - st
            if rank == 0:
                q.put(time_taken)
        except Exception as e:
            print(e)
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

    def inner_ddp_execution(rank: int, world_size: int, task: Task, gpu_list: List[int], tid, override_batch_count=None):
        setup(rank, world_size, tid)
        torch.cuda.set_device(gpu_list[rank])

        model, loss_fn = task.get_model(fresh=True), task.loss_function
        if task.has_ckpt():
            model_ckpt = task.get_model(fresh=False)
            model.load_state_dict(model_ckpt)
        
        ddp_model = DDP(model, device_ids=[rank])
        dist.barrier()
        
        iterator = task.get_iterator(DDPExecutor.gen_dataloader(
            rank, world_size, task.internal_dl()))


        lr = task.hparams.lr
        optimizer = task.hparams.optimizer_cls(ddp_model.parameters(), lr=lr)
        torch.distributed.barrier()
        ctr = 0
        while True:
            try:
                batch, label = next(iterator)
            except:
                iterator = DDPExecutor.gen_dataloader(rank, world_size, task.internal_dl())
                batch, label = next(iterator)
            if override_batch_count is not None and ctr == override_batch_count:
                break
            ctr += 1
            out = ddp_model(batch.to(torch.cuda.current_device()))
            loss = loss_fn(out, label.to(torch.cuda.current_device()))
            loss.backward()
            optimizer.step()
            ddp_model.zero_grad()
            optimizer.zero_grad()
        if rank == 0:
            task.save(ddp_model.module.state_dict())
