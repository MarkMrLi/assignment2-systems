import timeit
import argparse
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
import torch
import pandas as pd
import submitit
import itertools

def init_model(vocab_size, context_length, d_model, d_ff, num_layers, num_heads):
    
    return BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=10000
    )
    

def generate_data(vocab_size, batch_size, context_length, device:str="cpu"):
    mock_input = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    mock_target = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return mock_input, mock_target

def benchmark_one_step(
        d_model, 
        d_ff, 
        num_layers, 
        num_heads, 
        vocab_size, 
        context_length,
        batch_size,
        warmup_step,
    ):


    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Init model")
    model = init_model(
        vocab_size,
        context_length,
        d_model,
        d_ff,
        num_layers,
        num_heads
    )
    if device == "cuda":
        model.to(device)

    optimizer = AdamW(params=model.parameters())

    for _ in range(warmup_step):
        # train loop start
        optimizer.zero_grad()
        inputs, targets = generate_data(
            vocab_size,
            batch_size,
            context_length,
            device=device
        )

        # forward
        start_forward_time = timeit.default_timer()
        y_hat = model.forward(inputs)

        if device == "cuda":
            torch.cuda.synchronize()
        loss = cross_entropy(y_hat, targets)
        # backward
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        optimizer.step()
        
    # train loop start
    optimizer.zero_grad()
    inputs, targets = generate_data(
        vocab_size,
        batch_size,
        context_length,
        device=device
    )

    # forward
    start_forward_time = timeit.default_timer()
    y_hat = model.forward(inputs)

    if device == "cuda":
        torch.cuda.synchronize()

    forward_duration = timeit.default_timer() - start_forward_time
    print(f"forward duration:{forward_duration}")

    loss = cross_entropy(y_hat, targets)
    # backward
    start_backward_time = timeit.default_timer()
    loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    backward_duration = timeit.default_timer() - start_backward_time
    print(f"backward duration:{backward_duration}")

    optimizer.step()

    return {"forward_duration": forward_duration, "backward_duration": backward_duration}



def eval():
    d_model = [768, 1024, 1280, 1600]
    d_ff = [3072, 4096, 5120, 6400, 10240]
    num_layers = [12, 24, 36, 48, 32]
    num_heads = [12, 16, 20, 25, 32]
    vocab_size = 10000
    context_length = 256
    batch_size = 4
    warmup_step = 5

    executor = submitit.AutoExecutor(folder="logs/slurm_logs")

    executor.update_parameters(
        timeout_min=1,
        cpus_per_task=2
    )
    jobs = []
    # 2. 暂时只跑 ONE 个任务来测试，不要跑循环
    # 确保单个任务能跑通
    print("Running debug test for one config...")
    job = executor.submit(
        benchmark_one_step, 
        d_model[2], d_ff[2], num_layers[2], num_heads[2], # 只取第一个配置
        vocab_size, context_length, batch_size, warmup_step
    )
    
    # 3. 直接获取结果
    print("Result:", job.result())

    return 

    with executor.batch():
        for d_m, d_f, n_l, n_h in zip(
            d_model,
            d_ff,
            num_layers,
            num_heads
        ):
            job = executor.submit(
                benchmark_one_step,
                d_m,
                d_f,
                n_l,
                n_h,
                vocab_size,
                context_length,
                batch_size,
                warmup_step
            )
            jobs.append(job)
    results = [job.result() for job in jobs]





    
def main():
    parser = argparse.ArgumentParser(description="Benchmark your training infra")
    parser.add_argument("--eval", action="store_true", help="benchmark with default configs")
    parser.add_argument("--d_model", type=int, default=768, help="d_model")
    parser.add_argument("--d_ff", type=int, default=3072, help="d_ff")
    parser.add_argument("--num_layers", type=int, default=12, help="num_layers")
    parser.add_argument("--num_heads", type=int, default=12, help="num_heads")
    parser.add_argument("--context_length", type=int, default=256, help="context length")
    args = parser.parse_args()

    if args.eval:
        eval()
        return 


if __name__ == "__main__":
    main()

