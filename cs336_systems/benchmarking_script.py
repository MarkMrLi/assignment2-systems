import timeit
import argparse
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
import torch
import pandas as pd
import submitit
import warnings
import os
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning, module='submitit')

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
    device_id
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"{'='*20} Benchmark Configuration {'='*20}")
    print(f"Device: {device}")
    print(f"d_model: {d_model}")
    print(f"d_ff: {d_ff}")
    print(f"num_layers: {num_layers}")
    print(f"num_heads: {num_heads}")
    print(f"vocab_size: {vocab_size}")
    print(f"context_length: {context_length}")
    print(f"batch_size: {batch_size}")
    print(f"warmup_step: {warmup_step}")
    print(f"device_id: {device_id}")
    print(f"{'='*53}\n")
    
    print(f"d_model")
    print("Init model")
    with torch.cuda.nvtx.range("Init model"):
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
        optimizer.zero_grad()
        inputs, targets = generate_data(vocab_size,batch_size,context_length,device=device)
        start_forward_time = timeit.default_timer()
        y_hat = model.forward(inputs)
        loss = cross_entropy(y_hat, targets)
        loss.backward()
        optimizer.step()
        if device == "cuda": torch.cuda.synchronize()
    
    optimizer.zero_grad()
    inputs, targets = generate_data(
        vocab_size,
        batch_size,
        context_length,
        device=device
    )
    
    fwd_times = []
    bwd_times = []
    
    for _ in range(10):
        torch.cuda.nvtx.range_push(f"step_{_}")
        start_forward_time = timeit.default_timer()
        with torch.cuda.nvtx.range("Forward"):
            y_hat = model.forward(inputs)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        forward_duration = timeit.default_timer() - start_forward_time
        fwd_times.append(forward_duration)
        
        loss = cross_entropy(y_hat, targets)
        with torch.cuda.nvtx.range("Backward"):
            start_backward_time = timeit.default_timer()
            loss.backward()
            if device == "cuda":
                torch.cuda.synchronize()
            backward_duration = timeit.default_timer() - start_backward_time
            bwd_times.append(backward_duration)
        optimizer.step()
        torch.cuda.nvtx.range_pop()
    
    return {
        "fwd_mean": np.mean(fwd_times),
        "fwd_std": np.std(fwd_times),
        "bwd_mean": np.mean(bwd_times),
        "bwd_std": np.std(bwd_times),
        "total_mean": np.mean(fwd_times) + np.mean(bwd_times)
    }

def eval():
    model_size = ["Small", "Medium", "Large", "xl", "2.7B"]
    d_model = [768, 1024, 1280, 1600, 2560]
    d_ff = [3072, 4096, 5120, 6400, 10240]
    num_layers = [12, 24, 36, 48, 32]
    num_heads = [12, 16, 20, 25, 32]
    device_ids = [0, 4, 5, 6, 7]
    vocab_size = 10000
    context_length = 256
    batch_size = 4
    warmup_step = 5
    
    executor = submitit.AutoExecutor(folder="logs/slurm_logs")
    executor.update_parameters(timeout_min=60)
    
    jobs = []
    
    with executor.batch():
        for d_m, d_f, n_l, n_h, d_id in zip(
            d_model,
            d_ff,
            num_layers,
            num_heads,
            device_ids
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
                warmup_step,
                d_id
            )
            jobs.append(job)
    durations = [job.result() for job in jobs]
    
    results = []
    for id, r in enumerate(durations):
        results.append({
            "Model size": model_size[id],
            "d_model": d_model[id],
            "d_ff": d_ff[id],
            "num_layers": num_layers[id],
            "num_heads": num_heads[id],
            "Fwd Mean (s)": f"{r['fwd_mean']:.2f}",
            "Fwd Std": f"{r['fwd_std']:.2f}",
            "Bwd Mean (s)": f"{r['bwd_mean']:.2f}",
            "Bwd Std": f"{r['bwd_std']:.2f}",
            "Total (s)": f"{r['total_mean']:.2f}"
        })
    
    df = pd.DataFrame(results)
    file_path = "/home/llz/repo/assignment2-systems/results/benchmark.md"
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"\n### Model Benchmarking Results with warm-up step:{warmup_step}\n")
        f.write(df.to_markdown(index=False))
    
    print(f"\n### Model Benchmarking Results with warm-up step:{warmup_step}")
    print(df.to_markdown(index=False))

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
    else:
        benchmark_one_step(args.d_model, args.d_ff, args.num_layers, args.num_heads, 10000, 256, 4, 5, device_id=0)

if __name__ == "__main__":
    main()