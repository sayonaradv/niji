"""Benchmarking utilities for measuring model performance.

This module provides functionality to benchmark inference speed and memory usage
of toxicity classification models. It supports both local checkpoints and
pre-trained models from the AVAILABLE_MODELS dictionary.
"""

import time
from typing import Any

import numpy as np
import torch
from jsonargparse import auto_cli
from lightning.pytorch import LightningModule

from ruffle.models import Classifier
from ruffle.predictor import AVAILABLE_MODELS
from ruffle.types import TextInput


class InferenceBenchmark:
    """Benchmark inference speed and memory usage for toxicity classification models."""

    def __init__(
        self,
        model_name: str = "bert-tiny",
        ckpt_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the benchmark with a model.

        Args:
            model_name: Name of the pre-trained model to use from AVAILABLE_MODELS.
            ckpt_path: Path to a local model checkpoint. If None, downloads from repository.
            device: PyTorch device specification for inference.
        """
        self.model_name = model_name
        self.device = device
        self.model = self._load_model(ckpt_path)
        self.model.eval()

    def _load_model(self, ckpt_path: str | None) -> LightningModule:
        """Load model from checkpoint path or download URL."""
        if ckpt_path is None:
            if self.model_name not in AVAILABLE_MODELS:
                available = ", ".join(AVAILABLE_MODELS.keys())
                raise ValueError(
                    f"Unknown model '{self.model_name}'. Available: {available}"
                )
            ckpt_path = AVAILABLE_MODELS[self.model_name]

        return Classifier.load_from_checkpoint(ckpt_path, map_location=self.device)

    def benchmark_inference_speed(
        self,
        texts: TextInput,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> dict[str, Any]:
        """Benchmark inference speed by running multiple iterations.

        Args:
            texts: Input text(s) to use for benchmarking.
            num_iterations: Number of inference iterations to average over.
            warmup_iterations: Number of warmup iterations to run before benchmarking.

        Returns:
            Dictionary containing benchmark results including:
                - avg_inference_time_ms: Average inference time in milliseconds
                - std_inference_time_ms: Standard deviation of inference times
                - min_inference_time_ms: Minimum inference time
                - max_inference_time_ms: Maximum inference time
                - throughput_samples_per_sec: Throughput in samples per second
                - batch_size: Number of samples in each batch
        """
        text_list = [texts] if isinstance(texts, str) else texts
        batch_size = len(text_list)

        print(f"Running inference benchmark on {self.device}")
        print(f"Model: {self.model_name}")
        print(f"Batch size: {batch_size}")
        print(f"Warmup iterations: {warmup_iterations}")
        print(f"Benchmark iterations: {num_iterations}")
        print("-" * 50)

        # Warmup iterations
        print("Running warmup...")
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = self.model(text_list)

        # Benchmark iterations
        print("Running benchmark...")
        times = []

        with torch.no_grad():
            for i in range(num_iterations):
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()

                start_time = time.perf_counter()
                _ = self.model(text_list)

                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                inference_time_ms = (end_time - start_time) * 1000
                times.append(inference_time_ms)

                if (i + 1) % 25 == 0:
                    print(f"Progress: {i + 1}/{num_iterations}")

        # Calculate statistics
        times_array = np.array(times)
        avg_time = np.mean(times_array)
        std_time = np.std(times_array)
        min_time = np.min(times_array)
        max_time = np.max(times_array)
        throughput = (batch_size * 1000) / avg_time  # samples per second

        results = {
            "avg_inference_time_ms": float(avg_time),
            "std_inference_time_ms": float(std_time),
            "min_inference_time_ms": float(min_time),
            "max_inference_time_ms": float(max_time),
            "throughput_samples_per_sec": float(throughput),
            "batch_size": batch_size,
            "num_iterations": num_iterations,
            "device": self.device,
            "model_name": self.model_name,
        }

        self._print_benchmark_results(results)
        return results

    def _print_benchmark_results(self, results: dict[str, Any]) -> None:
        """Print formatted benchmark results."""
        print("\n" + "=" * 60)
        print("INFERENCE SPEED BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Model: {results['model_name']}")
        print(f"Device: {results['device']}")
        print(f"Batch size: {results['batch_size']}")
        print(f"Iterations: {results['num_iterations']}")
        print("-" * 60)
        print(
            f"Average inference time: {results['avg_inference_time_ms']:.2f} ± {results['std_inference_time_ms']:.2f} ms"
        )
        print(f"Min inference time: {results['min_inference_time_ms']:.2f} ms")
        print(f"Max inference time: {results['max_inference_time_ms']:.2f} ms")
        print(f"Throughput: {results['throughput_samples_per_sec']:.2f} samples/sec")
        print("=" * 60)


def benchmark_speed(
    texts: list[str] | None = None,
    model_name: str = "bert-tiny",
    ckpt_path: str | None = None,
    device: str = "cpu",
    num_iterations: int = 100,
    warmup_iterations: int = 10,
) -> None:
    """Command-line interface for running inference speed benchmarks.

    Args:
        texts: List of texts to benchmark with. If None, uses default sample texts.
        model_name: Name of the pre-trained model to benchmark.
        ckpt_path: Path to a local model checkpoint file.
        device: PyTorch device specification for inference.
        num_iterations: Number of inference iterations to average over.
        warmup_iterations: Number of warmup iterations before benchmarking.

    Example:
        ```bash
        python -m ruffle.benchmarks --texts '["Hello world", "Test text"]' --device cuda
        python benchmarks.py --model_name bert-tiny --num_iterations 200
        ```
    """
    if texts is None:
        texts = [
            "I love this product! It's amazing.",
            "This is terrible and I hate it.",
            "The weather is nice today.",
            "You are an idiot and I hope you fail.",
            "Thanks for your help, much appreciated!",
        ]

    benchmark = InferenceBenchmark(
        model_name=model_name,
        ckpt_path=ckpt_path,
        device=device,
    )

    benchmark.benchmark_inference_speed(
        texts=texts,
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations,
    )


def cli_main() -> None:
    """Entry point for the command-line interface."""
    auto_cli(benchmark_speed)


if __name__ == "__main__":
    cli_main()
