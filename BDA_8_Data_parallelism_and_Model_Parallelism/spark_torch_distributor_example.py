"""
Skeleton example showing how one would integrate TorchDistributor (PyTorch on Spark)
into a Spark-based ETL -> training flow. This file is a template and must be adapted
to the specific Spark/TorchDistributor version and cluster configuration.

This script is not intended to be executed as-is in this environment. It documents
the typical structure and can be used as a starting point.
"""
from textwrap import dedent

EXAMPLE = dedent('''
from spark_torch.distributor import TorchDistributor
from pyspark.sql import SparkSession

def train_on_executor(local_rank, world_size, **kwargs):
    # This function runs inside each Spark executor/process.
    # Set up torch.distributed and run training loop similar to train_ddp.py
    import os
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(kwargs.get('model_name'))
    model = AutoModelForSequenceClassification.from_pretrained(kwargs.get('model_name'), num_labels=2)
    # init process group and DDP etc.
    # load local shard of Parquet data (each executor reads its assigned partitions)
    # implement training loop and checkpointing
    print('Executor local_rank', local_rank, 'world_size', world_size)

if __name__ == '__main__':
    spark = SparkSession.builder.appName('torch-distrib-example').getOrCreate()
    td = TorchDistributor(backend='nccl' if torch.cuda.is_available() else 'gloo')
    # number of processes/executors must be configured according to your cluster
    td.run(train_on_executor, kwargs={'model_name': 'bert-base-uncased'})

''')

def print_example():
    print("This file is a template. Read comments and adapt to your cluster.")
    print(EXAMPLE)

if __name__ == '__main__':
    print_example()
