import fire
import sys
import wandb

from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness


def entry_point(
    sample_file: str,
    k: str = "1,10",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    wandb.init(project="human_eval_results")  # Replace with your entity
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    print(results)

    # Log the results as a bar chart
    data = [[f'pass@{ki}', results[f'pass@{ki}']] for ki in k]
    table = wandb.Table(data=data, columns=["k_value", "pass_rate"])
    wandb.log({"Functional Correctness": wandb.plot.bar(table, "k_value", "pass_rate", title="Functional Correctness at different k-values")})

    # Finish the wandb run
    wandb.finish()


def main():
    fire.Fire(entry_point)


sys.exit(main())