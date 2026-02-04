import itertools
import json
import numpy as np

from tqdm import tqdm
from typing import List, Union, Iterable, Dict

from math_verify import verify, parse
from openmathinst_utils import process_results


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )

if __name__ == "__main__":
    num_total_list, num_correct_list = [], []
    
    data = [json.loads(line) for line in open("examples/generation.jsonl").readlines()]
    for line in tqdm(data):
        gt_answer = line["expected_answer"]
        response_list = line["response"]
        verify_results = [
                (
                    process_results(
                        resp,
                        gt_answer,
                        response_extract_from_boxed=True,
                    ) or
                    process_results(
                        resp,
                        gt_answer,
                        response_extract_from_boxed=False,
                        response_extract_regex=r"The answer is: (.+)$",
                    ) or
                    verify(parse(f"\\boxed{{${gt_answer}}}$"), parse(resp))
                ) for resp in response_list
            ]
        # assert verify_results == line["correct"]
        
        num_total_list.append(len(verify_results))
        num_correct_list.append(sum(verify_results))
        
    n = num_total_list[0]
    ks = [2 ** e for e in range(0, 7)]
    ks = [k for k in ks if (2 * k) <= n or k == 1]
    
    for k in ks:
        pass_at_k = estimate_pass_at_k(num_total_list, num_correct_list, k)
        print(f"pass@{k} = {np.mean(pass_at_k)}")
