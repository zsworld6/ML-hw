import argparse
# from ..reflexion.reflexion import run_reflexion
from reflexion.reflexion import run_reflexion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="The directory where the model is stored.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="The directory where the train and test data is stored."
    )
    parser.add_argument(
        "--prompt_method",
        type=int,
        default=1,
        help="1 or 2",
    )
    parser.add_argument(
        "--describe_task",
        type=bool,
        default=False,
        help="Whether to describe the task in detail.",
    )
    parser.add_argument(
        "--reflexion_iterations",
        type=int,
        default=10,
        help="The number of iterations to run the reflexion method for.",
    )
    parser.add_argument(
        "--caafe_method",
        type=int,
        default=0,
        help="0 or 1 or 2",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="The max length generated.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=6,
        help="The max batch size.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="The temperature.",
    )
    parser.add_argument(
        "--top_k",
        type=float,
        default=1,
        help="The top k.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="The top p.",
    )    
    return parser.parse_args()

def main():
    args = parse_args()    
    run_reflexion(args)
    

if __name__ == "__main__":
    main()