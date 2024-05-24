import subprocess
from typing import List, Tuple
from utility import submit

class PyExecutor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def evaluate(self, code: str) -> Tuple[bool, str]:
        with open("gen_code.py", "w") as f:
            f.write(code)
        is_failed, feedback = subprocess.getstatusoutput("python gen_code.py")
        if not is_failed:
            is_failed, feedback = submit()
        else:
            feedback = "Your implementation is wrong. Here are some error information:\n" + feedback
        return not is_failed, feedback
        
# PyTest
# exe = PyExecutor("Tasks/house-prices-advanced-regression-techniques")
# exe.evaluate("import pandas as pd\n")
