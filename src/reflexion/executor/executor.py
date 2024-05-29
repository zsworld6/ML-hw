import os
import subprocess
from typing import List, Tuple
from utils.utility import submit

class PyExecutor:    
    def evaluate(self, code: str) -> Tuple[bool, str]:
        current_path = os.path.dirname(os.path.realpath(__file__))
        parent_path = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
        target_path = os.path.join(parent_path, "gen", "codes")
        gen_code_path = os.path.join(target_path, "gen_code.py")
        print(gen_code_path)
        with open(gen_code_path, "w") as f:
            f.write(code)
        is_failed, feedback = subprocess.getstatusoutput(f"cd {target_path} && python {gen_code_path}")
                                                        #  python {gen_code_path}")
        if not is_failed:
            is_failed, feedback = submit()
        return not is_failed, feedback
        