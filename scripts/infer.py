"""Run infer; see docs/reproduction.md."""
import sys
from loveda_rescue.cli import main
if __name__ == "__main__": main(["infer", *sys.argv[1:]])
