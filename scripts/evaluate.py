"""Run evaluate; see docs/reproduction.md."""
import sys
from loveda_rescue.cli import main
if __name__ == "__main__": main(["evaluate", *sys.argv[1:]])
