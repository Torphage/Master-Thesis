import subprocess
import sys

if __name__ == "__main__":
    num_threads = sys.argv[1]
    subprocess.run(["./build/temp", num_threads])