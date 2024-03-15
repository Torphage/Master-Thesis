import sys
import subprocess


if __name__ == "__main__":
    args = sys.argv[1:]
    print(args)
    subprocess.run(args)
    
    with open("./benchmark/benchmark.json", 'r') as fin:
        print(fin.read())




# ssh bayes.ita.chalmers.se
# bash /opt/local/bin/run_job.sh job.py