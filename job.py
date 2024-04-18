import os
import sys
import subprocess


def remove_duplicates_and_get_indices(physical_ids, core_ids, processes):
    cores = {}
    
    for (i, p, c) in zip(processes, physical_ids, core_ids):
        # print(i, p, c)
        if c not in cores:
            cores[c] = {}
        if p not in cores[c] and p != -1:
            cores[c][p] = i 
    
    indices = [value for inner_dict in cores.values() for value in inner_dict.values()]
    
    return indices


if __name__ == "__main__":
    run_info = subprocess.check_output(["""grep 'core id' /proc/cpuinfo -C 2 | awk -F'\t' 'BEGIN{ OFS="\t"; i=0 } { if (NR % 6 == 0) {i++}; print i, $0}'"""], shell=True).decode("utf-8")
    
    pr = subprocess.check_output(["grep 'processor' /proc/cpuinfo"], shell=True).decode("utf-8")
    proc = subprocess.check_output(["grep 'core id' /proc/cpuinfo -C 2"], shell=True).decode("utf-8")
    ls = pr.splitlines()
    lines = proc.splitlines()

    available_threads = []
    is_slurm = False
    try:
        run_info = subprocess.check_output(["scontrol -dd show job $SLURM_JOB_ID | grep 'CPU_IDs' | sed 's/^ *//'"], shell=True).decode("utf-8")
        index = run_info.find("CPU_IDs") + 8
        end_index = run_info.find(" ", index)
        substr = run_info[index:end_index]
        # print(substr)
        ranges = substr.split(",")
        for r in ranges:
            if r.find("-") != -1:
                indices = r.split("-")
                start = int(indices[0])
                end = int(indices[1])
                available_threads += list(range(start, end + 1))
            else:
                available_threads += int(r)
        is_slurm = True
    except:
        available_threads = [i for i in range(len(ls))]
        pass
    # available_threads = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93]
    

    phys = []
    apicids = []
    physical_ids = []
    cores = []
    processes = []
    print(available_threads)
    siblings = 0

    j = 0
    for i in range(0, len(lines), 6):
        processor = ls[j]
        physical_id = lines[i]
        siblings = lines[i + 1]
        coreid = lines[i + 2]
        apicid = lines[i + 4]
        process = int(processor[(processor.find(":") + 2):])
        a_value = int(apicid[(apicid.find(":") + 2):])
        p_value = int(physical_id[(physical_id.find(":") + 2):])
        s_value = int(siblings[(siblings.find(":") + 2):])
        c_value = int(coreid[(coreid.find(":") + 2):])

        if process in available_threads:
            physical_ids.append(p_value)
            cores.append(c_value)
            phys.append(p_value)
            processes.append(process)

        j += 1

    indices = remove_duplicates_and_get_indices(physical_ids, cores, processes)
    # indices2 = remove_duplicates_and_get_indices(physical_ids, cores, processes)

    prefix = "taskset --cpu-list " + ",".join([str(i) for i in indices])
    # prefix2 = "taskset --cpu-list " + ",".join([str(i) for i in indices2])
    
    print("Sum: ", sum([1 for _ in indices]), "     should be: >", 48 - (96 - 80))

    if not "OMP_NUM_THREADS" in os.environ:
        print("$OMP_NUM_THREADS is not set")
    os.environ["OMP_NUM_THREADS"] = str(len(indices))

    args = sys.argv[1:]
    print(prefix + " " + " ".join(args))
    subprocess.run([prefix + " " + " ".join(args)], shell=True)
    # subprocess.run([prefix2 + " " + " ".join(args)], shell=True)
#    subprocess.run(["cat", "/proc/meminfo"])
#    with open("./benchmark/benchmark.json", 'r') as fin:
#        print(fin.read())




# ssh bayes.ita.chalmers.se
# bash /opt/local/bin/run_job.sh job.py
