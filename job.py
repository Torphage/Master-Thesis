import os
import sys
import subprocess


def remove_duplicates_and_get_indices(physical_ids, core_ids, processes):
    cores = []
    indices = []
    
    for (i, p, c) in zip(processes, physical_ids, core_ids):
        # print(i, p, c)
        if [p, c] not in cores:
            cores.append([p, c])
            indices.append(i)
    
    return indices


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def get_list_from_ranges(s):
    available_threads = []
    ranges = s.split(",")
    for r in ranges:
        if r.find("-") != -1:
            indices = r.split("-")
            start = int(indices[0])
            end = int(indices[1])
            available_threads += list(range(start, end + 1))
        else:
            available_threads += int(r)
    return available_threads

if __name__ == "__main__":
    infos = subprocess.check_output(["""grep -E 'processor|physical id|core id' /proc/cpuinfo"""], shell=True).decode("utf-8")
    lines = list(chunks(infos.splitlines(), 3))

    # not_available_threads = []
    available_threads = []
    is_slurm = False
    # not_my_job_id = -1
    try:
        run_info = subprocess.check_output(["scontrol -dd show job $SLURM_JOB_ID | grep 'CPU_IDs' | sed 's/^ *//'"], shell=True).decode("utf-8")
        # not_my_job_id = int(subprocess.check_output(["sacct -r gpu-shannon -u oscpal --state RUNNING -oJobID | grep '.batch' | sed 's/^ *//'"], shell=True).decode("utf-8")[:5])
        # run_info2 = subprocess.check_output([f"scontrol -dd show job {not_my_job_id} | grep 'CPU_IDs' | sed 's/^ *//'"], shell=True).decode("utf-8")
        
        index = run_info.find("CPU_IDs") + 8
        end_index = run_info.find(" ", index)
        substr = run_info[index:end_index]
        # index2 = run_info2.find("CPU_IDs") + 8
        # end_index2 = run_info2.find(" ", index2)
        # substr2 = run_info2[index2:end_index2]
        print(substr)
        # print(substr2)
        available_threads = get_list_from_ranges(substr)
        # not_available_threads = get_list_from_ranges(substr2)
        is_slurm = True
    except:
        available_threads = [i for i in range(len(lines))]

    if (len(sys.argv) >= 2):
        rang = get_list_from_ranges(sys.argv[1])
        dont_use = []
        for dont in rang:
            dont_use.append(dont)
            dont_use.append(dont-48)
            dont_use.append(dont+48)
        available_threads = [a for a in available_threads if a not in dont_use]

    physical_ids = []
    cores = []
    processes = []
    for attr in lines:
        processor = attr[0]
        physical_id = attr[1]
        core_id = attr[2]

        process = int(processor[(processor.find(":") + 2):])
        p_value = int(physical_id[(physical_id.find(":") + 2):])
        c_value = int(core_id[(core_id.find(":") + 2):])

        if process in available_threads:
            physical_ids.append(p_value)
            cores.append(c_value)
            processes.append(process)

    indices = remove_duplicates_and_get_indices(physical_ids, cores, processes)

    prefix = "taskset --cpu-list " + ",".join([str(i) for i in indices])

    os.environ["OMP_NUM_THREADS"] = str(len(indices))

    args = sys.argv[2:]
    
    
    print("Number of cores used: ", len(indices))
    print(list(zip(physical_ids, cores, processes)))
    print("Available threads: ", available_threads)
    print(prefix + " " + " ".join(args))

    subprocess.run([prefix + " " + " ".join(args)], shell=True)