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

if __name__ == "__main__":
    infos = subprocess.check_output(["""grep -E 'processor|physical id|core id' /proc/cpuinfo"""], shell=True).decode("utf-8")
    lines = list(chunks(infos.splitlines(), 3))

    available_threads = []
    is_slurm = False
    try:
        run_info = subprocess.check_output(["scontrol -dd show job $SLURM_JOB_ID | grep 'CPU_IDs' | sed 's/^ *//'"], shell=True).decode("utf-8")
        index = run_info.find("CPU_IDs") + 8
        end_index = run_info.find(" ", index)
        substr = run_info[index:end_index]
        print(substr)
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
        available_threads = [i for i in range(len(lines))]

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

    args = sys.argv[1:]
    
    print("Number of cores used: ", len(indices))
    print(list(zip(physical_ids, cores, processes)))
    print("Available threads: ", available_threads)
    print(prefix + " " + " ".join(args))

    subprocess.run([prefix + " " + " ".join(args)], shell=True)

