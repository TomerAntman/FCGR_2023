import subprocess
import sys

def sendjob(isolate_name, FCGR=False, argument_tupples=None):
    outputpath=f"/home/labs/zeevid/Analyses/2022-HGR/BAM2FCGR/Isolates/{isolate_name}/bsub_output.txt"
    errorpath=f"/home/labs/zeevid/Analyses/2022-HGR/BAM2FCGR/Isolates/{isolate_name}/bsub_error.txt"
    command = f"bsub -R rusage[mem=32G] -n 1 -q new-short -o {outputpath} -e {errorpath} " \
              f"\" module load Pysam ; source /home/labs/zeevid/tomerant/FCGR_project/pip_envs/preprocess_env/bin/activate ; " \
              f"cd /home/labs/zeevid/Analyses/2022-HGR/BAM2FCGR/FCGR_2023/Feature_Engineering ; "
    if FCGR:
        command += f"python CGR2FCGR.py --isolate_name {isolate_name} "
    else:
        command += f"python Bam2CGR.py --isolate_name {isolate_name} "

    if argument_tupples is not None:
        for argument, value in argument_tupples:
            command += f" {argument} {value}"
    command += "\""
    subprocess.call(command, shell=True)
#main:
if __name__ == '__main__':
    # command calling this function can be:
    #1) python SEND_CGRs.py isolate_name1 isolate_name2 isolate_name3 ...
    #2) python SEND_CGRs.py FCGR isolate_name1 isolate_name2 isolate_name3 ...
    #3) python SEND_CGRs.py FCGR isolate_name1 isolate_name2 isolate_name3 ... --channels 1 --IPD_stride max
    #4) python SEND_CGRs.py isolate_name1 isolate_name2 isolate_name3 ... --channels 1 --IPD_stride max

    # if the first argument is "FCGR" then we want to run the FCGR script. otherwise, run the CGR script
    # check if any arguments start with "--" or "-"
    if len(sys.argv) >= 2 and not any([arg.startswith('--') or arg.startswith('-') for arg in sys.argv[1:]]):
        is_FCGR = True if sys.argv[1] == 'FCGR' else False
        isolate_names = sys.argv[2:] if is_FCGR else sys.argv[1:]
        for isolate_name in isolate_names:
            sendjob(isolate_name, FCGR=is_FCGR)
    else:
        first_argument_to_start_with_dash = [i for i, arg in enumerate(sys.argv) if arg.startswith('--') or arg.startswith('-')][0]
        is_FCGR = True if sys.argv[1] == 'FCGR' else False
        isolate_names = sys.argv[2:first_argument_to_start_with_dash] if is_FCGR else sys.argv[1:first_argument_to_start_with_dash]
        # the rest of the arguments might be other arguments for the script that we want to change
        # such as "--channels 3". take the arguments that start with "--" or "-" and send them (and the following argument) to the script
        argument_tupples = [(sys.argv[i], sys.argv[i+1]) for i, arg in enumerate(sys.argv) if arg.startswith('--') or arg.startswith('-')]
        for isolate_name in isolate_names:
            sendjob(isolate_name, FCGR=is_FCGR, argument_tupples=argument_tupples)