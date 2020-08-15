import shlex, subprocess
#command_line = input()
#result = subprocess.run(["python3", "load_eta.py", "0.2", "5000"], stdout=subprocess.PIPE)
#subprocess.run(["python3", "load_eta.py", "0.2", "5000"], stdout=subprocess.PIPE).stdout
#print(result.stdout)
#subprocess.run(["ls", "-l", "/dev/null"], capture_output=True)
#subprocess.run(["ls", "-l"])  # doesn't capture output

#output = subprocess.check_output(["python3", "load_eta.py", "0.2", "5000"])
#print(output)

subprocess.getstatusoutput('ls')
