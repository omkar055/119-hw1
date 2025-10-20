"""
Part 3: Short Exercises on the Shell

For the third and last part of this homework,
we will complete a few tasks related to shell programming
and shell commands, particularly, with relevance to how
the shell is used in data science.

Please note:
The "project proposal" portion will be postponed to part of Homework 2.

===== Questions 1-5: Setup Scripting =====

1. For this first part, let's write a setup script
that downloads a dataset from the web,
clones a GitHub repository, and runs the Python script
contained in `script.py` on the dataset in question.

For the download portion, we have written a helper
download_file(url, filename) which downloads the file
at `url` and saves it in `filename`.

You should use Python subprocess to run all of these operations.

To test out your script, and as your answer to this part,
run the following:
    setup(
        "https://github.com/DavisPL-Teaching/119-hw1",
        "https://raw.githubusercontent.com/DavisPL-Teaching/119-hw1/refs/heads/main/data/test-input.txt",
        "test-script.py"
    )

Then read the output of `output/test-output.txt`,
convert it to an integer and return it. You should get "12345".

Note:
Running this question will leave an extra repo 119-hw1 lying around in your repository.
We recommend adding this to your .gitignore file so it does not
get uploaded when you submit.
"""

# You may need to conda install requests or pip3 install requests
import requests
import subprocess
import os

def download_file(url, filename):
    r = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(r.content)

def clone_repo(repo_url):
    repo_name = repo_url.rstrip('/').split('/')[-1]

    if os.path.exists(repo_name):
        print("Repo already exists, skipping clone.")
        return
    subprocess.run(['git', 'clone', repo_url], check=True)


def run_script(script_path, data_path):
    subprocess.run(["python3", script_path, data_path], check=True)


def setup(repo_url, data_url, script_path):
    data_file = "input_data.txt"
    download_file(data_url, data_file)
    
    # Clone the repo
    clone_repo(repo_url)
    
    # Run the script
    run_script(script_path, data_file)


def q1():
    # Call setup as described in the prompt
    setup(
        "https://github.com/DavisPL-Teaching/119-hw1",
        "https://raw.githubusercontent.com/DavisPL-Teaching/119-hw1/refs/heads/main/data/test-input.txt",
        "test-script.py"
    )
    # Read the file test-output.txt to a string
    with open("output/test-output.txt") as f:
    # Return the integer value of the output
        output = int(f.read().strip())
    
    return output

"""
2.
Suppose you are on a team of 5 data scientists working on
a project; every 2 weeks you need to re-run your scripts to
fetch the latest data and produce the latest analysis.

a. When might you need to use a script like setup() above in
this scenario?

=== ANSWER Q2a BELOW ===
Scripts will be useful when automating data pipelines that I want to run more than once like in our case,
which ensuring reproducibility and avoids redundancy across the team, making the results consistent for everyone.
=== END OF Q2a ANSWER ===

Do you see an alternative to using a script like setup()?

=== ANSWER Q2b BELOW ===
Yeah, I know some technologies exist like Docker, Kuberneties, Cloud, that can help with managing 
consistency and accuracy of environments across various different machines.
=== END OF Q2b ANSWER ===

3.
Now imagine we have to re-think our script to
work on a brand-new machine, without any software installed.
(For example, this would be the case if you need to run
your pipeline inside an Amazon cloud instance or inside a
Docker instance -- to be more specific you would need
to write something like a Dockerfile, see here:
https://docs.docker.com/reference/dockerfile/
which is basically a list of shell commands.)

Don't worry, we won't test your code for this part!
I just want to see that you are thinking about how
shell commands can be used for setup and configuration
necessary for data processing pipelines to work.

Think back to HW0. What sequence of commands did you
need to run?
Write a function setup_for_new_machine() that would
be able to run on a brand-new machine and set up
everything that you need.

Assume that you need your script to work on all of the packages
that we have used in HW1 (that is, any `import` statements
and any other software dependencies).

Assume that the new server machine is identical
in operating system and architecture to your own,
but it doesn't have any software installed.
It has Python 3.12
and conda or pip3 installed to get needed packages.

Hint: use subprocess again!

Hint: search for "import" in parts 1-3. Did you miss installing
any packages?
"""

def setup_for_new_machine():
    py_packages_needed = [
        'pandas',        
        'matplotlib', 
        'math',
        'time',            
        'requests',
        'subprocess',
        'os'
    ]
    
    # Install packages using pip3
    for package in py_packages_needed:
        print(f"Installing {package}...")
        result = subprocess.run(['conda', 'install', package], 
                              capture_output=True, text=True)
    
    # Create necessary directories
    directories = ['output', 'data']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Creating directory: {directory}")
    


def q3():
    # As your answer, return a string containing
    # the operating system name that you assumed the
    # new machine to have.
    os = 'MacOS'
    return os

"""
4. This question is open ended :)
It won't be graded for correctness.

What percentage of the time do you think real data scientists
working in larger-scale projects in industry have to write
scripts like setup() and setup_for_new_machine()
in their day-to-day jobs?

=== ANSWER Q4 BELOW ===
I think maybe around 25% of the time because while scripting is importing, I think data scientists focus
more on modelling, fine-tuning and analyzing to derive insights from the data. This seems something Data
Engineers might be doing more often. 
=== END OF Q4 ANSWER ===

5.
Copy your setup_for_new_machine() function from Q3
(remove the other code in this file)
to a new script and run it on a friend's machine who
is not in this class. Did it work? What problems did you run into?

Only answer this if you actually did the above.
Paste the output you got when running the script on the
new machine:

If you don't have a friend's machine, please speculate about
what might happen if you tried. You can guess.

=== ANSWER Q5 BELOW ===
I didn't check with a friend but I think Q3 should work if my friend has conda, has a similar machine (OS and architecture)
as long as he has no permission issues or shell issues and a directory set up to run the script so that appropriate 
sub-directories are made.
=== END OF Q5 ANSWER ===

===== Questions 6-9: A comparison of shell vs. Python =====

The shell can also be used to process data.

This series of questions will be in the same style as part 2.
Let's import the part2 module:
"""

import part2
import pandas as pd

"""
Write two versions of a script that takes in the population.csv
file and produces as output the number of rows in the file.
The first version should use shell commands and the second
should use Pandas.

For technical reasons, you will need to use
os.popen instead of subprocess.run for the shell version.
Example:
    os.popen("echo hello").read()

Runs the command `echo hello` and returns the output as a string.

Hints:
    1. Given a file, you can print it out using
        cat filename

    2. Given a shell command, you can use the `tail` command
        to skip the first line of the output. Like this:

    (shell command that spits output) | tail -n +2

    Note: if you are curious why +2 is required here instead
        of +1, that is an odd quirk of the tail command.
        See here: https://stackoverflow.com/a/604871/2038713

    3. Given a shell command, you can use the `wc` command
        to count the number of lines in the output

   (shell command that spits output) | wc -l

NOTE:
The shell commands above require that population.csv
has a newline at the end of the file.
Otherwise, it will give an off-by-one error
FYI, if this were not the case you can replace
    cat filename
with:
    (cat filename ; echo)
.
"""

def pipeline_shell():
    command = "cat data/population.csv | tail -n +2 | wc -l"
    result = os.popen(command).read().strip()
    return int(result)

def pipeline_pandas():
    df = pd.read_csv('data/population.csv')
    return len(df)

def q6():
    # As your answer to this part, check that both
    # integers are the same and return one of them.
    shell = pipeline_shell()
    pandas = pipeline_pandas()
    
    # Verify they're the same
    if (shell == pandas): 
        print("Counts match")
    else:
        print("Counts don't match")
    
    return shell

"""
Let's do a performance comparison between the two methods.

Use use your ThroughputHelper and LatencyHelper classes
from part 2 to get answers for both pipelines.

Additionally, generate a plot and save it in
    output/part3-q7.png

7. Throughput
"""

def q7():
    # Return a list of two floats
    # [throughput for shell, throughput for pandas]
    # (in rows per second)
    # Get the row count for throughput calculation
    row_count = q6()
    
    t = part2.ThroughputHelper()
    t.add_pipeline('shell', row_count, pipeline_shell)
    t.add_pipeline('pandas', row_count, pipeline_pandas)
    
    throughputs = t.compare_throughput()
    t.generate_plot('output/part3-q7.png')
    
    return throughputs

"""
8. Latency

For latency, remember that we should create a version of the
pipeline that processes only a single row! (As in Part 2).
However, for this question only, it is OK if you choose to run
latency on the entire pipeline instead.

Additionally, generate a plot and save it in
    output/part3-q8.png
"""

def q8():
    # Return a list of two floats
    # [latency for shell, latency for pandas]
    # (in milliseconds)
    l = part2.LatencyHelper()

    l.add_pipeline('shell', pipeline_shell)
    l.add_pipeline('pandas', pipeline_pandas)
    
    latencies = l.compare_latency()
    l.generate_plot('output/part3-q8.png')
    
    return latencies

"""
9. Which method is faster?
Comment on anything else you notice below.

=== ANSWER Q9 BELOW ===
Pandas is much faster in terms of both Throughput and Latency because I'm assuming shell commands through
os.popen() or subprocess have a lot of setting up in the background, slowing the process, as well as more
functions/commands used in shell compared to just len in pandas.
=== END OF Q9 ANSWER ===
"""

"""
===== Wrapping things up =====

**Don't modify this part.**

To wrap things up, we have collected
your answers and saved them to a file below.
This will be run when you run the code.
"""

ANSWER_FILE = "output/part3-answers.txt"
UNFINISHED = 0

def log_answer(name, func, *args):
    try:
        answer = func(*args)
        print(f"{name} answer: {answer}")
        with open(ANSWER_FILE, 'a') as f:
            f.write(f'{name},{answer}\n')
            print(f"Answer saved to {ANSWER_FILE}")
    except NotImplementedError:
        print(f"Warning: {name} not implemented.")
        with open(ANSWER_FILE, 'a') as f:
            f.write(f'{name},Not Implemented\n')
        global UNFINISHED
        UNFINISHED += 1

def PART_3_PIPELINE():
    open(ANSWER_FILE, 'w').close()

    # Q1-5
    log_answer("q1", q1)
    # 2a: commentary
    # 2b: commentary
    log_answer("q3", q3)
    # 4: commentary
    # 5: commentary
    log_answer("q6", q6)
    log_answer("q7", q7)
    log_answer("q8", q8)
    # 9: commentary

    # Answer: return the number of questions that are not implemented
    if UNFINISHED > 0:
        print("Warning: there are unfinished questions.")

    return UNFINISHED

"""
=== END OF PART 3 ===

Main function
"""

if __name__ == '__main__':
    log_answer("PART 3", PART_3_PIPELINE)
