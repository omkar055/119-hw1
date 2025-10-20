"""
Part 2: Performance Comparisons

In this part, we will explore comparing the performance
of different pipelines.
First, we will set up some helper classes.
Then we will do a few comparisons
between two or more versions of a pipeline
to report which one is faster.
"""

import part1
import time
import matplotlib.pyplot as plt
import pandas as pd
import math

"""
=== Questions 1-5: Throughput and Latency Helpers ===

We will design and fill out two helper classes.

The first is a helper class for throughput (Q1).
The class is created by adding a series of pipelines
(via .add_pipeline(name, size, func))
where name is a title describing the pipeline,
size is the number of elements in the input dataset for the pipeline,
and func is a function that can be run on zero arguments
which runs the pipeline (like def f()).

The second is a similar helper class for latency (Q3).

1. Throughput helper class

Fill in the add_pipeline, eval_throughput, and generate_plot functions below.
"""

# Number of times to run each pipeline in the following results.
# You may modify this as you go through the file if you like, but make sure
# you set it back to 10 at the end before you submit.
NUM_RUNS = 10

class ThroughputHelper:
    def __init__(self):
        # Initialize the object.
        # Pipelines: a list of functions, where each function
        # can be run on no arguments.
        # (like: def f(): ... )
        self.pipelines = []

        # Pipeline names
        # A list of names for each pipeline
        self.names = []

        # Pipeline input sizes
        self.sizes = []

        # Pipeline throughputs
        # This is set to None, but will be set to a list after throughputs
        # are calculated.
        self.throughputs = None

    def add_pipeline(self, name, size, func):
        self.names.append(name)
        self.sizes.append(size)
        self.pipelines.append(func)

    def compare_throughput(self):
        # Measure the throughput of all pipelines
        # and store it in a list in self.throughputs.
        # Make sure to use the NUM_RUNS variable.
        # Also, return the resulting list of throughputs,
        # in **number of items per second.**
        self.throughputs = []

        for i in range(len(self.pipelines)):
            fun = self.pipelines[i]
            size = self.sizes[i]

            start_time = time.time() # use the time module to get the start and end times
            for _ in range(NUM_RUNS):
                fun()
            end_time = time.time()

            throughput = size / ((end_time - start_time) / NUM_RUNS)
            self.throughputs.append(throughput)

        return self.throughputs


    def generate_plot(self, filename):
        # Generate a plot for throughput using matplotlib.
        # You can use any plot you like, but a bar chart probably makes
        # the most sense.
        # Make sure you include a legend.
        # Save the result in the filename provided.
        if self.throughputs is None:
            raise ValueError("Initialize throughputs before generating plot")

        plt.figure(figsize=(12, 8))
        bars = plt.bar(self.names, self.throughputs, color='cyan', edgecolor='black')
        plt.xlabel('Pipelines')
        plt.ylabel('Throughput in items/second')
        plt.title("Pipeline vs Throughput")
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


"""
As your answer to this part,
return the name of the method you decided to use in
matplotlib.

(Example: "boxplot" or "scatter")
"""

def q1():
    # Return plot method (as a string) from matplotlib
    return "bar chart"

"""
2. A simple test case

To make sure your monitor is working, test it on a very simple
pipeline that adds up the total of all elements in a list.

We will compare three versions of the pipeline depending on the
input size.
"""

LIST_SMALL = [10] * 100
LIST_MEDIUM = [10] * 100_000
LIST_LARGE = [10] * 100_000_000

def add_list(l):
    # TODO
    # Please use a for loop (not a built-in)
    sum = 0
    for element in l:
        sum += element

    return sum

def q2a():
    # Create a ThroughputHelper object
    h = ThroughputHelper()
    # Add the 3 pipelines.
    # (You will need to create a pipeline for each one.)
    # Pipeline names: small, medium, large
    h.add_pipeline("small", len(LIST_SMALL), lambda: add_list(LIST_SMALL))
    h.add_pipeline("medium", len(LIST_MEDIUM), lambda: add_list(LIST_MEDIUM))
    h.add_pipeline("large", len(LIST_LARGE), lambda: add_list(LIST_LARGE))

    test_throughputs = h.compare_throughput()
    
    # Generate a plot.
    # Save the plot as 'output/part2-q2a.png'.
    h.generate_plot('output/part2-q2a.png')
    
    # Finally, return the throughputs as a list.
    return test_throughputs

"""
2b.
Which pipeline has the highest throughput?
Is this what you expected?

=== ANSWER Q2b BELOW ===
The large list has the highest throughput compared to medium and small and it is what I expected
because throughput increases linearly up to a certain point and then taper off. And we can see this 
in the bar plot where the increase from medium to large lists is smaller than the increase from 
small to medium lists. 
=== END OF Q2b ANSWER ===
"""

"""
3. Latency helper class.

Now we will create a similar helper class for latency.

The helper should assume a pipeline that only has *one* element
in the input dataset.

It should use the NUM_RUNS variable as with throughput.
"""

class LatencyHelper:
    def __init__(self):
        # Initialize the object.
        # Pipelines: a list of functions, where each function
        # can be run on no arguments.
        # (like: def f(): ... )
        self.pipelines = []

        # Pipeline names
        # A list of names for each pipeline
        self.names = []

        # Pipeline latencies
        # This is set to None, but will be set to a list after latencies
        # are calculated.
        self.latencies = None

    def add_pipeline(self, name, func):
        self.names.append(name)
        self.pipelines.append(func)


    def compare_latency(self):
        # Measure the latency of all pipelines
        # and store it in a list in self.latencies.
        # Also, return the resulting list of latencies,
        # in **milliseconds.**
        self.latencies = []

        for fun in self.pipelines:
            start_time = time.time() # use the time module to get the start and end times
            for _ in range(NUM_RUNS):
                fun()
            end_time = time.time()

            latency = ((end_time - start_time) / NUM_RUNS) * 1000
            self.latencies.append(latency)

        return self.latencies


    def generate_plot(self, filename):
        # Generate a plot for latency using matplotlib.
        # You can use any plot you like, but a bar chart probably makes
        # the most sense.
        # Make sure you include a legend.
        # Save the result in the filename provided.
        if self.latencies is None:
            raise ValueError("Initialize latencies before generating plot")

        plt.figure(figsize=(12, 8))
        bars = plt.bar(self.names, self.latencies, color='red', edgecolor='black')
        plt.xlabel('Pipelines')
        plt.ylabel('Latency in milliseconds')
        plt.title("Pipeline vs Latency")
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

"""
As your answer to this part,
return the number of input items that each pipeline should
process if the class is used correctly.
"""

def q3():
    # Return the number of input items in each dataset,
    # for the latency helper to run correctly.
    return 1

"""
4. To make sure your monitor is working, test it on
the simple pipeline from Q2.

For latency, all three pipelines would only process
one item. Therefore instead of using
LIST_SMALL, LIST_MEDIUM, and LIST_LARGE,
for this question run the same pipeline three times
on a single list item.
"""

LIST_SINGLE_ITEM = [10] # Note: a list with only 1 item

def q4a():
    # Create a LatencyHelper object
    h = LatencyHelper()
    # Add the single pipeline three times.
    h.add_pipeline("first", lambda: add_list(LIST_SINGLE_ITEM))
    h.add_pipeline("second", lambda: add_list(LIST_SINGLE_ITEM))
    h.add_pipeline("third", lambda: add_list(LIST_SINGLE_ITEM))

    test_latencies = h.compare_latency()
    # Generate a plot.
    # Save the plot as 'output/part2-q4a.png'.
    h.generate_plot('output/part2-q4a.png')

    # Finally, return the latencies as a list.
    return test_latencies

"""
4b.
How much did the latency vary between the three copies of the pipeline?
Is this more or less than what you expected?

=== ANSWER Q4b BELOW ===
Latency is more or less what I expected, I excepted slight variation (+/- a few ms) but we are only adding
a single element so it is extremely fast. And the downward trend is possibly due to memory management and 
caching as the data is in RAM now.
=== END OF Q4b ANSWER ===
"""

"""
Now that we have our helpers, let's do a simple comparison.

NOTE: you may add other helper functions that you may find useful
as you go through this file.

5. Comparison on Part 1

Finally, use the helpers above to calculate the throughput and latency
of the pipeline in part 1.
"""

# You will need these:
# part1.load_input
# part1.PART_1_PIPELINE

def q5a():
    # Return the throughput of the pipeline in part 1.
    t = ThroughputHelper()

    data_list = part1.load_input()
    total_size = sum(len(df) for df in data_list)

    t.add_pipeline("part 1 pipeline", total_size, part1.PART_1_PIPELINE)

    throughput = t.compare_throughput()

    return throughput[0]


def q5b():
    # Return the latency of the pipeline in part 1.
    l = LatencyHelper()

    l.add_pipeline('Part 1 Pipeline', part1.PART_1_PIPELINE)

    return l.compare_latency()


"""
===== Questions 6-10: Performance Comparison 1 =====

For our first performance comparison,
let's look at the cost of getting input from a file, vs. in an existing DataFrame.

6. We will use the same population dataset
that we used in lecture 3.

Load the data using load_input() given the file name.

- Make sure that you clean the data by removing
  continents and world data!
  (World data is listed under OWID_WRL)

Then, set up a simple pipeline that computes summary statistics
for the following:

- *Year over year increase* in population, per country

    (min, median, max, mean, and standard deviation)

How you should compute this:

- For each country, we need the maximum year and the minimum year
in the data. We should divide the population difference
over this time by the length of the time period.

- Make sure you throw out the cases where there is only one year
(if any).

- We should at this point have one data point per country.

- Finally, as your answer, return a list of the:
    min, median, max, mean, and standard deviation
  of the data.

Hints:
You can use the describe() function in Pandas to get these statistics.
You should be able to do something like
df.describe().loc["min"]["colum_name"]

to get a specific value from the describe() function.

You shouldn't use any for loops.
See if you can compute this using Pandas functions only.
"""

def load_input(filename):
    # Return a dataframe containing the population data

    df = pd.read_csv(filename)
    # **Clean the data here**
    df = df[(df['Code'].notna()) & (df['Code'] != "OWID_WRL")]

    return df

def population_pipeline(df):
    # Input: the dataframe from load_input()
    # Return a list of min, median, max, mean, and standard deviation
    grouped_df = df.groupby('Entity')

    year_span = grouped_df['Year'].max() - grouped_df['Year'].min()
    
    # Filter out countries with only one year of data
    countries_to_keep = year_span[year_span > 0].index
    df_filtered = df[df['Entity'].isin(countries_to_keep)]
    grouped_df = df_filtered.groupby('Entity')
    
    # Calculate population span
    pop = grouped_df['Population (historical)']
    pop_span = pop.max() - pop.min()
    
    # Get year span again for the filtered data
    year_span = grouped_df['Year'].max() - grouped_df['Year'].min()
    
    # Calculate year-over-year increase
    year_over_year_increase = pop_span / year_span
    
    # Get statistics
    stats = year_over_year_increase.describe()
    
    # Return [min, median, max, mean, std]
    return [
        stats['min'],
        stats['50%'],  
        stats['max'],
        stats['mean'],
        stats['std']
    ]
 


def q6():
    # As your answer to this part,
    # call load_input() and then population_pipeline()
    # Return a list of min, median, max, mean, and standard deviation
    df = load_input("data/population.csv") 
    return population_pipeline(df)


"""
7. Varying the input size

Next we want to set up three different datasets of different sizes.

Create three new files,
    - data/population-small.csv
      with the first 600 rows
    - data/population-medium.csv
      with the first 6000 rows
    - data/population-single-row.csv
      with only the first row
      (for calculating latency)

You can edit the csv file directly to extract the first rows
(remember to also include the header row)
and save a new file.

Make four versions of load input that load your datasets.
(The _large one should use the full population dataset.)
Each should return a dataframe.

The input CSV file will have 600 rows, but the DataFrame (after your cleaning) may have less than that.
"""

def load_input_small():
    df = load_input("data/population.csv") 
    df.head(600).to_csv('data/population-small.csv', index=False)

    return df.head(600)

def load_input_medium():
    df = load_input("data/population.csv") 
    df.head(6000).to_csv('data/population-medium.csv', index=False)

    return df.head(6000)

def load_input_large():
    df = load_input("data/population.csv") 

    return df

def load_input_single_row():
    # This is the pipeline we will use for latency.
    df = load_input("data/population.csv")
    df.head(1).to_csv('data/population-single-row.csv', index=False)

    # Return the single row dataframe
    return df.head(1)


def q7():
    # Don't modify this part
    s = load_input_small()
    m = load_input_medium()
    l = load_input_large()
    x = load_input_single_row()
    return [len(s), len(m), len(l), len(x)]

"""
8.
Create baseline pipelines

First let's create our baseline pipelines.
Create four pipelines,
    baseline_small
    baseline_medium
    baseline_large
    baseline_latency

based on the three datasets above.
Each should call your population_pipeline from Q6.

Your baseline_latency function will not be very interesting
as the pipeline does not produce any meaningful output on a single row!
You may choose to instead run an example with two rows,
or you may fill in this function in any other way that you choose
that you think is meaningful.
"""

def baseline_small():
    df = load_input_small()
    return population_pipeline(df)


def baseline_medium():
    df = load_input_medium()
    return population_pipeline(df)


def baseline_large():
    df = load_input_large()
    return population_pipeline(df)


def baseline_latency():
    df = load_input_single_row()
    if len(df) > 0:
        return [df['Population (historical)'].iloc[0]] * 5  # Return same value 5 times
    return [0, 0, 0, 0, 0]



def q8():
    # Don't modify this part
    _ = baseline_medium()
    return ["baseline_small", "baseline_medium", "baseline_large", "baseline_latency"]

"""
9.
Finally, let's compare whether loading an input from file is faster or slower
than getting it from an existing Pandas dataframe variable.

Create four new dataframes (constant global variables)
directly in the script.
Then use these to write 3 new pipelines:
    fromvar_small
    fromvar_medium
    fromvar_large
    fromvar_latency

These pipelines should produce the same answers as in Q8.

As your answer to this part;
a. Generate a plot in output/part2-q9a.png of the throughputs
    Return the list of 6 throughputs in this order:
    baseline_small, baseline_medium, baseline_large, fromvar_small, fromvar_medium, fromvar_large
b. Generate a plot in output/part2-q9b.png of the latencies
    Return the list of 2 latencies in this order:
    baseline_latency, fromvar_latency
"""

# TODO
POPULATION_SMALL = load_input_small()
POPULATION_MEDIUM = load_input_medium()
POPULATION_LARGE = load_input_large()
POPULATION_SINGLE_ROW = load_input_single_row()

def fromvar_small():
    return population_pipeline(POPULATION_SMALL)

def fromvar_medium():
    return population_pipeline(POPULATION_MEDIUM)

def fromvar_large():
    return population_pipeline(POPULATION_LARGE)

def fromvar_latency():
    return population_pipeline(POPULATION_SINGLE_ROW)

def q9a():
    # Add all 6 pipelines for a throughput comparison
    # Generate plot in ouptut/q9a.png
    # Return list of 6 throughputs
    t = ThroughputHelper()

    t.add_pipeline('baseline small', len(POPULATION_SMALL), baseline_small)
    t.add_pipeline('baseline medium', len(POPULATION_MEDIUM), baseline_medium)
    t.add_pipeline('baseline large', len(POPULATION_LARGE), baseline_large)

    t.add_pipeline('fromvar small', len(POPULATION_SMALL), fromvar_small)
    t.add_pipeline('fromvar medium', len(POPULATION_MEDIUM), fromvar_medium)
    t.add_pipeline('fromvar large', len(POPULATION_LARGE), fromvar_large)

    throughputs = t.compare_throughput()
    t.generate_plot('output/part2-q9a.png')

    return throughputs


def q9b():
    # Add 2 pipelines for a latency comparison
    # Generate plot in ouptut/q9b.png
    # Return list of 2 latencies
    l = LatencyHelper()
    
    l.add_pipeline('baseline latency', baseline_latency)
    l.add_pipeline('fromvar latency', fromvar_latency)

    latencies = l.compare_latency()
    l.generate_plot('output/part2-q9b.png')

    return latencies


"""
10.
Comment on the plots above!
How dramatic is the difference between the two pipelines?
Which differs more, throughput or latency?
What does this experiment show?

===== ANSWER Q10 BELOW =====
The difference between the throughput and the latency between the two pipelines is very dramatic. Using 
an existing pandas DataFrame is much more efficient than loading the input from a file. I'm assuming 
this is because the DataFrame is in memory already compared to reading the files from the disk. Throughput,
especially, skyrockets when using in memory pandas DataFrame and latency decreases. This experiment shows
that one should utilize their RAM and use existing pandas dataframe over reading-in files from Disk.
===== END OF Q10 ANSWER =====
"""

"""
===== Questions 11-14: Performance Comparison 2 =====

Our second performance comparison will explore vectorization.

Operations in Pandas use Numpy arrays and vectorization to enable
fast operations.
In particular, they are often much faster than using for loops.

Let's explore whether this is true!

11.
First, we need to set up our pipelines for comparison as before.

We already have the baseline pipelines from Q8,
so let's just set up a comparison pipeline
which uses a for loop to calculate the same statistics.

Your pipeline should produce the same answers as in Q6 and Q8.

Create a new pipeline:
- Iterate through the dataframe entries. You can assume they are sorted.
- Manually compute the minimum and maximum year for each country.
- Compute the same answers as in Q6.
- Manually compute the summary statistics for the resulting list (min, median, max, mean, and standard deviation).
"""

def for_loop_pipeline(df):
    # Input: the dataframe from load_input()
    # Return a list of min, median, max, mean, and standard deviation
    
    summary_stats_country = []

    country = df.iloc[0]["Entity"]
    first_year = df.iloc[0]["Year"]
    last_year = df.iloc[0]["Year"]
    first_pop = df.iloc[0]["Population (historical)"]
    last_pop = df.iloc[0]["Population (historical)"]

    for index, row in df.iloc[1:].iterrows():
        curr_country = row["Entity"]
        curr_year = row["Year"]
        curr_population = row["Population (historical)"]

        if curr_country == country:
            last_year = curr_year
            last_pop = curr_population
        else:
            year_span = last_year - first_year
            pop_span = last_pop - first_pop
                
            if year_span > 0: 
                if (pop_span / year_span) > 0:
                    summary_stats_country.append(pop_span / year_span)

            # reset trackers for next country
            country = curr_country
            first_year = curr_year
            last_year = curr_year
            first_pop = curr_population
            last_pop = curr_population

    # Handle last country
    year_span = last_year - first_year
    if year_span > 0:
        year_over_year_increase = (last_pop - first_pop) / year_span
        if year_over_year_increase >= 0:
            summary_stats_country.append(year_over_year_increase)

    # Now compute summary stats
    minimum = min(summary_stats_country)
    maximum = max(summary_stats_country)
    mean = sum(summary_stats_country) / len(summary_stats_country)

    #getting the median
    sorted_values = sorted(summary_stats_country)
    n = len(sorted_values)
    median = (
        (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        if n % 2 == 0
        else sorted_values[n // 2]
    )
    # standard deviation
    variance = sum((x - mean) ** 2 for x in summary_stats_country) / (len(summary_stats_country) - 1)
    sd = math.sqrt(variance)

    return [minimum, median, maximum, mean, sd]


def q11():
    # As your answer to this part, call load_input() and then
    # for_loop_pipeline() to return the 5 numbers.
    # (these should match the numbers you got in Q6.)
    df = load_input("data/population.csv")
    return for_loop_pipeline(df)

"""
12.
Now, let's create our pipelines for comparison.

As before, write 4 pipelines based on the datasets from Q7.
"""

def for_loop_small():
    df = load_input_small()
    return for_loop_pipeline(df)

def for_loop_medium():
    df = load_input_medium()
    return for_loop_pipeline(df)

def for_loop_large():
    df = load_input_large()
    return for_loop_pipeline(df)

def for_loop_latency():
    # For single row, just return a simple computation
    df = load_input_single_row()
    if len(df) > 0:
        return [df['Population (historical)'].iloc[0]] * 5
    return [0, 0, 0, 0, 0]


def q12():
    # Don't modify this part
    _ = for_loop_medium()
    return ["for_loop_small", "for_loop_medium", "for_loop_large", "for_loop_latency"]

"""
13.
Finally, let's compare our two pipelines,
as we did in Q9.

a. Generate a plot in output/part2-q13a.png of the throughputs
    Return the list of 6 throughputs in this order:
    baseline_small, baseline_medium, baseline_large, for_loop_small, for_loop_medium, for_loop_large

b. Generate a plot in output/part2-q13b.png of the latencies
    Return the list of 2 latencies in this order:
    baseline_latency, for_loop_latency
"""

def q13a():
    # Add all 6 pipelines for a throughput comparison
    # Generate plot in ouptut/q13a.png
    # Return list of 6 throughputs
    t = ThroughputHelper()
    t.add_pipeline('baseline small', len(POPULATION_SMALL), baseline_small)
    t.add_pipeline('baseline medium', len(POPULATION_MEDIUM), baseline_medium)
    t.add_pipeline('baseline large', len(POPULATION_LARGE), baseline_large)
    t.add_pipeline('for loop small', len(POPULATION_SMALL), for_loop_small)
    t.add_pipeline('for loop_medium', len(POPULATION_MEDIUM), for_loop_medium)
    t.add_pipeline('for loop large', len(POPULATION_LARGE), for_loop_large)
    
    throughputs = t.compare_throughput()
    t.generate_plot('output/part2-q13a.png')

    return throughputs


def q13b():
    # Add 2 pipelines for a latency comparison
    # Generate plot in ouptut/q13b.png
    # Return list of 2 latencies
    l = LatencyHelper()
    l.add_pipeline('baseline latency', baseline_latency)
    l.add_pipeline('for loop latency', for_loop_latency)
    
    latencies = l.compare_latency()
    l.generate_plot('output/part2-q13b.png')
    
    return latencies


"""
14.
Comment on the results you got!

14a. Which pipelines is faster in terms of throughput?

===== ANSWER Q14a BELOW =====
The vectorized (baseline) pipelines are much faster in terms of throughput than the for loop pipelines, 
especially for the larger datasets. 
===== END OF Q14a ANSWER =====

14b. Which pipeline is faster in terms of latency?

===== ANSWER Q14b BELOW =====
Interestingly, the for loop pipelines are actually slightly faster for latency on the smallest datasets.
===== END OF Q14b ANSWER =====

14c. Do you notice any other interesting observations?
What does this experiment show?

===== ANSWER Q14c BELOW =====
Yeah I'm perplexed by the for loop pipelines having faster latency than the vectorized pipelines, I assume
it has something to do with the small dataset size. Also, for loops are really bad at scaling throughput 
for larger datasets.
===== END OF Q14c ANSWER =====
"""

"""
===== Questions 15-17: Reflection Questions =====
15.

Take a look at all your pipelines above.
Which factor that we tested (file vs. variable, vectorized vs. for loop)
had the biggest impact on performance?

===== ANSWER Q15 BELOW =====
Definitely vectorized vs for loop for throughputs, the boost in performance is unmatched!
===== END OF Q15 ANSWER =====

16.
Based on all of your plots, form a hypothesis as to how throughput
varies with the size of the input dataset.

(Any hypothesis is OK as long as it is supported by your data!
This is an open ended question.)

===== ANSWER Q16 BELOW =====
Throughput tends to scale linearly, or even exponentially in some cases with dataset size 
for vectorized operations, and drastically less for for-loops.
===== END OF Q16 ANSWER =====

17.
Based on all of your plots, form a hypothesis as to how
throughput is related to latency.

(Any hypothesis is OK as long as it is supported by your data!
This is an open ended question.)

===== ANSWER Q17 BELOW =====
Throughput = items/time, Latency = time/item. The relationship looks inverse to me, i.e. as throughput
increases, latency should decrease.
===== END OF Q17 ANSWER =====
"""

"""
===== Extra Credit =====

This part is optional.

Use your pipeline to compare something else!

Here are some ideas for what to try:
- the cost of random sampling vs. the cost of getting rows from the
  DataFrame manually
- the cost of cloning a DataFrame
- the cost of sorting a DataFrame prior to doing a computation
- the cost of using different encodings (like one-hot encoding)
  and encodings for null values
- the cost of querying via Pandas methods vs querying via SQL
  For this part: you would want to use something like
  pandasql that can run SQL queries on Pandas data frames. See:
  https://stackoverflow.com/a/45866311/2038713

As your answer to this part,
as before, return
a. the list of 6 throughputs
and
b. the list of 2 latencies.

and generate plots for each of these in the following files:
    output/part2-ec-a.png
    output/part2-ec-b.png
"""

# Extra credit (optional)

def extra_credit_a():
    raise NotImplementedError

def extra_credit_b():
    raise NotImplementedError

"""
===== Wrapping things up =====

**Don't modify this part.**

To wrap things up, we have collected
your answers and saved them to a file below.
This will be run when you run the code.
"""

ANSWER_FILE = "output/part2-answers.txt"
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

def PART_2_PIPELINE():
    open(ANSWER_FILE, 'w').close()

    # Q1-5
    log_answer("q1", q1)
    log_answer("q2a", q2a)
    # 2b: commentary
    log_answer("q3", q3)
    log_answer("q4a", q4a)
    # 4b: commentary
    log_answer("q5a", q5a)
    log_answer("q5b", q5b)

    # Q6-10
    log_answer("q6", q6)
    log_answer("q7", q7)
    log_answer("q8", q8)
    log_answer("q9a", q9a)
    log_answer("q9b", q9b)
    # 10: commentary

    # Q11-14
    log_answer("q11", q11)
    log_answer("q12", q12)
    log_answer("q13a", q13a)
    log_answer("q13b", q13b)
    # 14: commentary

    # 15-17: reflection
    # 15: commentary
    # 16: commentary
    # 17: commentary

    # Extra credit
    log_answer("extra credit (a)", extra_credit_a)
    log_answer("extra credit (b)", extra_credit_b)

    # Answer: return the number of questions that are not implemented
    if UNFINISHED > 0:
        print("Warning: there are unfinished questions.")

    return UNFINISHED

"""
=== END OF PART 2 ===

Main function
"""

if __name__ == '__main__':
    log_answer("PART 2", PART_2_PIPELINE)
