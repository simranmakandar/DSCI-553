#import time
import sys
import time
import itertools
#import json
#import re
from pyspark import SparkContext

#system arguments: take task1.py, case, support, input_file : small1.csv, output_file: output.txt
commandArgs = sys.argv

#SON algorithm using the Apache Spark Framework.

case = int(commandArgs[1])
support =  int(commandArgs[2])
input_file = commandArgs[3]
output_file =  commandArgs[4]

partition_num = 0

if (len(commandArgs) != 5):
    print("Check if the number of arguments passed are = 5.")
    print("Usage of command line arguments: spark-submit task1.py <case_number> (Hint: Either 1 or 2) <support integer> <input_file> <output_file>")
    exit(1)

# Input format:
# 1. Case number: Integer that specifies the case. 1 for Case 1 and 2 for Case 2 .
# 2. Support: Integer that defines the minimum count to qualify as a frequent itemset.
# 3. Input file path: This is the path to the input file including path, file name and extension.
# 4. Output file path: This is the path to the output file including path, file name and extension.

#out of PCY, multihash, apriori, we will choose apriori for finding frequent sets
def apriorifunction(iterator):
    partition_support = support / partition_num
    local_baskets = list()
    counts = dict()
    local_frequentsets = list()
    #counting singletons from the data
    for basket in iterator:
        local_baskets.append(basket)
        for item in basket:
            if item in counts:
                counts[item] += 1
            else:
                counts[item] = 1
    # emitting frequent_items further on
    previous_frequentsets = list()
    for item, count in counts.items():
        if count >= partition_support:
            local_frequentsets.append(frozenset([item]))
            previous_frequentsets.append(frozenset([item]))
    # generate larger frequent item sets
    size = 2
    while len(previous_frequentsets) > 0:
        previous_frequentsets = emitfurtherFrequentsets(local_baskets, previous_frequentsets, size)
        local_frequentsets += previous_frequentsets
        size += 1
    return local_frequentsets

#next freq itms

def emitfurtherFrequentsets(baskets, previous_frequentsets, size):
    candidates = set()
    for combination in itertools.combinations(previous_frequentsets, 2):
        union = combination[0].union(combination[1])
        if len(union) == size: #combination
            candidates.add(frozenset(union))
    counts = dict()
    for basket in baskets:
        for candi in candidates:
            if candi.issubset(basket):
                if candi in counts:
                    counts[candi] += 1
                else:
                    counts[candi] = 1
    partition_support = support / partition_num #support per each partition
    newer_frequentsets = list() #new frequent sets , list() function
    for item, count in counts.items():
        if count >= partition_support: #if count is greater than or equal to 't'
            newer_frequentsets.append(item) #adding new freq sets to newer_frequentsets
    return newer_frequentsets

#import default dict

def tupleKey(items):
    return (len(items),) + items
#return length (items) + itms

def write_output(tup_list, fout, label):
    fout.write(label+':\n')
    current_size = 1 #output lines for different size. singletons, pairs, triplets etc
    outputting_line = ''
    for item_set in tup_list:
        if len(item_set) == current_size:
            if current_size == 1:
                outputting_line += ('(\''+item_set[0]+'\'),')
            else:
                outputting_line += (str(item_set)+',')
        else: #else write to out in
            fout.write(outputting_line[0:-1]+'\n\n')
            outputting_line = (str(item_set)+',')
            current_size += 1
    if outputting_line: #written output here
        fout.write(outputting_line[0:-1]+'\n\n')

#count candidates , basket iter

def count_of_Candidates(basket):
    candidatesin_basket = list()
    for candi in candidates.value:
        if set(candi).issubset(basket):
            candidatesin_basket.append((candi, 1))
    return candidatesin_basket

#time funct for displaying execution elapsed time
start = time.time()
sc = SparkContext(master='local[*]', appName='hw2_task1')
min_partition_num = max(support//4, 1)
review_file = sc.textFile(input_file, min_partition_num)
partition_num = review_file.getNumPartitions()
review_file = review_file.filter(lambda u: u != 'user_id,business_id') #use this in filter function

if case == 2:
    baskets = review_file.distinct().map(lambda u: [str(u.split(',')[1]), str(u.split(',')[0])]) \
        .groupByKey().mapValues(set).values().persist()
#business1: [user11, user12, user13, ...]business2: [user21, user22, user23, ...]business3: [user31, user32, user33, ...]
else:
    baskets = review_file.distinct().map(lambda u: [str(u.split(',')[0]), str(u.split(',')[1])]) \
        .groupByKey().mapValues(set).values().persist()
#user1: [business11, business12, business13, ...]user2: [business21, business22, business23, ...]user3: [business31, business32, business33, ...]

candidates_list = baskets.mapPartitions(apriorifunction).map(lambda u: (tuple(sorted(list(u))), 1)) \
    .reduceByKey(lambda x, y: 1).keys().collect()
candidates_list.sort(key=tupleKey)
fout = open(output_file, 'w')
write_output(tup_list=candidates_list, fout=fout, label='Candidates')
candidates = sc.broadcast(candidates_list)

#check broadcast function

#frequent sets in baskets , reducebykey using lambda anonymous
frequent_sets = baskets.flatMap(count_of_Candidates).reduceByKey(lambda x, y: x+y) \
    .filter(lambda u: u[1]>=support).keys().collect()  #frequent items are those that have support>=s.
frequent_sets.sort(key=tupleKey)
write_output(tup_list=frequent_sets, fout=fout, label='Frequent Itemsets')
fout.close()
#close fout
print('Duration: ' + str(time.time()-start))

# Output format:
# 1. Console output - Runtime: the total execution time from loading the file till finishing writing the
# output file
# You need to print the runtime in the console with the “Duration” tag: “Duration: <time_in_seconds>”,
# e.g., “Duration: 100.00”
