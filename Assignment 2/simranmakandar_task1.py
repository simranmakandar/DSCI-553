#import time
import sys
import time
import itertools
#import json
#import re
from pyspark import SparkContext

#system arguments: take task1.py, case, support, input_file : small1.csv, output_file: output.txt
commandArgs = sys.argv

case = int(commandArgs[1])
support =  int(commandArgs[2])
input_file = commandArgs[3]
output_file =  commandArgs[4]

partition_num = 0

if (len(commandArgs) != 5):
    print("Check if the number of arguments passed are = 5.")
    print("Usage of command line arguments: spark-submit task1.py <case_number> (Hint: Either 1 or 2) <support integer> <input_file> <output_file>")
    exit(1)

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


def emitfurtherFrequentsets(baskets, previous_frequentsets, size):
    candidates = set()
    for comb in itertools.combinations(previous_frequentsets, 2):
        union = comb[0].union(comb[1])
        if len(union) == size:
            candidates.add(frozenset(union))
    counts = dict()
    for basket in baskets:
        for candi in candidates:
            if candi.issubset(basket):
                if candi in counts:
                    counts[candi] += 1
                else:
                    counts[candi] = 1
    partition_support = support / partition_num
    new_frequent_sets = list()
    for item, count in counts.items():
        if count >= partition_support:
            new_frequent_sets.append(item)
    return new_frequent_sets


def tupleKey(items):
    return (len(items),) + items


def output(tuple_list, fout, label):
    fout.write(label+':\n')
    curr_size = 1
    line = ''
    for item_set in tuple_list:
        if len(item_set) == curr_size:
            if curr_size == 1:
                line += ('(\''+item_set[0]+'\'),')
            else:
                line += (str(item_set)+',')
        else:
            fout.write(line[0:-1]+'\n\n')
            line = (str(item_set)+',')
            curr_size += 1
    if line:
        fout.write(line[0:-1]+'\n\n')


def countCandidates(basket):
    candi_in_basket = list()
    for candi in candidates.value:
        if set(candi).issubset(basket):
            candi_in_basket.append((candi, 1))
    return candi_in_basket


start = time.time()
sc = SparkContext(master='local[*]', appName='hw2_task1')
min_partition_num = max(support//4, 1)
reviews = sc.textFile(input_file, min_partition_num)
partition_num = reviews.getNumPartitions()
reviews = reviews.filter(lambda x: x != 'user_id,business_id')

if case == 1:
    baskets = reviews.distinct().map(lambda x: [str(x.split(',')[0]), str(x.split(',')[1])]) \
        .groupByKey().mapValues(set).values().persist()
else:
    baskets = reviews.distinct().map(lambda x: [str(x.split(',')[1]), str(x.split(',')[0])]) \
        .groupByKey().mapValues(set).values().persist()

candidates_list = baskets.mapPartitions(apriorifunction).map(lambda x: (tuple(sorted(list(x))), 1)) \
    .reduceByKey(lambda a, b: 1).keys().collect()
candidates_list.sort(key=tupleKey)
fout = open(output_file, 'w')
output(tuple_list=candidates_list, fout=fout, label='Candidates')
candidates = sc.broadcast(candidates_list)
frequent_sets = baskets.flatMap(countCandidates).reduceByKey(lambda a, b: a+b) \
    .filter(lambda x: x[1]>=support).keys().collect()
frequent_sets.sort(key=tupleKey)
output(tuple_list=frequent_sets, fout=fout, label='Frequent Itemsets')
fout.close()

print('Duration: ' + str(time.time()-start))
