from collections import OrderedDict
import json
from pyspark.sql.types import IntegerType
import sys
# from pyspark import SparkContext
# import time
from pyspark import SparkContext
import time

commandArgs = sys.argv

if len(commandArgs) != 6:
    print('Check the number of arguments passed. Hint: number should be 6')
    print('Usage spark-submit task3.py <inputfile> <outputfile> <partitiontype> <partitions> <n>')
    sys.exit(1)

    #taking sys argms from command line.
#'Usage in the command arguments: spark_submit task2.py <review_file> <business_file > <output_file> <spark or no_spark> <n>
#task2.py = name convention of second task
#<review_file> = name of review file on your system. "review_json" in my case
#<business_file> = name of business file on your system. "business_json" in my case
#<output_file> = the output file written by this python script to present the output in json type format
#The encoding in these .json files is UTF-8. encoding = 'utf8'

sc = SparkContext()
numofpartitions = int(commandArgs[4])

n = int(commandArgs[5])
partition_type = commandArgs[3]

#sc = SparkContext()
#numofpartitions = int(commandArgs[4])

#n = int(commandArgs[5])
#partition_type = commandArgs[3]

solution = OrderedDict()


def partition_func(key):
    return hash(key) % numofpartitions

#default or customized type partitions 
if partition_type == 'default':
    rdd = sc.textFile(commandArgs[1])
    json_rdd = rdd.map(json.loads)
    transformation = json_rdd.map(lambda x: x['business_id']).map(lambda x: (x, 1))
    #created transformation here
    n_items = transformation.glom().map(len).collect()
    #glom() usage
    result = transformation.reduceByKey(lambda x, v: x + v).filter(lambda x: x[1] > n).collect()
    #transformation result
    solution['n_partitions'] = json_rdd.getNumPartitions()
    solution['n_items'] = n_items
    #n_items into solution
    solution['result'] = result


else:
    rdd = sc.textFile(commandArgs[1], numofpartitions)
    json_rdd = rdd.map(json.loads)
    transformation = json_rdd.map(lambda x: x['business_id']).map(lambda x: (x, 1)) \
        .partitionBy(numofpartitions, partition_func)
    n_items = transformation.glom().map(len).collect()
    result = transformation.reduceByKey(lambda x, v: x + v).filter(lambda x: x[1] > n).collect()
    solution['n_partitions'] = numofpartitions
    solution['n_items'] = n_items
    #n_items into solution
    solution['result'] = result

with open(commandArgs[2], 'w') as file:
    json.dump(solution, file)


#check runtime of both separately: defualt, customized
