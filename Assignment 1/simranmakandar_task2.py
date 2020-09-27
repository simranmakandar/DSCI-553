import re
import sys
#import time
#For Json import
import json
from collections import defaultdict
#from pyspark.sql import SparkContext
#from pyspark.sql.types import IntegerType
from pyspark import SparkContext
import time


commandArgs = sys.argv
if len(commandArgs)!=6:
    print('Please check the number of command arguments. ')
    print('HINT: The number of command Arguments must be equal to 6')
    print ('Usage spark_submit task2.py <review_file> <business_file > <output_file> <if_spark> <n>')
    sys.exit(1)

    
#commandArgs = sys.argv
#if len(commandArgs)!=6:
    #print('Please check the number of command arguments. ')
    #print('HINT: The number of command Arguments must be equal to 6')
    #print ('Usage spark_submit task2.py <review_file> <business_file > <output_file> <if_spark> <n>')
    #sys.exit(1)
    
sc = SparkContext()   

if_spark = commandArgs[4]
count = int(commandArgs[5])
solution = {}
def func(x):
    res = []
    if x[1] is not None:
        temp = x[1].strip()
        temp = temp.split(',')
        for values in temp:
            values = values.strip()
            res.append((x[0],values))
            
    
    return res
    
    
#if-else between spark and no_spark
if if_spark == "spark":
    review_rdd = sc.textFile(sys.argv[1])
    business_rdd = sc.textFile(sys.argv[2])
    #review_df = sc.rdd
    #directly read into rdd
    review_rdd_json = review_rdd.map(json.loads)
    business_rdd_json = business_rdd.map(json.loads)
    l1 = review_rdd_json.map(lambda x:(x['business_id'],x['stars']))
    result =l1.join(business_rdd_json.map(lambda x:(x['business_id'],(x['categories']))).flatMap(func))\
    .map(lambda x:(x[1][1],x[1][0])).aggregateByKey((0,0),lambda u,x:(u[0]+1,u[1]+x),lambda a,b:((a[0]+b[0]),(a[1]+b[1])))\
    .map(lambda x:(x[0],(x[1][1]/x[1][0]))).sortBy(lambda x:(-x[1],x[0])).take(count)
    
    solution['result'] = result
 
else:
    with open(commandArgs[1],'r') as f1:
        review_file = f1.readlines()
    with open(commandArgs[2],'r') as f1:
        business_file = f1.readlines()
        #readLines() for without spark
        review_file_details = []

    for values in review_file:
        values = json.loads(values)
        review_file_details.append((values['business_id'],values['stars']))
        business_file_data = []
        #review and business file creation
    for values in business_file:
        values = json.loads(values)
        if values['categories'] is not None:
            categories = values['categories'].strip()
            categories = categories.split(',')
            for category in categories:
                category = category.strip()
                business_file_data.append((values['business_id'],category))
    d = defaultdict(list)
    
    #RF details
    for row in review_file_details:
        d[row[0]].append(row[1])
    result =defaultdict(list)
    for row in business_file_data:
        if row[0] in d:
            for values in d[row[0]]:
               
                result[row[1]].append(values)
    avg_dict ={}
    for key,value in result.items():
        #avg = sum/count
        #avg_dict[key] = sum(value)/len(value)
        avg_dict[key] = sum(value)/len(value)
       
        
    #sorted result    
    result = sorted(avg_dict.items(), key=lambda x: (x[0]))
    result = sorted(result, key=lambda x: (x[1]) ,reverse=True)[:count]
    solution['result'] = result

with open(sys.argv[3],'w') as file:
    json.dump(solution,file)
#written to json to output



