import re
import json
import sys
from pyspark import SparkContext
from pyspark.sql.types import IntegerType

commandArgs = sys.argv

if (len(commandArgs) != 7):
    print("Check if the number of arguments passed are = 7.")
    print("Usage of command line arguments: spark-submit task1.py review.json output_file <stopwords> <year> <m> <n>")
    exit(1)

# creating list here out of stopwords_file 
stopWordsList = open(commandArgs[3], 'r').readlines()
stopWordsList = [x.rstrip() for x in stopWordsList]
#print(stopWordsList)
#The stopwords are created and ouput is a list of all stopwords separated by a comma


# Function to get the valid lower case words from the given sentence
def word_tokenize(s):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", s.lower())

sc = SparkContext.getOrCreate()

# creating review RDD from data file
#spark.read.option("mode", "MALFORMED").option("inferSchema", "true").json(commandArgs[1])
#review_source_rdd = spark.read.json(commandArgs[1], multiLine=True)
review_source_rdd = sc.textFile(commandArgs[1]).map(json.loads)

# Getting total number of reviews given by users
solA = review_source_rdd.count()
#print('Total Number of Reviews : ' + str(solA))

# Getting total number of reviews by users in given year (given from commandArgs)
solB = review_source_rdd.filter(lambda x: x['date'][:4] == commandArgs[4]).count()
#print('Total Number of Reviews in ' + commandArgs[4] + ' : ' + str(solB))

# Getting number of distinct businesses who have user reviews
solC = review_source_rdd.map(lambda x: x['business_id']).distinct().count()
#print('Total number of distinct users who have written reviews is : ' + str(solC))

# Getting number of top 'm' users having largest number of reviews in descending order
solD = review_source_rdd.map(lambda x: (x['user_id'], 1)).reduceByKey(lambda x, y: x + y).map(
    lambda x: (x[1], x[0])).sortByKey(False).map(lambda x: (x[1], x[0])).take(int(commandArgs[5]))
#print('Top ' + str(commandArgs[5]) + ' users having largest number of reviews : ' + str(solD))

# Getting Top n frequent words in the review text 
solE = review_source_rdd.flatMap(lambda x: word_tokenize(x['text'])).filter(lambda x: x not in stopWordsList).map(
    lambda x: (x, 1)).reduceByKey(lambda x, y: x + y).map(lambda x: (x[1], x[0])).sortByKey(False).map(
    lambda x: (x[1], x[0])).map(lambda x: x[0]).take(int(commandArgs[6]))
#print('Top ' + str(commandArgs[6]) + ' frequent words in the text : ' + str(solE))

solutionDic = {
    'A': solA, 'B': solB, 'C': solC, 'D': solD, 'E': solE
}

solution = json.dumps(solutionDic)
#print(solution)

# Writing final output to output_file
f = open(commandArgs[2], 'a')
#f.write('Task 1 Results : \n')
f.write(solution + '\n')
f.close()
