from pyspark import SparkContext, SparkConf
import sys
import time
import json
import math
import operator


def reading_file():
    """
    Read file and do word counting per business
    :return: Word counts
    eg ('WEeMwRLhgCyO1b4kikVcuQ', {'mushrooms': 7, 'opinion': 4, ...})
    """
    # Read stopwords file
    with open(stopwords_file) as file:
        stopwords = file.read().splitlines()

    # Parse file, remove puntuations stopwords
    # eg reviews = [('zK7sltLeRRioqYwgLiWUIA', "second time i've first time whatever burger side"), ...]
    reviews = lines.filter(lambda line: len(line) != 0) \
        .map(lambda s: (json.loads(s)['business_id'], json.loads(s)['text'])) \
        .filter(lambda x: x[0] is not None and x[1] is not None and x[0] != "" and x[1] != "") \
        .mapValues(lambda line: line.translate({ord(i): None for i in '([,.!?:;])&\"0123456789'}).lower()) \
        .mapValues(lambda line: " ".join([word for word in line.split() if word not in stopwords]))

    # Concat according to each business
    reviews_concat = reviews.reduceByKey(lambda a, b: str(a) + " " + str(b))

    # Word count, using business id and word as composite key eg (('djAWtGq2IxKaxMIMw-P58A', 'due'), 10)
    # Also I dont want any words appearing only once
    counts = reviews.flatMapValues(lambda line: line.split(" ")) \
        .map(lambda x: (x, 1)) \
        .filter(lambda x: x[0][1] != "") \
        .reduceByKey(lambda a, b: a + b) \
        .filter(lambda x: x[1] > 1)

    # Change key/value, group by key, convert to dict
    # eg ('WEeMwRLhgCyO1b4kikVcuQ', {'mushrooms': 7, 'opinion': 4, ...})
    counts1 = counts.map(lambda x: (x[0][0], (x[0][1], x[1])))\
        .groupByKey()\
        .map(lambda x: (x[0], dict(x[1].data)))\
        .collect()
    return counts1


def tfhelper(words: dict):
    """
    Calculate TF of a word dictionary
    :param words:
    :return: result dictionary
    eg {'mushrooms': 0.041666666666666664, 'opinion': 0.023809523809523808, 'make': 0.06547619047619048, ...}
    """
    # Find max value of the words dictionary
    max_value = max(words.items(), key=operator.itemgetter(1))[1]
    result = {word: value / max_value for word, value in words.items()}
    return result


def idf_function(wordcount_rdd, N):
    """
    Calculate IDF for each word
    :param wordcount_rdd:
    :param N: Number of businesses/ documents
    :return: IDF dictionary
    eg {'everything': 0.7432705137137011, 'let': 1.2765910361657753, 'sauces': 3.11411633920867, 'bobby': 6.982500694663834, 'tasteless': 4.767761845870821, ...}
    """
    # Emit (word, 1) if the word appears in a document/business from the dictionary
    idf = wordcount_rdd.values() \
        .flatMap(lambda dictx: dictx.keys()) \
        .map(lambda x: (x, 1))
    # Regular word count
    idf1 = idf.reduceByKey(lambda a, b: a + b)
    # Apply IDF equation
    idf2 = idf1.mapValues(lambda x: math.log2(N / x)).collect()
    result = dict(idf2)
    return result


if __name__ == '__main__':

    # ========================================== Initializing ==========================================
    time1 = time.time()
    conf = SparkConf()
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "4g")
    conf.set("spark.master", "local[*]")
    conf.set("spark.app.name", "task1")
    conf.set("spark.driver.maxResultSize", "4g")
    sc = SparkContext.getOrCreate(conf)
    sc.setLogLevel("ERROR")
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    stopwords_file = sys.argv[3]

    # ============================ Read file and Word Count ==========================
    lines = sc.textFile(input_file).distinct().persist()
    wordcount = reading_file()
    totaltime = time.time() - time1
    print("Duration Read and Count: " + str(totaltime))

    # ============================ TFIDF ==========================
    wordcount_rdd = sc.parallelize(wordcount)
    tf = wordcount_rdd.mapValues(lambda x: tfhelper(x))
    idf = idf_function(wordcount_rdd, len(wordcount))

    # Multiply tf by idf according to IDF dict
    # eg ('WEeMwRLhgCyO1b4kikVcuQ', {'mushrooms': 0.1336377936411145, 'opinion': 0.06398903922325573, 'make': 0.033220508168680656, ...})
    tfidf = tf.mapValues(lambda dictx: {key: (value * idf.get(key)) for key, value in dictx.items()})
    totaltime = time.time() - time1
    print("Duration TFIDF: " + str(totaltime))

    # ========================== Business Profile ==========================
    # Sort dictionary by TFIDF values and select top 200 words
    # eg ('WEeMwRLhgCyO1b4kikVcuQ', {'ketchup': 3.0334656552186, 'burger': 2.9076654617412436, 'fries': 0.9789390946357696, ...})
    busi_profile = tfidf.mapValues(lambda d: dict(sorted(d.items(), key=lambda d: d[1], reverse=True)[0:200])).collect()

    totaltime = time.time() - time1
    print("Duration business profile: " + str(totaltime))

    # ========================== User Profile ==========================
    # eg ('u_wqt9RshdZsoj8ikLqoEQ', {'scary', "they're", 'reflect', 'imitation', 'slushee', ... for 8710 words})

    # Extract keywords only
    busi_profile_rdd = sc.parallelize(busi_profile)
    busi_profile_rdd1 = busi_profile_rdd.mapValues(lambda d: set(d.keys())).persist()
    busi_profile_noscore = busi_profile_rdd1.collect()

    # Get business/user pairs
    user_profile_rdd = lines.filter(lambda line: len(line) != 0) \
        .map(lambda s: (json.loads(s)['business_id'], json.loads(s)['user_id'])) \
        .filter(lambda x: x[0] is not None and x[1] is not None and x[0] != "" and x[1] != "")
    lines.unpersist()

    # Joining, throw away keys, and union the keywords by reduce
    user_profile = user_profile_rdd.join(busi_profile_rdd1) \
        .values() \
        .groupByKey(lambda a, b: a.union(b)) \
        .collect()
    busi_profile_rdd1.unpersist()

    totaltime = time.time() - time1
    print("Duration user profile: " + str(totaltime))

    # ========================== Writing results ==========================
    with open(output_file, "w") as file:
        for item in busi_profile_noscore:
            entry = {"business_id": item[0], "words": list(item[1])}
            file.write(json.dumps(entry))
            file.write("\n")

        for item in user_profile:
            entry = {"user_id": item[0], "words": list(item[1])}
            file.write(json.dumps(entry))
            file.write("\n")

    # ========================================== Ending ==========================================
    totaltime = time.time() - time1
    print("Duration: " + str(totaltime))
    sc.stop()