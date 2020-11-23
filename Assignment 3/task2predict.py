from pyspark import SparkContext, SparkConf
import sys
import time
import json
import math


def reading_files():
    """
    Read model file to the same format as before. Read test file and drop stuff not in model
    :return: busi_profile, user_profile, test_pairs
    test_pairs eg (user_id1, business_id1), ...
    """
    busi_profile = lines.filter(lambda line: len(line) != 0 and 'business_id' in json.loads(line)) \
        .map(lambda s: (json.loads(s)['business_id'], json.loads(s)['words'])) \
        .filter(lambda x: x[0] is not None and x[1] is not None and x[0] != "" and x[1] != "") \
        .mapValues(lambda x: set(x)) \
        .persist()
    user_profile = lines.filter(lambda line: len(line) != 0 and 'user_id' in json.loads(line)) \
        .map(lambda s: (json.loads(s)['user_id'], json.loads(s)['words'])) \
        .filter(lambda x: x[0] is not None and x[1] is not None and x[0] != "" and x[1] != "") \
        .mapValues(lambda x: set(x)) \
        .persist()

    # Read test file
    test_pairs = test_lines.filter(lambda line: len(line) != 0) \
        .map(lambda s: (json.loads(s)['user_id'], json.loads(s)['business_id'])) \
        .filter(lambda x: x[0] is not None and x[1] is not None and x[0] != "" and x[1] != "")

    # Filter out businesses/users not in model
    users = user_profile.keys().collect()
    businesses = busi_profile.keys().collect()
    test_pairs = test_pairs.filter(lambda x: x[0] in users and x[1] in businesses)
    return busi_profile, user_profile, test_pairs


def cosine_sim(a: set, b: set):
    """
    Cosine similarity of two sets
    :param a:
    :param b:
    :return: Cosine similarity
    """
    return len(a.intersection(b)) / math.sqrt(len(a) * len(b))


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
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    output_file = sys.argv[3]

    # ============================ Read model/test file ==========================
    lines = sc.textFile(model_file).distinct().persist()
    test_lines = sc.textFile(test_file).distinct()
    busi_profile, user_profile, test_pairs = reading_files()
    lines.unpersist()

    totaltime = time.time() - time1
    print("Duration reading: " + str(totaltime))

    # ============================ Cosine similarity ==========================
    # Join test pairs with user profile, and make business_id the key
    # eg ('bKbYRUZKDYonSPOjzchJJg', ('aZtJzH3fRIRzrGnQRIVaRg', {'blah', 'bakery', 'ladies', 'sunday', ...}))
    test_pairs1 = test_pairs.join(user_profile).map(lambda x: (x[1][0], (x[0], x[1][1])))
    # Join with business profile, and make the key back to (userid, businessid)
    test_pairs2 = test_pairs1.join(busi_profile).map(lambda x: ((x[1][0][0], x[0]), (x[1][1], x[1][0][1])))

    # Compute final result
    # eg (('75COHfu_drTAx1G9rtZnAA', 'dlTgsi51JXinnmijEVuITw'), 0.2103865254412277)
    result = test_pairs2.map(lambda x: ((x[0][0], x[0][1]), cosine_sim(x[1][0], x[1][1])))\
        .filter(lambda x: x[1] >= 0.01)\
        .collect()

    # ============================ Write results ==========================
    with open(output_file, "w") as file:
        for item in result:
            entry = {"user_id": item[0][0], "business_id": item[0][1], "sim": item[1]}
            file.write(json.dumps(entry))
            file.write("\n")

    totaltime = time.time() - time1
    print("Duration: " + str(totaltime))

