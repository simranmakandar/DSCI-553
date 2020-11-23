from pyspark import SparkContext, SparkConf
import sys
import time
import json
import random
import math
from collections import defaultdict


def initialize():
    """
    Read file and make dictionaries to shorten the long user and business IDs
    :return: reviews, businesses_inv, users_inv, businesses_dict, users_dict
    eg for reviews [(20513, 2236, 5.0), (24264, 7332, 4.0), (16861, 9483, 5.0), ...]

    Number of businesses = 10253
    Number of users = 26184
    """
    # Get reviews
    # eg ('VTbkwu0nGwtD6xiIdtD00Q', 'fjMXGgOr3aCxnN48kovZ_Q', 5.0)
    reviews_long = lines.filter(lambda line: len(line) != 0) \
        .map(lambda line: (json.loads(line))) \
        .map(lambda x: (x['user_id'], x['business_id'], x['stars'])) \
        .filter(lambda x: x[0] is not None and x[1] is not None and x[2] is not None and x[0] != "" and x[1] != "" and x[2] != "") \
        .persist()

    # Get lists of unique businesses and users as inverse dictionary from integer code to ID
    businesses_inv = tuple(reviews_long.map(lambda x: x[1]).distinct().collect())
    users_inv = tuple(reviews_long.map(lambda x: x[0]).distinct().collect())

    # Make dictionaries to convert long IDs to integer code
    businesses_dict = defaultdict(int)
    for i in range(len(businesses_inv)):
        businesses_dict[businesses_inv[i]] = i
    users_dict = defaultdict(int)
    for i in range(len(users_inv)):
        users_dict[users_inv[i]] = i

    # Get a shorter version of the reviews_long crap
    # eg [(20513, 2236, 5.0), (24264, 7332, 4.0), (16861, 9483, 5.0), ...]
    reviews = reviews_long.map(lambda x: (users_dict[x[0]], businesses_dict[x[1]], x[2])) \
        .persist()
    reviews_long.unpersist()
    return reviews, businesses_inv, users_inv, businesses_dict, users_dict


def corated_helper(business_reviews_tuple, a, b):
    """
    Check if business pair has more than 3 corated users
    :param business_reviews_tuple: Tuple of business reviews
    :param a: Business A's number
    :param b: Business B's number
    :return: True if corated users >= 3
    """
    if len(set(business_reviews_tuple[a].keys()).intersection(set(business_reviews_tuple[b].keys()))) >= 3:
        return True
    return False


def item_based():
    """
    Case 1 item based. Get candidate business pairs with more than 3 corated users
    :return: Candidate pairs and business_reviews_tuple
    candidate_pairs eg [(0, 7336), (0, 9492), (0, 5908), (0, 5152), (0, 6622)]
    business_reviews_tuple eg ({24267: 1.0, 5670: 3.0, 15085: 2.0, 7731: 3.0, 300: 3.0, ...}, {...})
    """
    # Group the reviews by business
    business_reviews = reviews.map(lambda x: (x[1], (x[0], x[2])))\
        .groupByKey()\
        .map(lambda x: (x[0], dict(x[1].data)))\
        .persist()

    # Output as a tuple of dict, index = business number
    # eg ({24267: 1.0, 5670: 3.0, 15085: 2.0, 7731: 3.0, 300: 3.0, ...}, {...})
    business_reviews_tuple = tuple(business_reviews.sortByKey().map(lambda x: x[1]).collect())

    # Generate all pairs of businesses
    businesses = business_reviews.map(lambda x: x[0])
    all_pairs = businesses.cartesian(businesses)\
        .filter(lambda x: x[0] < x[1])

    # Remove those who has less than 3 corated users
    candidate_pairs = all_pairs.filter(lambda x: corated_helper(business_reviews_tuple, x[0], x[1]))
    # print("Number of candidate pairs: " + str(candidate_pairs.count()))
    return candidate_pairs, business_reviews_tuple


def format_output(final_result):
    """
    Format output file
    :return: List of dictionaries
    :param final_result: List of tuples
    eg [{'b1': businessid1, 'b2': businessid2, 'sim': 0.07693}, {'b1': businessid1, 'b2': businessid2, 'sim': 0.052632}, ...]
    """
    result = []
    for item in final_result:
        result.append({'b1': item[0], 'b2': item[1], 'sim': item[2]})
    return result


def pearson_helper(data, a, b):
    """
    Pearson for item based
    :param data: Tuple of business reviews or user reviews
    :param a: User/Business A's number
    :param b: User/Business B's number
    :param avg_a: User/Business A's average rating
    :param avg_b: User/Business B's average rating
    :return: Pearson correlation value
    """
    # Find corated items
    corate_set = set(data[a].keys()).intersection(set(data[b].keys()))

    # Get the normalized vectors of a and b
    vec_a_pre = [data[a].get(item) for item in corate_set]
    vec_b_pre = [data[b].get(item) for item in corate_set]
    avg_a = sum(vec_a_pre) / len(vec_a_pre)
    avg_b = sum(vec_b_pre) / len(vec_b_pre)
    vec_a = list(map(lambda x: x - avg_a, vec_a_pre))
    vec_b = list(map(lambda x: x - avg_b, vec_b_pre))

    numerator = sum([x * y for x, y in zip(vec_a, vec_b)])
    denominator = math.sqrt(sum([x ** 2 for x in vec_a])) * math.sqrt(sum([x ** 2 for x in vec_b]))
    if denominator == 0:
        return 0
    else:
        result = numerator / denominator
    return result


def user_based():
    """
    Group the reviews by business and remove ratings and business code to prepare for minhash
    :return: business_reviews
    eg [{15107, 9477, 773, ...}, {13698, 19075, 14980, ...}]
    The index of the list is the business code!
    """
    business_reviews = reviews.map(lambda x: (x[1], x[0])) \
        .groupByKey() \
        .map(lambda x: (x[0], set(x[1].data))) \
        .sortByKey() \
        .map(lambda x: x[1]) \
        .collect()
    return business_reviews


def minhash(table, a, b, num_business):
    """
    * Copied from task 1

    Minhash method
    :param table: Characteristic table with row = users
    :param a:
    :param b:
    :param num_business: Total number of or users
    :return: Minhash of one hash function
    eg [749, 724, 194, 11, 103, 115, 216, 955, 192, 322, 105, 704, 32, ...]
    """
    table = table.value
    m = len(table)
    p = 479001599
    result = [(m + 10) for _ in range(num_business)]
    for row in range(len(table)):
        hashvalue = ((a * row + b) % p) % m
        for business_id in table[row]:
            if hashvalue < result[business_id]:
                result[business_id] = hashvalue
    return result


def hash_func_generate(num_func):
    """
    * Copied from task 1

    Generate hash functions a and b
    :return: list of a and b pairs
    eg [[983, 294], [1777, 208], [557, 236], ...]
    """
    result = []
    primes = random.sample(range(10000000, sys.maxsize), num_func)
    b = random.sample(range(10000000, sys.maxsize), num_func)
    for i in range(0, num_func):
        result.append([primes[i], b[i]])
    return result


def user_based_minhash():
    """
    Minhashing as in task 1 but with users instead of businesses
    :return: Minhashing results
    """
    business_reviews_broad = sc.broadcast(business_reviews)
    hash_ab = hash_func_generate(num_func=30)
    hash_ab_rdd = sc.parallelize(hash_ab)
    result_minhash = hash_ab_rdd.map(lambda x: minhash(business_reviews_broad, x[0], x[1], len(users_inv))).collect()
    business_reviews_broad.destroy()
    return result_minhash


def lsh_signature(minhashes):
    """
    * Copied from task 1

    LSH with band size of 1 rows (r = 1)
    :param minhashes: 2d list of minhashes
    :return: Set of Candidate pairs
    eg {(241, 235), (3242 ,2352), ...}
    """
    result = set()
    num_business = len(minhashes[0])
    for i in range(num_business):
        for j in range(i + 1, num_business):
            if minhashes[0][i] == minhashes[0][j]:
                result.add((i, j))
    return result


def user_based_lsh():
    """
    Do LSH user based as in Task 1, improved union step
    :return:
    """
    R = 1   # Number of rows in a band
    lsh_input = []
    for i in range(0, len(result_minhash), R):
        lsh_input.append(result_minhash[i:i+R])
    lsh_input1 = sc.parallelize(lsh_input)
    result_lsh = lsh_input1.map(lsh_signature)\
        .reduce(lambda a, b: a.union(b))
    return result_lsh


def jaccard(pair):
    """
    * Copied and edited for task 1

    Find jaccard of a pair of users
    :param pair: user num (int)
    :return: tuple of users and their Jaccard similarity
    eg (436, 21398, 0.05)
    """
    # Look up user_reviews_tuple for the sets
    a = set(user_reviews_tuple[pair[0]].keys())
    b = set(user_reviews_tuple[pair[1]].keys())
    union = len(a.union(b))
    if union == 0:
        similarity = 0
    else:
        similarity = len(a.intersection(b)) / union
    return tuple((pair[0], pair[1], similarity))


def user_based_after():
    """
    Case 2 user based. Get candidate user pairs with more than 3 corated businesses. As in item_based()
    :return: Candidate pairs and user_reviews_tuple
    """
    # Group the reviews by user
    user_reviews = reviews.map(lambda x: (x[0], (x[1], x[2])))\
        .groupByKey()\
        .map(lambda x: (x[0], dict(x[1].data)))\
        .persist()

    # Output as a tuple of dict, index = business number
    # eg ({24267: 1.0, 5670: 3.0, 15085: 2.0, 7731: 3.0, 300: 3.0, ...}, {...})
    user_reviews_tuple = tuple(user_reviews.sortByKey().map(lambda x: x[1]).collect())

    # Remove those who has less than 3 corated users
    candidate_pairs = result_lsh_rdd.filter(lambda x: corated_helper(user_reviews_tuple, x[0], x[1]))
    return candidate_pairs, user_reviews_tuple


def format_output_user_based(final_result):
    """
    Format output file for user based
    :return: List of dictionaries
    :param final_result: List of tuples
    eg [{'u1': userid1, 'u2': userid2, 'sim': 0.07693}, {'u1': userid1, 'u2': userid2, 'sim': 0.052632}, ...]
    """
    result = []
    for item in final_result:
        result.append({'u1': item[0], 'u2': item[1], 'sim': item[2]})
    return result


if __name__ == '__main__':

    # ========================================== Initializing ==========================================
    time1 = time.time()
    conf = SparkConf()
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "4g")
    conf.set("spark.master", "local[*]")  # Change to local[*] on vocareum
    conf.set("spark.app.name", "task3")
    conf.set("spark.driver.maxResultSize", "4g")
    sc = SparkContext.getOrCreate(conf)
    sc.setLogLevel("ERROR")
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    cf_type = sys.argv[3]
    business_avg_file = "business_avg.json"
    user_avg_file = "user_avg.json"

    # ============================ Read file and Initialize ==========================
    lines = sc.textFile(input_file).distinct()
    reviews, businesses_inv, users_inv, businesses_dict, users_dict = initialize()
    totaltime = time.time() - time1
    print("Duration Initialize: " + str(totaltime))

    # ============================ Item/business based ==========================
    if cf_type == "item_based":
        candidate_pairs, business_reviews_tuple = item_based()
        final_pairs_pre = candidate_pairs.map(lambda x: (x[0], x[1], pearson_helper(data=business_reviews_tuple, a=x[0], b=x[1])))\
            .filter(lambda x: x[2] > 0)

        # Get the business ID back
        final_result = final_pairs_pre.map(lambda x: (businesses_inv[x[0]], businesses_inv[x[1]], x[2])).collect()
        print("Number of pairs final: " + str(len(final_result)))

        totaltime = time.time() - time1
        print("Duration Item Based: " + str(totaltime))

    # =================================== User based =====================================
    else:
        business_reviews = user_based()

        # Doing the Minhashing (As in task 1)
        result_minhash = user_based_minhash()
        totaltime = time.time() - time1
        print("Duration minhash: " + str(totaltime))

        # Doing LSH
        result_lsh = user_based_lsh()
        totaltime = time.time() - time1
        print("Duration LSH: " + str(totaltime))

        # Filter less than 3 corated
        result_lsh_rdd = sc.parallelize(list(result_lsh))
        candidate_pairs, user_reviews_tuple = user_based_after()

        # Do Jaccard and Pearson
        candidate_pairs2 = candidate_pairs.map(lambda x: jaccard(x)) \
            .filter(lambda x: x[2] >= 0.01)
        final_pairs_pre = candidate_pairs2.map(
            lambda x: (x[0], x[1], pearson_helper(data=user_reviews_tuple, a=x[0], b=x[1]))) \
            .filter(lambda x: x[2] > 0)

        # Get the user ID back
        final_result = final_pairs_pre.map(lambda x: (users_inv[x[0]], users_inv[x[1]], x[2])).collect()
        print("Number of pairs final: " + str(len(final_result)))

        totaltime = time.time() - time1
        print("Duration User Based: " + str(totaltime))

    # ======================================= Write results =======================================
    if cf_type == "item_based":
        final_result_write = format_output(final_result)
    else:
        final_result_write = format_output_user_based(final_result)

    with open(output_file, "w") as file:
        for line in final_result_write:
            file.write(json.dumps(line))
            file.write("\n")

    # ========================================== Ending ==========================================
    totaltime = time.time() - time1
    print("Duration: " + str(totaltime))
    sc.stop()