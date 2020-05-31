import json
import sys
import csv

sys.path.append('..')
from plsa import Corpus, Pipeline, Visualize
from plsa.pipeline import DEFAULT_PIPELINE
from plsa.algorithms import PLSA

"""
Returns a list of ids for businesses in the provided category.
"""
def extract_businesses_by_category(filepath, category, state):
    ids = set()
    counter = 0

    with open(filepath, 'r') as f:
        for line in f.readlines():
            counter += 1

            business_json = json.loads(line)
            bjc = business_json['categories']
            if category in bjc:
                if len(bjc) > 1 and business_json['state'] == state:
                    ids.add(business_json['business_id'])

    business_count = len(ids)
    print(f'Processed {counter} businesses')
    print(f'Found {business_count} businesses for category {category} in {state}')
    return ids


"""
Collects reviews for the provided business ids. Writes results to a CSV file.
"""
def extract_business_reviews(filepath, business_ids, outputfile, max_docs, stars):
    counter = 0
    reviews = list()

    with open(filepath, 'r') as f:
        for line in f.readlines():
            counter += 1
            if len(reviews) > max_docs:
                break

            review_json = json.loads(line)
            business_id = review_json['business_id']
            if business_id in business_ids:
                text = review_json.get('text')
                star = review_json.get('stars', 0)

                if text and (star in stars):
                    reviews.append((business_id, text, star))

    collected_reviews = len(reviews)
    print(f'Processed {counter} reviews')
    print(f'Writing {collected_reviews} reviews to CSV')

    with open(outputfile, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for review in reviews:
            csvwriter.writerow(review)

    return collected_reviews


"""
Runs PLSA on review corpus to extract topics.
"""
def extract_topics(filepath, column_index, k_topics, max_docs):
    print('Running PLSA')
    pipeline = Pipeline(*DEFAULT_PIPELINE)
    corpus = Corpus.from_csv(filepath, pipeline, column_index, 'latin_1', max_docs)

    plsa = PLSA(corpus, k_topics, True)
    result = plsa.best_of(5)

    print(result)
    return result


"""
Transform PLSA result object into JSON structure suitable for d3.js
"""
def transform_results_to_cluster_json(result, k_topics, outputfile):
    clusters = dict()
    clusters['name'] = 'Topics'
    clusters['children'] = list()

    for topic in range(k_topics):
        name = f'Topic {topic}'
        node = { 'name': name, 'children': [] }

        for name, size in result.word_given_topic[topic][:10]:
            node['children'].append({ 'name': name, 'size': size })

        clusters['children'].append(node)

    with open(outputfile, 'w') as fp:
        json.dump(clusters, fp)


if __name__=='__main__':
    """
    Specify path to yelp dataset business.json
    """
    filepath = '../yelp_dataset/yelp_academic_dataset_business.json'
    business_ids = extract_businesses_by_category(filepath, 'Restaurants', 'AZ')

    """
    Specify path to yelp dataset review.json
    """
    filepath = '../yelp_dataset/yelp_academic_dataset_review.json'
    extract_business_reviews(filepath, business_ids, 'reviews.csv', 1000, [1, 2, 3, 4, 5])

    """
    Configure PLSA parameters here
    """
    k_topics = 4
    result = extract_topics('reviews.csv', 1, k_topics, 1000)
    transform_results_to_cluster_json(result, k_topics, 'topics.json')
