# minhash_utils.py
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

def linear_mod_hash(value, scale=251, shift=739, prime=30011):
  
    #applies an hash function in a limitated range
    value = np.asarray(value) 
    return (scale*value +shift)%prime


def bitwise_shift_hash(value, scale=251, shift=1531, prime=30011):
    #calculate an hash bitwise combining XOR
    x = np.asarray(value)  # ensure input is a NumPy array
    return (np.bitwise_xor(x,scale)+shift)%prime

def random_signature(user_ids, user_items_map, num_hashes=10, prime=30011, seed=42):
      
      
      '''
    Generates signature vectors of specified length for each user, using randomized hash functions.
    Parameters:
    - user_ids (ndarray of integers): array of user IDs to create signature vectors for.
    - user_items_map (dict): mapping of users to their associated items (userId: [item1, item2, ...]).
    - prime (int): a prime number greater than the maximum possible item ID.
    - num_hashes (int): number of hash functions to generate the signature vectors.
    - seed (int): seed value to ensure reproducibility of the hash functions.
    Returns:
    - signature_matrix: a matrix containing the generated signature vectors.
    - chosen_hashes: a list of the selected hash functions and their parameters.
     '''
      rng = np.random.RandomState(seed)  #initialize the casual generator
      random_choices = rng.randint(0, 2, num_hashes)  #generate random 0 and 1 
      signature_matrix = np.zeros((num_hashes, user_ids.shape[0]), dtype=int)  #initialize the signatures matrix

      chosen_hashes = [] 

      for i in range(num_hashes):
        # Case 1: use the linear hash
        if random_choices[i] == 0:
            a, b = rng.randint(1, prime), rng.randint(0, prime)
            chosen_hashes.append((0, a, b))  # salva tipo e parametri dell'hash
            for idx, user in enumerate(user_ids):
                hashed_values = linear_hash(user_items_map[user], a, b, prime)
                signature_matrix[i, idx] = np.min(hashed_values)

        # Case 2: use the hash bitwise
        else:
            a, b = rng.randint(1, prime), rng.randint(0, prime)
            chosen_hashes.append((1, a, b))  # salva tipo e parametri dell'hash
            for idx, user in enumerate(user_ids):
                hashed_values = bitwise_hash(user_items_map[user], a, b, prime)
                signature_matrix[i, idx] = np.min(hashed_values)

      return signature_matrix, chosen_hashes

def minhash(user_ids, user_items_map, hash_choices, linear_params=None, bitwise_params=None, num_hashes=10, prime=30011):
    '''
    Generates a signature matrix using predefined hash functions.
    Parameters:
    - user_ids (ndarray of integers): user IDs to create signature vectors for.
    - user_items_map (dict): mapping of user IDs to their associated items (userId: [item1, item2, ...]).
    - hash_choices (list): list of 0s and 1s indicating which hash function to use (0 for linear, 1 for bitwise).
    - linear_params (dict): parameters for linear hash functions.
    - bitwise_params (dict): parameters for bitwise hash functions.
    - num_hashes (int): number of hash functions (length of the signature vector).
    - prime (int): large prime number for modulo operations.
    Returns:
    - signature_matrix: matrix of signature vectors.
    '''
    signature_matrix = np.full((num_hashes, len(user_ids)), np.inf)  # Initialize with large values

    for i in range(num_hashes):
        # Use linear hash
        if hash_choices[i] == 0:
            a, b = linear_params[i]
            for idx, user in enumerate(user_ids):
                hashed_items = linear_mod_hash(user_items_map[user], a, b, prime)
                signature_matrix[i, idx] = np.min(hashed_items)
        
        # Use bitwise hash
        elif hash_choices[i] == 1:
            a, b = bitwise_params[i]
            for idx, user in enumerate(user_ids):
                hashed_items = bitwise_shift_hash(user_items_map[user], a, b, prime)
                signature_matrix[i, idx] = np.min(hashed_items)

    return signature_matrix

def random_minhash(user_ids, user_items_map, num_hashes=10, prime=30011, seed=42):
    '''
    Generates a signature matrix using random hash functions.
    Parameters:
    - user_ids (ndarray of integers): user IDs to create signature vectors for.
    - user_items_map (dict): mapping of user IDs to their associated items (userId: [item1, item2, ...]).
    - num_hashes (int): number of hash functions (length of the signature vector).
    - prime (int): large prime number for modulo operations.
    - seed (int): seed for reproducibility of hash function selection.
    Returns:
    - signature_matrix: matrix of signature vectors.
    - hash_functions: list of tuples (type, a, b) where type is 0 (linear) or 1 (bitwise).
    '''
    rng = np.random.default_rng(seed)  # Initialize random number generator
    hash_choices = rng.integers(0, 2, size=num_hashes)  # Randomly choose hash function types
    signature_matrix = np.full((num_hashes, len(user_ids)), np.inf)  # Initialize with large values
    hash_functions = []

    for i in range(num_hashes):
        a, b = rng.integers(1, prime, size=2)
        hash_functions.append((hash_choices[i], a, b))

        if hash_choices[i] == 0:  # Linear hash
            for idx, user in enumerate(user_ids):
                hashed_items = linear_mod_hash(user_items_map[user], a, b, prime)
                signature_matrix[i, idx] = np.min(hashed_items)
        else:  #bitwise hash
            for idx, user in enumerate(user_ids):
                hashed_items = bitwise_shift_hash(user_items_map[user], a, b, prime)
                signature_matrix[i, idx] = np.min(hashed_items)

    return signature_matrix, hash_functions
def minhash_results(user_ids, user_movies_dict, k_hash_function_choices, linear_parameters=None, bitwise_parameters=None, num_hashes=10, p=27281):
    '''
    Function that computes the signature matrix of user IDs and compares the signature vectors with 
    the similarity between the original watched movie sets associated with the user IDs
    Inputs:
    - user_ids (ndarray of integers): categories we want to create the signature vectors for
    - user_movies_dict (dict): dictionary made of items of the structure userId: [movieId1, movieId2, ...]
    - k_hash_function_choices (list): list of 0s and 1s representing the hash function chosen (0 for linear_hash, 1 for bitwise_hash)
    - linear_parameters (dict): None by default, else a dictionary that for key 'i', if it exists, contains the function parameters for linear_hash
    - bitwise_parameters (dict): None by default, else a dictionary that for key 'i', if it exists, contains the function parameters for bitwise_hash
    - p (int): large prime number (larger than max movieId)
    - k (int): length of the signature vectors
    Outputs:
    - error (int): error rate of this configuration
    - hash_functions (list): list of tuples (i,a,b) where 'i' identifies the chosen hash function, and 'a', 'b' are its parameters
    '''
    signature_matrix = minhash(user_ids, user_movies_dict, k_hash_function_choices, linear_parameters, bitwise_parameters, num_hashes = 10)
    
    N = 1000 # number of users to compare
    # List of all user columns
    all_user_columns = list(range(signature_matrix.shape[1]))
    # Sampled user columns
    sampled_user_columns = random.sample(all_user_columns, N)
    # Split the sampled users into two groups
    first_group = sampled_user_columns[:(N//2)]
    second_group = sampled_user_columns[(N//2):]

    errors = [] # Initialize list to store errors, the abs differences between P(s1[i]=s2[i]) and Jaccard(user1, user2)
    prob_values = [] # Initialize list to store probability values P(s1[i]=s2[i]) as defined above
    jaccard_values = [] # Initialize list to store Jaccard similarity values between pairs of users

    # Pair-wise compare jaccard similarity of the users with their
    # probability of having corresponding signature vector elements
    for i in range(N//2):

        # Signature vectors of the current pair of users
        signature1 = signature_matrix[:,first_group[i]]
        signature2 = signature_matrix[:,second_group[i]]
        # Compare signature vectors
        prob_same_el = sum(signature1==signature2)/num_hashes
        prob_values.append(prob_same_el) # add probability to prob_values list

        # User ID of the current user
        user1 = user_ids[first_group[i]]
        user2 = user_ids[second_group[i]]
        # Sets of watched movies of the users (actually the virtual movie rows, but the result is the same)
        watched_movies1 = set(user_movies_dict[user1])
        watched_movies2 = set(user_movies_dict[user2])
        # Jaccard similarity of the users
        jaccard_sim = calculate_jaccard(watched_movies1, watched_movies2)
        jaccard_values.append(jaccard_sim) # add jaccard similarity to the list

        # Calculate error
        errors.append(abs(prob_same_el - jaccard_sim))

    return np.mean(errors)
def random_minhash_results(user_ids, user_movies_dict, k = 10, p=27281, seed=42):
    '''
    Function that computes the signature matrix of user IDs and compares the signature vectors with 
    the similarity between the original watched movie sets associated with the user IDs
    Inputs:
    - user_ids (ndarray of integers): categories we want to create the signature vectors for
    - user_movies_dict (dict): dictionary made of items of the structure userId: [movieId1, movieId2, ...]
    - k_hash_function_choices (list): list of 0s and 1s representing the hash function chosen (0 for linear_hash, 1 for bitwise_hash)
    - linear_parameters (dict): None by default, else a dictionary that for key 'i', if it exists, contains the function parameters for linear_hash
    - bitwise_parameters (dict): None by default, else a dictionary that for key 'i', if it exists, contains the function parameters for bitwise_hash
    - p (int): large prime number (larger than max movieId)
    - k (int): length of the signature vectors
    Outputs:
    - error (int): error rate of this configuration
    - hash_functions (list): list of tuples (i,a,b) where 'i' identifies the chosen hash function, and 'a', 'b' are its parameters
    '''
    signature_matrix, hash_functions = random_minhash(user_ids, user_movies_dict, k, seed=seed)
    N = 1000 # number of users to compare
    # List of all user columns
    all_user_columns = list(range(signature_matrix.shape[1]))
    # Sampled user columns
    sampled_user_columns = random.sample(all_user_columns, N)
    # Split the sampled users into two groups
    first_group = sampled_user_columns[:(N//2)]
    second_group = sampled_user_columns[(N//2):]

    errors = [] # Initialize list to store errors, the abs differences between P(s1[i]=s2[i]) and Jaccard(user1, user2)
    prob_values = [] # Initialize list to store probability values P(s1[i]=s2[i]) as defined above
    jaccard_values = [] # Initialize list to store Jaccard similarity values between pairs of users

    # Pair-wise compare jaccard similarity of the users with their
    # probability of having corresponding signature vector elements
    for i in range(N//2):

        # Signature vectors of the current pair of users
        signature1 = signature_matrix[:,first_group[i]]
        signature2 = signature_matrix[:,second_group[i]]
        # Compare signature vectors
        prob_same_el = sum(signature1==signature2)/k
        prob_values.append(prob_same_el) # add probability to prob_values list

        # User ID of the current user
        user1 = user_ids[first_group[i]]
        user2 = user_ids[second_group[i]]
        # Sets of watched movies of the users (actually the virtual movie rows, but the result is the same)
        watched_movies1 = set(user_movies_dict[user1])
        watched_movies2 = set(user_movies_dict[user2])
        # Jaccard similarity of the users
        jaccard_sim = calculate_jaccard(watched_movies1, watched_movies2)
        jaccard_values.append(jaccard_sim) # add jaccard similarity to the list

        # Calculate error
        errors.append(abs(prob_same_el - jaccard_sim))

    return np.mean(errors), hash_functions
def linear_dot_prod_hash(values, scale_factors, shift, prime):
    '''
    Hashes an array of integers using linear dot product hashing.
    Parameters:
    - values (ndarray): array of integers to hash.
    - scale_factors (ndarray): scaling factors for the dot product.
    - shift (int): additive parameter.
    - prime (int): large prime number for modulo operations.
    Returns:
    - int or ndarray: hashed value(s) constrained to the range [1, prime].
    '''
    values = np.asarray(values, dtype=np.int64)  # Ensure input is 64-bit
    scale_factors = np.asarray(scale_factors, dtype=np.int64)
    return (np.dot(values, scale_factors) + shift) % prime
def LSH(signature_matrix, user_ids, rows_per_band=4, prime=65537, seed=4294967295):
    '''
    Performs Locality-Sensitive Hashing (LSH) on a signature matrix.
    Parameters:
    - signature_matrix (ndarray): matrix of signature vectors.
    - user_ids (ndarray of integers): user IDs corresponding to the signature matrix columns.
    - rows_per_band (int): number of rows per band.
    - prime (int): prime number for modulo operations.
    - seed (int): seed for reproducibility.
    Returns:
    - buckets (defaultdict): mapping of buckets to users hashed to them.
    - candidates (defaultdict): mapping of users to all users sharing buckets.
    '''
    rng = np.random.default_rng(seed)  # Random number generator
    num_bands = signature_matrix.shape[0] // rows_per_band

    a = rng.integers(1, prime, size=(rows_per_band,))
    b = rng.integers(0, prime)

    buckets = defaultdict(list)
    candidates = defaultdict(set)

    for band in range(num_bands):
        band_start = band * rows_per_band
        band_end = (band + 1) * rows_per_band
        for idx, user_id in enumerate(user_ids):
            band_vector = signature_matrix[band_start:band_end, idx]
            bucket_id = linear_dot_prod_hash(band_vector, a, b, prime)
            buckets[bucket_id].append(user_id)
            for user in buckets[bucket_id]:
                if user != user_id:
                    candidates[user_id].add(user)

    return buckets, candidates

def LSH_performance(signature_matrix, user_ids, similar_pairs, rows_per_band=4, prime=65537, seed=4294967295):
    '''
    Evaluates LSH performance by comparing the identified candidate pairs with the actual similar pairs.
    Parameters:
    - signature_matrix (ndarray): matrix of signature vectors.
    - user_ids (ndarray of integers): user IDs corresponding to the signature matrix columns.
    - similar_pairs (list of tuples): actual similar user pairs and their similarities.
    - rows_per_band (int): number of rows per band.
    - prime (int): prime number for modulo operations.
    - seed (int): seed for reproducibility.
    Returns:
    - avg_candidate_set_size (float): average number of candidates per user.
    - percentage_similar_candidates (float): percentage of actual similar pairs identified as candidates.
    '''
    buckets, candidates = LSH(signature_matrix, user_ids, rows_per_band, prime, seed)

    # Evaluate candidate set sizes
    avg_candidate_set_size = np.mean([len(candidates[user]) for user in user_ids])

    # Count actual similar pairs identified as candidates
    identified_similar_pairs = 0
    for user1, user2 in similar_pairs:
        if user2 in candidates[user1] and user1 in candidates[user2]:
            identified_similar_pairs += 1

    percentage_similar_candidates = identified_similar_pairs / len(similar_pairs) if similar_pairs else 0

    return avg_candidate_set_size, percentage_similar_candidates





def calculate_jaccard(set_a, set_b):
    """
    Calcola la similarità di Jaccard tra due insiemi.
    """
    if not set_a and not set_b:  # Gestione insiemi vuoti
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)

def similar_users(query_user, candidates, user_ids, user_movies_dict):
 
    user_candidates = candidates[query_user] # get the candidates of the query user
    similarity = lambda user: calculate_jaccard(set(user_movies_dict[query_user]), set(user_movies_dict[user])) # lambda function that computes the jaccard similarity between two users

    # Sort candidates by similarity, and always pick the top two
    sorted_candidates = sorted(user_candidates, key=similarity, reverse=True) # get the user with the highest jaccard similarity
    most_similar = sorted_candidates[0]
    second_most_similar = sorted_candidates[1]
    third_most_similar = sorted_candidates[2]

    # List of the two most similar users
    most_similar_users = [
        (most_similar, calculate_jaccard(set(user_movies_dict[query_user]), set(user_movies_dict[most_similar]))),
        (second_most_similar, calculate_jaccard(set(user_movies_dict[query_user]), set(user_movies_dict[second_most_similar]))),
        (third_most_similar, calculate_jaccard(set(user_movies_dict[query_user]), set(user_movies_dict[third_most_similar])))

    ]

    # create dataframe of the two most similar users with their similarities to the input user
    most_similar_df = pd.DataFrame(most_similar_users, columns = ['User', 'Similarity'])
    return most_similar_df

def get_rated_movies(user, rating):
    '''
    Function that retrieves all the rated movies of a user and the rating they gave it
    Inputs:
    - user (int): user ID
    - rating (DataFrame): DataFrame containing information about userId, movieId, rating and timestamps
    Outputs:
    - rated_movies (list): list of tuples containing the movieId and the rating the user gave it
    '''
    # Restrict the dataframe to the rows where the user ID appears
    rated_movies_df = rating[rating['userId']==user][['movieId', 'rating']]
    # Turn the DataFrame into a list of tuples (movieId, rating)
    rated_movies = list(zip(rated_movies_df['movieId'], rated_movies_df['rating']))
    return rated_movies
def rated_movies_intersection(user1, user2, rating):
    '''
    Function that finds the rated movies that two users have rated in common
    Inputs:
    - user1 (int): user ID of the first user
    - user2 (int): user ID of the second user
    - rating (DataFrame): DataFrame containing information about userId, movieId, rating and timestamps
    Outputs:
    - movie_recs (list): list of tuples containing the movieId and the rating the user gave it
    '''
    # Get movies rated by the two users and the rating
    rated_movies1 = get_rated_movies(user1, rating)
    #print(sorted(rated_movies1, key=lambda x: x[0]))
    rated_movies2 = get_rated_movies(user2, rating)
    #print(sorted(rated_movies2, key=lambda x: x[0]))
    # Just the movies rated by the users
    movies1 = set([movie[0] for movie in rated_movies1])
    movies2 = set([movie[0] for movie in rated_movies2])
    # Get intersection of movies
    movies_intersection = list(movies1.intersection(movies2))
    # Initialize movie recommendations list
    movie_recs = []
    for movie in movies_intersection:
        # Get the rating of the movie (perhaps some users rated the same movie twice)
        rating1 = [mov[1] for mov in rated_movies1 if mov[0]==movie]
        rating2 = [mov[1] for mov in rated_movies2 if mov[0]==movie]
        # Get the average rating
        avg_rating = (np.mean(rating1) + np.mean(rating2))/2
        movie_recs.append((movie, avg_rating))
    return movie_recs
def recommend_movies(query_user, candidates, user_ids, user_movies_dict, rating, movie_df):
    '''
    Function that recommends movies based on a user's two most similar users and their rated movies
    Inputs:
    - query_user (int): user ID of the query user
    - candidates (defaultdict): defaultdict containing the candidates of each user
    - user_ids (ndarray of integers): user IDs
    - user_movies_dict (defaultdict): defaultdict containing the virtual rows of the movies each user has watched
    Outputs:
    - recommended_movies (DataFrame): movie ID(S) and rating(s) of the recommended movie(s)
    '''
    if query_user not in user_ids:
        print('User not found')
        return None

    # Get the two most similar users to the query user
    two_most_similar = similar_users(query_user, candidates, user_ids, user_movies_dict)
    userId1, userId2 = two_most_similar['User'][0], two_most_similar['User'][1]

    # Find the intersection of movies and average ratings
    movies_intersection = rated_movies_intersection(userId1, userId2, rating)

    # Remove from these movies the ones seen by the query user
    movies_intersection = [(movie[0], movie[1]) for movie in movies_intersection if movie[0] not in user_movies_dict[query_user]]

    # If the intersection is not empty, return a DataFrame with the recommended movies
    if movies_intersection:
        # Replace movie Id with the movie name using the movie_df DataFrame
        movies_intersection = [(movie_df[movie_df['movieId']==movie[0]]['title'].values[0], movie[1]) for movie in movies_intersection]
        movies_intersection_df = pd.DataFrame(movies_intersection, columns = ['movie', 'rating'])
        movies_intersection_df.sort_values(by='rating', ascending=False, inplace=True) # Sort the DataFrame by average rating

        # If there are at least five movies in the intersection, return these five
        if len(movies_intersection_df) >= 5:
            return movies_intersection_df.iloc[:5]
               # If the two most similar users have less than five movies in common
        else: # Fill the remaining recommendations with top rated movies from the most similar user
            top_rated = get_rated_movies(userId1, rating)
            top_rated = [(movie[0], movie[1]) for movie in top_rated if movie[0] not in user_movies_dict[query_user]] # Eliminate the movies that the user has seen
            top_rated.sort(key=lambda x: x[1], reverse=True) # Sort the movies by rating
            num_movies_from_most_similar = 5 - len(movies_intersection_df) # Return five movie recommendations, the ones movie_intersection_df plus movies from the most similar user until we get to five movies
            top_rated = top_rated[:num_movies_from_most_similar] # Restrict number of top rated movies from most similar user to num_movies_from_most_similar
            top_rated = [(movie_df[movie_df['movieId']==movie[0]]['title'].values[0], movie[1]) for movie in top_rated] # Replace movie Id with the movie name using the movie_df DataFrame
            movies_intersection_df = pd.concat([movies_intersection_df, pd.DataFrame(top_rated[:num_movies_from_most_similar], columns = ['movie', 'rating'])]) # Concatenate movies in the intersection with movies rated by the most similar user
            movies_intersection_df.sort_values(by='rating', ascending=False, inplace=True)  # Sort movies again because we concatenated two dfs
            return movies_intersection_df

    else:
        top_rated = get_rated_movies(userId1, rating)
        top_rated.sort(key=lambda x: x[1], reverse=True)
        return top_rated[:5]


