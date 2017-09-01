import os
import os.path
import random
import cv2
import numpy as np

# Constants
input_dir = "IMFDB_FIXED"
num_training_pairs = 4000 # This is the number of same-person pairs and the number of different-people pairs.
num_testing_pairs = 2000 # This is the number of same-person pairs and the number of different-people pairs.

def pickle(num_pairs, outfile_prefix):
    """ 
        This script does not currently create two mutally-exclusive training and testing sets.
        If you use this to create a training and a testing set (by running it twice),
        there might be overlap!
    """


    """
        First, we need to get a list of all the people in IMFDB.
        Then we need to build the same-person pairs.

        To do this, we will rotate through all the people, picking two random images from each
        until we fill our correct number of pairs.
    """

    people = os.listdir(input_dir)

    same_person_pairs_names = []

    p_idx = 0
    for i in range(0, num_pairs):
        images = [img for img in os.listdir(os.path.join(input_dir, people[p_idx])) if img.startswith("mod_")] # Get all the images that start with "mod_" for the current person

        image1 = random.choice(images)
        image2 = random.choice(images)

        same_person_pairs_names.append((people[p_idx], image1, image2)) # Append a tuple like ("Actor Name", "Image 1 Name", "Image 2 Name")

        p_idx += 1

        if p_idx == len(people):
            p_idx = 0

    """
        To pick different-people pairs, we need to do something more complicated.

        A potential problem is that the pairs are not evenly distributed, meaning that
        some people are represented more than others in the different-people pairs.

        To make sure that this doesn't happen, we will rotate through the people like above,
        and then just choose the other person randomly. This is sub-par because there is
        still the problem of some people having more representation than others. But for now,
        this is good enough.

        One way to fix this would be to choose even distributions for each person before we start
        by dividing the number of different-people pairs by the number of people and then again
        by the number of people - 1. That would be how many people can be matched with each person.
        Then we want to shift this amount mod the number of people for each person's matches such
        that all people are matches for other people at approximately the same frequency.
    """

    different_people_pairs_names = []

    p_idx = 0
    for i in range(0, num_pairs):
        person = people[p_idx]
        other_person = random.choice([p for p in people if p != person]) # Choose a random person that is not the current person.

        person_images = [img for img in os.listdir(os.path.join(input_dir, person)) if img.startswith("mod_")] # Get all the images that start with "mod_" for the current person
        other_person_images = [img for img in os.listdir(os.path.join(input_dir, other_person)) if img.startswith("mod_")] # Get all the images that start with "mod_" for the other person

        image1 = random.choice(person_images)
        image2 = random.choice(other_person_images)

        different_people_pairs_names.append((person, image1, other_person, image2)) # Append a tuple like ("Person 1", "Image for Person 1", "Person 2", "Image for person 2")

        p_idx += 1

        if p_idx == len(people):
            p_idx = 0


    """
        Now we must pickle the data into the proper format.

        For each pair, we read the images into 64x64x1 numpy arrays.
        Then we stack them on top of each other (augment in the z direction)
        to get 64x64x2 arrays. Then we put those in one array and put
        1 for same-person or 0 for different-people results in another
        array. Those will then be pickled.
    """

    same_person_pairs_x = []

    for same_person_pair_tuple in same_person_pairs_names:
        person = same_person_pair_tuple[0]
        image1_name = same_person_pair_tuple[1]
        image2_name = same_person_pair_tuple[2]

        image1 = cv2.imread(os.path.join(input_dir, person, image1_name), 0)
        image2 = cv2.imread(os.path.join(input_dir, person, image2_name), 0)

        both_images = np.stack((image1, image2), axis = 2)
        same_person_pairs_x.append(both_images)


    different_people_pairs_x = []

    for different_people_pair_tuple in different_people_pairs_names:
        person1 = different_people_pair_tuple[0]
        image1_name = different_people_pair_tuple[1]

        person2 = different_people_pair_tuple[2]
        image2_name = different_people_pair_tuple[3]

        image1 = cv2.imread(os.path.join(input_dir, person1, image1_name), 0)
        image2 = cv2.imread(os.path.join(input_dir, person2, image2_name), 0)

        both_images = np.stack((image1, image2), axis = 2)
        different_people_pairs_x.append(both_images)


    same_person_pairs_y = np.ones((num_pairs), dtype=np.uint8)
    different_people_pairs_y = np.zeros((num_pairs), dtype=np.uint8)

    """
        Combine the same_person_pairs_x and different_people_pairs_x and do it for the y's as well.
        This way, we have one input array and one output array. Half of the inputs are same-person
        and half are different-people, in the same way that half of the outputs are 1's and half are 0's.
    """

    pairs_x = np.append(np.array(same_person_pairs_x), np.array(different_people_pairs_x), axis = 0)
    pairs_y = np.append(same_person_pairs_y, different_people_pairs_y, axis = 0)

    """
        Finally, pickle these arrays into two files, "pairs_x.pickle" and "pairs_y.pickle".
    """

    print(pairs_x.shape)
    print(pairs_y.shape)

    np.save(outfile_prefix + "_pairs_x", pairs_x)
    np.save(outfile_prefix + "_pairs_y", pairs_y)

pickle(num_training_pairs, "train")
pickle(num_testing_pairs, "test")