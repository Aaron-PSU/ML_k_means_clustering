# Aaron Dawson
# PSUID: 900818217
# 3/3/20

import numpy as np
THRESHOLD = 5

class Cluster(object):
    def __init__(self, training_data_received, num_centers):
        self.set_s = training_data_received  # set S = x1, x2, x3 .....x(n)
        self.num_centers = num_centers
        self.centers = self.pick_random_centers(training_data_received, num_centers)
        self.set_s_membership = self.cluster_membership(self.k_means_cluster())
        self.digits_in_cluster_count = np.zeros([self.num_centers, 10])
        self.center_labels = np.zeros(len(self.centers))

    def print_centers(self):
        print(self.centers)

    def pick_random_centers(self, set_s, num_of_centers):
        chosen_centers = np.ones(shape=[num_of_centers, 64])
        for i in range(num_of_centers):
            chosen_centers[i] = set_s[np.random.randint(low=0, high=len(set_s), dtype=int), 0:-1]

        # check if duplicates just in case
        if len(np.unique(chosen_centers, axis=0)) != num_of_centers:
            print("pick centers failed: trying again...")
            return self.pick_random_centers(set_s, num_of_centers)  # try again
        else:
            return chosen_centers

    # receives a single element x from set_s
    # returns matrix with distance values (squared) from each cluster center (centroid)
    def k_means(self, element_x):
        running_sum = 0
        distance_matrix = np.ones(len(self.centers))
        for i in range(self.num_centers):
            for j in range(63):
                running_sum += np.square(element_x[j] - self.centers[i, j])
            distance_matrix[i] = running_sum
            running_sum = 0
        return distance_matrix

    # returns matrix of all distances (squared) to all centers: shape (len(set_s), num_centers)
    def k_means_cluster(self):
        cluster_matrix = np.ones([len(self.set_s), self.num_centers])
        for i in range(len(self.set_s)):
            cluster_matrix[i] = self.k_means(self.set_s[i])
        return cluster_matrix

    # receives cluster matrix with Euclidean distances of every element in set S to each cluster center
    # returns array with the membership of each element in set S.
    #         array shape: (len(set_s)) where value is cluster_number_it_is_a_member_of)
    # for example: if index [4] had a value of 5, that indicates that the corresponding index [4] from
    # self.set_s is a member of cluster 5
    def cluster_membership(self, cluster_matrix):
        memberships_array = np.ones(len(self.set_s))
        for i in range(len(cluster_matrix)):
            memberships_array[i] = int(np.argmin(cluster_matrix[i]))
        return memberships_array

    def train(self):
        prev_centers = np.zeros(shape=self.centers.shape)
        while centers_difference(prev_centers, self.centers) > THRESHOLD:
            prev_centers = self.centers
            self.update_centers(self.cluster_membership(self.k_means_cluster()))
            print("difference: {0}".format(centers_difference(prev_centers, self.centers)))
        self.update_center_labels()

    # using membership data that holds which cluster each element in set S is a member of
    # calculates then sets new centers
    def update_centers(self, memberships):
        all_sums = np.zeros([self.num_centers, 64])
        elements_in_cluster = np.zeros(self.num_centers)

        # add within each cluster
        for i in range(len(memberships)):
            all_sums[int(memberships[i])] += self.set_s[i, 0:-1]
            elements_in_cluster[int(memberships[i])] += 1

        # divide by number of elements in each cluster
        for i in range(self.num_centers):
            all_sums[i] /= elements_in_cluster[i]

        # set the new centers
        self.centers = all_sums

    # after training, need to figure out which clusters correspond to which digits then set self.center_labels
    # example: if self.center_labels[0]'s value is 3, that means that center[0] corresponds to the digit 3
    def update_center_labels(self):
        # update membership information on set_s
        self.set_s_membership = self.cluster_membership(self.k_means_cluster())

        # create container for holding how many times each digit is found in each cluster
        # then fill using all elements in S
        for i in range(len(self.set_s)):
            membership_of_current = int(self.set_s_membership[i])
            answer_of_current = int(self.set_s[i, 64])
            self.digits_in_cluster_count[membership_of_current, answer_of_current] += 1

        for i in range(len(self.centers)):
            self.center_labels[i] = np.argmax(self.digits_in_cluster_count[i])

    # using centers found from training, calculate closest center and use that center's label as guess
    # returns digit guess as int
    def make_guess(self, element):
        distances = self.k_means(element)
        closest_center = np.argmin(distances)
        return int(self.center_labels[closest_center])

    # simple "how many times did it guess right" over the total number of guesses, calculation
    # test data received should follow same format as set_s
    # returns accuracy as a float
    def accuracy(self, test_data):
        correct_count = 0
        for i in range(len(test_data)):
            guess = self.make_guess(test_data[i])
            if guess == int(test_data[i, 64]):
                correct_count += 1
        return correct_count / len(test_data)

    # mean square error calculation
    # should only be called after training
    def average_mse(self):
        all_distances = self.k_means_cluster()
        all_mse_sums = np.zeros(self.num_centers)
        for i in range(len(self.set_s_membership)):
            all_mse_sums[int(self.set_s_membership[i])] += all_distances[i, int(self.set_s_membership[i])]
        elements_in_each_cluster = np.sum(self.digits_in_cluster_count, axis=1)
        mse_of_each_cluster = all_mse_sums / elements_in_each_cluster
        answer = np.sum(mse_of_each_cluster)
        answer /= self.num_centers
        return answer

    # mean square separation calculation
    def mss(self):
        print("hi")

    def mean_entropy(self):
        print("hello")


def centers_difference(prev, current):
    dif_array = np.square(prev - current)
    dif = np.sum(dif_array)
    return dif


# load the .train and .test files into a numpy array
def load_data(file):
    arrays = np.loadtxt(file, delimiter=',')
    return arrays


if __name__ == '__main__':
    training_data = load_data("optdigits.train")
    testing_data = load_data("optdigits.test")

    my_cluster = Cluster(training_data, 10)

    my_cluster.print_centers()
    my_cluster.train()
    my_cluster.print_centers()

    print(my_cluster.accuracy(testing_data))
    print(my_cluster.average_mse())
    # my_cluster.test_accuracy(testing_data)
