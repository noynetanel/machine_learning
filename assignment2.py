#################################
# Your name: Noy Netanel
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        X = np.random.uniform(0, 1, m)
        X.sort()
        Y = np.array([np.random.choice([0, 1], p=self.conditional_probability(x)) for x in X])
        return np.column_stack((X, Y))


    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        avg_emperical_errors = []
        avg_true_errors = []
        for n in range(m_first, m_last + 1, step):
            emperical_error = []
            true_error = []
            for t in range(T):
                sample = self.sample_from_D(n)
                best_intervals, count_error = intervals.find_best_interval(sample[:, 0], sample[:, 1], k)
                emperical_error.append(count_error / n)
                true_error.append(self.calc_true_error(best_intervals))
            avg_emperical_errors.append(np.average(emperical_error))
            avg_true_errors.append(np.average(true_error))
        n = np.arange(m_first, m_last+1, step)
        plt.plot(n, avg_emperical_errors, label='emperical error', color='red')
        plt.plot(n, avg_true_errors, label='true error', color='blue')
        plt.xlabel("samples")
        plt.legend()
        plt.show()


    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        emperical_errors = []
        true_error = []
        sample = self.sample_from_D(m)
        for k in range(k_first, k_last + 1, step):
            best_intervals, count_error = intervals.find_best_interval(sample[:, 0], sample[:, 1], k)
            emperical_errors.append(count_error / m)
            true_error.append(self.calc_true_error(best_intervals))
        k = np.arange(k_first, k_last + 1, step)
        plt.plot(k, emperical_errors, label='emperical error', color='red')
        plt.plot(k, true_error, label='true error', color='blue')
        plt.xlabel("k")
        plt.legend()
        plt.show()

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        emperical_errors = []
        true_error = []
        penalty = []
        sample = self.sample_from_D(m)
        for k in range(k_first, k_last + 1, step):
            best_intervals, count_error = intervals.find_best_interval(sample[:, 0], sample[:, 1], k)
            emperical_errors.append(count_error / m)
            true_error.append(self.calc_true_error(best_intervals))
            penalty.append(2 * np.sqrt((2 * k + np.log(2 / (0.1/pow(k, 2)))) / m))
        penalty = np.array(penalty)
        k = np.arange(k_first, k_last + 1, step)
        plt.plot(k, emperical_errors, label='emperical error', color='red')
        plt.plot(k, true_error, label='true error', color='blue')
        plt.plot(k, penalty, label='penalty', color='green')
        emperical_errors = np.array(emperical_errors)
        plt.plot(k, penalty+emperical_errors, label='penalty + empirical error', color='purple')
        plt.xlabel("k")
        plt.legend()
        plt.show()


    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        sample = self.sample_from_D(m)
        np.random.shuffle(sample)
        S1, S2 = sample[:int(m*0.8)], sample[int(m*0.8):]  # S2 will be 20% for the holdout validation
        S1 = S1[S1[:, 0].argsort()]
        best_hypo = [intervals.find_best_interval(S1[:, 0], S1[:, 1], k)[0] for k in range(1, 11)]
        S2_empirical_error = np.array([self.calc_emperical_error(S2, inter) for inter in best_hypo])
        return np.argmin(S2_empirical_error) + 1  # The best k value



    #################################
    # Place for additional methods
    def conditional_probability(self, X):
        """
        Input: X: array of x values
        returns: P(y=0|x), P(y=1|x) according to the information in section a
        """
        if (0.2 < X < 0.4) or (0.6 < X < 0.8):
            return [0.9, 0.1]  # [P(y=0|x), P(y=1|x)]
        else:  # x in [0,0.2] or [0.4,0.6] or [0.8,1]
            return [0.2, 0.8]  # [P(y=0|x), P(y=1|x)]

    def h_I(self, x, intervals):
        """
        hI as declared in the beginning of the question
        """
        for i in intervals:
            if i[0] <= x <= i[1]:  # x is in the interval
                return 1
        else:
            return 0

    def intersection_len(self, list1, list2):
        """
        returns the intersection length between the intervals in list1 and list2.
        """
        inter_len = 0
        list1_index = 0
        list2_index = 0
        while list1_index < len(list1) and list2_index < len(list2):
            right = min(list1[list1_index][1], list2[list2_index][1])
            left = max(list1[list1_index][0], list2[list2_index][0])
            if right > left:
                inter_len += (right - left)
            if list1[list1_index][1] == list2[list2_index][1]:
                list1_index += 1  # update both
                list2_index += 1
            elif list1[list1_index][1] < list2[list2_index][1]:
                list1_index += 1    # update only the first list index
            else:
                list2_index += 1    # update only the second list index
        return inter_len

    # section c and d functions:

    def calc_true_error(self, intervals):
        label1 = [(0, 0.2), (0.4, 0.6), (0.8, 1)]
        label0 = [(0.2, 0.4), (0.6, 0.8)]
        inside_label1_inter = self.intersection_len(label1, intervals)
        inside_label0_inter = self.intersection_len(label0, intervals)
        error_labeling1 = 0.8 * (0.6 - inside_label1_inter) + 0.2 * inside_label1_inter
        error_labeling0 = 0.1 * (0.4 - inside_label0_inter) + 0.9 * inside_label0_inter
        return error_labeling1 + error_labeling0

    # section e functions:

    def inside_interval(self, intervals_list, val):
        """
        check if the input val is inside the interval, return 1 if it is.
        """
        for interval in intervals_list:
            if interval[0] <= val <= interval[1]:
                return 1
        return 0

    def zero_one_loss(self, intervals_list, x, y):
        """
        zero one loss as seen in class
        """
        x_inside = self.inside_interval(intervals_list, x)
        if (y == 0 and x_inside == 0) or (y == 1 and x_inside == 1):
            return 0    # 0 if they are equal
        return 1

    def calc_emperical_error(self, sample, intervals_list):
        return sum([self.zero_one_loss(intervals_list, i, j) for i, j in sample]) / len(sample)

    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)

