# Original implementation from Katrin Erk http://www.katrinerk.com/courses/python-worksheets/demo-the-forward-backward-algorithm

import random
import numpy
import matplotlib.pyplot as plt

###########################
## generating the data
# generate 2/3 n from hot, then 1/3 n from cold.
# returns: observations, #hot, #cold
def generate_observations(n):
    # probabilities of ice cream amounts given hot / cold:
    # shown here as amounts of 1's, 2's, 3's out of 10 days
    hot = [ 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    cold = [ 1, 1, 1, 1, 1, 2, 2, 2, 2, 3]

    observations  = [ ]
    # choose 2n observations from "hot"
    numhot = int(2.0/3 * n)
    for i in range(numhot): observations.append(random.choice(hot))
    # choose n observations from "cold"
    numcold = n - numhot
    for i in range(numcold): observations.append(random.choice(cold))

    return (observations, numhot, numcold)

###########################
# data structures:
#
# numstates: number of states. we omit start and end state here, and assume equal probability of starting in either state.
#
# emission probabilities: numpy matrix of shape (N, |V|) (where |V| is the size of the vocabulary)
# where entry (j, o) has emis_j(o).
#
# transition probablities: numpy matrix of shape (N, N) where entry (i, j) has trans(i, j).
#
# forward and backward probabilities: numpy matrix of shape (N, T) where entry (s, t) has forward probability
# for state s and observation t

################
# computing forward and backward probabilities
# forward:
# alpha_t(j) = P(o1, ..., ot, qt = j | HMM)
# forward function: returns numpy matrix of size (N, T)
def forwardprobs(observations, initialprob, trans, emis, numstates, observation_indices):
    forwardmatrix = numpy.zeros((numstates, len(observations)))

    # initialization
    obs_index = observation_indices[ observations[0]]
    for s in range(numstates):
        forwardmatrix[ s, 0 ] = initialprob[s] * emis[ s, obs_index]

    # recursion step
    for t in range(1, len(observations)):
        obs_index = observation_indices[ observations[t]]
        for s in range(numstates):
            forwardmatrix[s, t] = emis[s, obs_index] * sum([forwardmatrix[s2, t-1] * trans[s2, s] \
                                       for s2 in range(numstates)])
    return forwardmatrix

# beta_t(j) = P(o_{t+1}, ..., o_T | qt = j, HMM)
# backward function: returns numpy matrix of size (N, T)
def backwardprobs(observations, trans, emis, numstates, observation_indices):
    backwardmatrix = numpy.zeros((numstates, len(observations)))

    # initialization
    for s in range(numstates):
        backwardmatrix[ s, len(observations) - 1 ] = 1.0

    # recursion
    for t in range(len(observations) - 2, -1, -1):
        obs_index = observation_indices[ observations[t+1]]
        for s in range(numstates):
            backwardmatrix[s, t] = sum([ trans[s, s2] * emis[s2, obs_index] * backwardmatrix[s2, t+1] \
                                         for s2 in range(numstates) ])

    return backwardmatrix
                                

def test_alphabeta():
    observations = [3,1,3]
    trans = numpy.matrix("0.7 0.3; 0.4 0.6")
    emis = numpy.matrix("0.2 0.4 0.4; 0.5 0.4 0.1")
    initialprob = numpy.array([0.8, 0.2])
    numstates = 2
    obs_indices = { 1 : 0, 2 : 1, 3: 2}

    print("FORWARD")
    print(forwardprobs(observations, initialprob, trans, emis, numstates, obs_indices))
    print("\n")
   
    print('BACKWARD')
    print(backwardprobs(observations, trans, emis, numstates, obs_indices))
    print("\n")

   
####
# expectation step:
# re-estimate xi_t(i, j) and gamma_t(j)
# returns two things:
# - gamma is a (N, T) numpy matrix
# - xi is a list of T numpy matrices of size (N, N)
def expectation(observations, trans, emis, numstates, observation_indices, forward, backward):
    # denominator: P(O | HMM)
    p_o_given_hmm = sum([forward[s_i, len(observations) -1] for s_i in range(numstates) ])
   
    # computing xi
    xi = [ ]
    for t in range(len(observations) - 1):
        obs_index = observation_indices[observations[t+1]]
       
        xi_t = numpy.zeros((numstates, numstates))
       
        for s_i in range(numstates):
            for s_j in range(numstates):
                xi_t[ s_i, s_j] = (forward[s_i, t] * trans[s_i, s_j] * emis[s_j, obs_index] * backward[s_j, t+1]) / p_o_given_hmm
        xi.append(xi_t)

    # computing gamma
    gamma = numpy.zeros((numstates + 2, len(observations)))
    for t in range(len(observations) - 1):
        for s_i in range(numstates):
            gamma[s_i, t] = sum([ xi[t][s_i, s_j] for s_j in range(numstates) ])

    for s_j in range(numstates):
        gamma[s_j, len(observations) - 1] = sum( [ xi[t][s_i, s_j] for s_i in range(numstates) ] )
           
    return (gamma, xi)

###
# maximization step:
# re-estimate trans, emis based on gamma, xi
# returns:
# - initialprob
# - trans
# - emis
def maximization(observations, gamma, xi, numstates, observation_indices, vocabsize):
    # re-estimate initial probabilities
    initialprob = numpy.array([gamma[s_i, 0] for s_i in range(numstates)])
   
    # re-estimate emission probabilities
    emis = numpy.zeros((numstates, vocabsize))

    for s in range(numstates):
        denominator = sum( [gamma[s, t] for t in range(len(observations))])
        for vocab_item, obs_index in observation_indices.items():
            emis[s, obs_index] = sum( [gamma[s, t] for t in range(len(observations)) if observations[t] == vocab_item] )/denominator

    # re-estimate transition probabilities
    trans = numpy.zeros((numstates, numstates))

    for s_i in range(numstates):
        denominator = sum( [gamma[s_i, t] for t in range(len(observations) - 1) ])
       
        for s_j in range(numstates):
            trans[s_i, s_j] = sum( [ xi[t][s_i, s_j] for t in range(len(observations) - 1) ] )/denominator


    return (initialprob, trans, emis)

##########
# testing forward/backward
def test_forwardbackward(numobservations, numiter):
    ########
    # generate observation
    observations, truenumhot, truenumcold = generate_observations(numobservations)
    obs_indices = { 1 : 0, 2 : 1, 3: 2}
    numstates = 2
    vocabsize = 3

    #####
    # HMM initialization
   
    # initialize initial probs
    unnormalized = numpy.random.rand(numstates)
    initialprob = unnormalized / sum(unnormalized)
   
    # initialize emission probs
    emis = numpy.zeros((numstates, vocabsize))
    for s in range(numstates):
        unnormalized = numpy.random.rand(vocabsize)
        emis[s] = unnormalized / sum(unnormalized)
   
    # initialize transition probs
    trans = numpy.zeros((numstates, numstates))
    for s in range(numstates):
        unnormalized = numpy.random.rand(numstates)
        trans[s] = unnormalized / sum(unnormalized)

    print("OBSERVATIONS:")
    print(observations)
    print("\n")
   
    print("Random initialization:")
    print("INITIALPROB")
    print(initialprob)
    print("\n")

    print("EMIS")
    print(emis)
    print("\n")

    print("TRANS")
    print(trans)
    print("\n")

    input()
   
    for iteration in range(numiter):
   
        forward = forwardprobs(observations, initialprob, trans, emis, numstates, obs_indices)
        backward = backwardprobs(observations, trans, emis, numstates, obs_indices)

        gamma, xi = expectation(observations, trans, emis, numstates, obs_indices, forward, backward)

        initialprob, trans, emis = maximization(observations, gamma, xi, numstates, obs_indices, vocabsize)

        print("Re-computed:")
        print("INITIALPROB")
        print(initialprob)
        print("\n")

        print("EMIS")
        print(emis)
        print("\n")

        print("TRANS")
        print(trans)
        print("\n")
   
        print("GAMMA(1)")
        print(gamma[0])
        print("\n")
       
        print("GAMMA(2)")
        print(gamma[1])
        print("\n")
       

        # the first truenumhot observations were generated from the "hot" state.
        # what is the probability of being in state 1 for the first
        # truenumhot observations as opposed to the rest
        avgprob_state1_for_truehot = sum(gamma[0][:truenumhot]) / truenumhot
        avgprob_state1_for_truecold = sum(gamma[0][truenumhot:]) / truenumcold
        print("Average prob. of being in state 1 when true state was Hot:", avgprob_state1_for_truehot)
        print("Average prob. of being in state 1 when true state was Cold:", avgprob_state1_for_truecold)

        # plot observations and probabilities of being in certain states
        from matplotlib import interactive
        interactive(True)
        xpoints = numpy.arange(len(observations))
        fig, ax1 = plt.subplots()
        ax1.plot(xpoints, observations, "b-")
        plt.ylim([0, 4])
        ax1.set_xlabel("timepoints")
        ax1.set_ylabel("observations", color = "b")
   
        ax2 = ax1.twinx()
        ax2.plot(xpoints, gamma[0], "r-")
        plt.ylim([0.0, 1.0])
        ax2.set_ylabel("prob", color = "r")
        plt.show()
        input()
        plt.close()