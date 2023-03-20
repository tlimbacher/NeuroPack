import numpy as np
from .memristorPulses import memristorPulses as memristorPulses
# This core implements feed forward NNs with LIF neurons. The output neurons of the NN can fire without restriction.
# Synapses are updated according to back propagation SGD, with the derivative of the step function replaced by noise.
# This core can be used for multi-layer cases.

def normalise_weight(net, w):
    PCEIL = 1.0/net.params['PFLOOR']
    PFLOOR = 1.0/net.params['PCEIL']

    val = (net.params['WEIGHTSCALE']*(float(w) - PFLOOR)/(PCEIL - PFLOOR))*3.0

    # Clamp weights in-between 0.0 and 1.0
    if val < 0.0:
        return 0.0
    elif val > 3.0:
        return 3.0
    else:
        return val


def de_normalise_resistance(self, w):
    PCEIL = 1.0 / self.params['PFLOOR']  # conductance ceil
    PFLOOR = 1.0 / self.params['PCEIL']  # conductance floor

    C = (w / 3.0) * (PCEIL - PFLOOR) / self.params['WEIGHTSCALE'] + PFLOOR
    R = 1 / C
    return R


def init(net):
    # make sure all counters are reset
    net.spikeTrain_cnt = 0
    net.errorSteps_cnt = 0
    net.errorStepsForTest_cnt = 0

    for postidx in range(len(net.ConnMat)):
        # For every presynaptic input the neuron receives.
        for preidx in np.where(net.ConnMat[:, postidx, 0] != 0)[0]:
            w, b=net.ConnMat[preidx, postidx, 0:2]
            net.state.weights[preidx, postidx - net.inputNum, 0] = 1.0/net.read(w, b)
            if net.params.get('NORMALISE', False):
                old_weight = net.state.weights[preidx, postidx - net.inputNum, 0]
                new_weight = normalise_weight(net, old_weight)
                net.state.weights[preidx, postidx - net.inputNum, 0] = new_weight


def neurons(net, time, phase='training'):

    rawin = net.rawin # Raw input
    if phase == 'test':
        stimin = net.stiminForTesting[:, time] # Stimulus input for current timestep
    else:
        stimin = net.stimin[:, time] # input stimuli at this time step
    inputStimMask = np.hstack((np.ones(net.inputNum), np.zeros(net.NETSIZE - net.inputNum)))
    outputLabelMask = np.hstack((np.zeros(net.NETSIZE - net.outputNum), np.ones(net.outputNum)))

    inputStim = np.bitwise_and([int(x) for x in inputStimMask], [int(x) for x in stimin])
    outputLabel = np.bitwise_and([int(x) for x in outputLabelMask], [int(x) for x in stimin])

    rawinArray = np.array(rawin)
    wantToFire = len(net.ConnMat)*[0]

    full_stim = np.bitwise_or([int(x) for x in wantToFire], [int(x) for x in inputStim])
    for i in range(net.inputNum, net.NETSIZE-net.outputNum):
        full_stim[i] = rawinArray[i]  # Recursive spikes as new input.
    old_full_stim = full_stim  # For traces.

    leakage = net.params.get('LEAKAGE', 1.0)
    refractory_t = net.params.get('N_REFRACTORY', 5)
    threshold = net.params.get('FIRETH', 0.001)
    if time > 0:
        if phase == 'test':
            # if this isn't the first step copy the accumulators
            # from the previous step onto the new one
            net.state.NeurAccumForTest[time] = net.state.NeurAccumForTest[time-1]  # size: NETSIZE - inputNum
            # reset the accumulators of neurons that have already fired
            net.state.NeurAccumForTest[time] = net.state.NeurAccumForTest[time] * np.where(rawinArray[net.inputNum : ]  == 0, 1, 0)
            # calculate the leakage term
            net.state.NeurAccumForTest[time] *= (1 - leakage)
        else:
            # if this isn't the first step copy the accumulators
            # from the previous step onto the new one
            net.state.NeurAccum[time] = net.state.NeurAccum[time-1]  # size: NETSIZE - inputNum
            net.log('membrane from last time step:', net.state.NeurAccum[time])
            # calculate the leakage term
            net.state.NeurAccum[time] *= (1 - leakage)
            net.log('membrane after adding leakage:', net.state.NeurAccum[time])

    for postidx in range(net.inputNum, net.NETSIZE):
        # For every presynaptic input the neuron receives.
        for preidx in np.where(net.ConnMat[:, postidx, 0] != 0)[0]:
            if phase == 'test':
                # Excitatory case
                if net.ConnMat[preidx, postidx, 2] > 0:
                    # net.log("Excitatory at %d %d" % (preidx, postidx))
                    # Accumulator increases as per standard formula.
                    net.state.NeurAccumForTest[time][postidx - net.inputNum] += \
                        full_stim[preidx] * net.weightsForTest[preidx, postidx - net.inputNum]

                # Inhibitory case
                elif net.ConnMat[preidx, postidx, 2] < 0:
                    # Accumulator decreases as per standard formula.
                    net.state.NeurAccumForTest[time][postidx - net.inputNum] -= \
                        full_stim[preidx]*net.weightsForTest[preidx, postidx - net.inputNum]
            else:
                # Excitatory case
                if net.ConnMat[preidx, postidx, 2] > 0:
                    # net.log("Excitatory at %d %d" % (preidx, postidx))
                    # Accumulator increases as per standard formula.
                    net.state.NeurAccum[time][postidx - net.inputNum] += \
                        full_stim[preidx] * net.state.weights[preidx, postidx - net.inputNum, 0]

                # Inhibitory case
                elif net.ConnMat[preidx, postidx, 2] < 0:
                    # Accumulator decreases as per standard formula.
                    net.state.NeurAccum[time][postidx - net.inputNum] -= \
                        full_stim[preidx]*net.state.weights[preidx, postidx - net.inputNum, 0]

        # Removing threshold value from neurons that fired in last step.
        if postidx < net.NETSIZE - net.outputNum and rawinArray[net.inputNum:][postidx - net.inputNum]:
            net.state.NeurAccum[time][postidx - net.inputNum] -= threshold

        if phase == 'test' and net.state.NeurAccumForTest[time][postidx - net.inputNum] > net.params.get('FIRETH', 0.001):
            wantToFire[postidx] = 1    # update the firehist to feedforward the spike
        elif phase == 'training' and net.state.NeurAccum[time][postidx - net.inputNum] > net.params.get('FIRETH', 0.001):
            # TODO David: maybe also add refractory time for testing later.
            if postidx >= net.NETSIZE - net.outputNum:  # No refractory time at output layer.
                wantToFire[postidx] = 1
            elif net.state.LastFiringTimes[postidx-net.inputNum] == -1 or time - net.state.LastFiringTimes[postidx-net.inputNum] >= refractory_t:
                wantToFire[postidx] = 1
                net.state.LastFiringTimes[postidx - net.inputNum] = time

        if postidx == net.NETSIZE-3:  # Set new inputs before moving to out layer.
            full_stim = np.bitwise_or([int(x) for x in wantToFire], [int(x) for x in inputStim]) #TODO David...

        net.log("POST=%d NeurAccum=%g in step %d" % (postidx, net.state.NeurAccum[time][postidx - net.inputNum], time))
        net.state.firingCellsPseudo = wantToFire # fire hist without wta. This info will be used to reset data.

    net.state.firingCells = wantToFire
    if phase == 'test':
        net.state.fireHistForTest[:-1, np.where(np.array(full_stim) != 0)[0]] = \
            net.state.fireHistForTest[1:, np.where(np.array(full_stim) != 0)[0]]
        # Save last firing time for all cells that fired in this time step.
        net.state.fireHistForTest[net.DEPTH, np.where(np.array(full_stim) != 0)[0]] = \
            time
        # Load 'NN'.
        net.state.fireCellsForTest[time] = wantToFire
    else:
        # Barrel shift history
        net.state.fireHist[:-1, np.where(np.array(full_stim) != 0)[0]] = \
            net.state.fireHist[1:, np.where(np.array(full_stim) != 0)[0]]
        # Save last firing time for all cells that fired in this time step.
        net.state.fireHist[net.DEPTH, np.where(np.array(full_stim) != 0)[0]] = \
            time
        net.state.fireCells[time] = wantToFire

    net.state.firingCells = wantToFire
    net.state.errorList = wantToFire - outputLabel

    if time > 0:
        net.state.trace[time] = net.state.trace[time-1]
    net.state.trace[time] *= (1-leakage)
    net.state.trace[time] = np.add(net.state.trace[time], old_full_stim[:-2])

    dampening_factor = 0.3

    tmp = dampening_factor * (1.0 - np.abs((net.state.NeurAccum[time][:-net.outputNum] - threshold) / threshold))
    net.state.dz_dh[time] = np.where(tmp > 0., tmp, 0.) / threshold
    for i in range(len(net.state.dz_dh[time])):
        if not (net.state.LastFiringTimes[i] == time or net.state.LastFiringTimes[i] == -1 or time - net.state.LastFiringTimes[i] >= refractory_t):
            net.state.dz_dh[time][i] = 0.

    print(f"Output angles time={time}: {net.state.NeurAccum[time][[net.NETSIZE - net.inputNum - 2]]}, {net.state.NeurAccum[time][[net.NETSIZE - net.inputNum - 1]]}")


def plast(net, time):
    pass


def additional_data(net):
    return None
