
import amitgroup as ag
import numpy as np
import matplotlib.pylab as plt

def surplus(costs, correct_label):
    """Metric for how close the correct score is from the current minimum score. Positive value is good."""
    mcost = costs[0][0]
    mdigit= costs[0][1]

    diff = 0 
    # The correct one is at the top
    if mdigit == correct_label:
        # Calculate where the next is
        for t in costs:
            cost, digit, mix_component = t
            if digit != correct_label:
                # We're dividing by cost, so that we don't promote classifiers that reduce the cost
                # for all classes, without improving the disciminative power.
                # mcost is the correct one
                diff = (cost - mcost)/mcost
                break

    # An incorrect one is at the top
    else:
        # Find the correct labels min cost
        for t in costs:
            cost, digit, mix_component = t 
            if digit == correct_label:
                # cost is the correct one
                diff = (mcost - cost)/cost
                break

    return diff 


# Classifer 
def classify(features, all_templates, means, variances, samples=None, use_deformation=True, correct_label=None, threshold_multiple=1.2, b0 = None, lmb0 = None, debug_plot=False):
    # min loglikelihood
    min_cost = None
    min_which = None
    costs = []
    for digit, templates in enumerate(all_templates):
        # Clip them, to avoid 0 probabilities

        for mix_component, template in enumerate(templates):
            #assert features.shape == template.shape
            # Compare mixture with features
            cost = -np.sum(features * np.log(template) + (1-features) * np.log(1-template))
            #print("Cost {0} digit {1} comp {2}".format(cost, digit, mix_component))
            costs.append( (cost, digit, mix_component) )
            if min_cost is None or cost < min_cost:
                min_cost = cost 
                min_which = (digit, mix_component) 

    costs = filter(lambda t: t[0] < min_cost * threshold_multiple, costs)
    costs.sort()

    info = {}

    # If all the costs left are the same digit, don't bother doing the deformation
    checked = [] 
    for t in costs:
        if t[1] not in checked:
            checked.append(t[1])
    

    info['contendors'] = len(checked)
    info['surplus_change'] = 0.0

    if len(checked) != 1:
        if correct_label is not None:
            info['surplus_before'] = surplus(costs, correct_label)

        # Filter so that we have only one of each mixture (Not necessary, could even damage results!)
        if 0:
            for i, t in enumerate(costs):
                if t[1] not in checked:
                    checked.append(t[1])
                else:
                    del[costs[i]]

        if use_deformation:
            new_costs = []
            for t in costs:
                cost, digit, mix_component = t 
                #print("Doing cost {0} digit {1} mix_component {2}".format(*t))
                # Do a deformation
                #means[digit, mix_component]
            
                F = all_templates[digit][mix_component]
                I = features
            
                #means=means[digit, mix_component], variances=variances[digit, mix_component], 
                #variances[digit, mix_component, :, 0, :, :, :] = 0.0009765625 
                #variances[digit, mix_component, :, 1, :, :, :] = 0.00390625 
                #variances[digit, mix_component, :, 2, :, :, :] = 0.0625 
                #variances[digit, mix_component, :, 3, :, :, :] = 16.0 
                #variances[digit, mix_component, :, -2:] = 0.0
                var = variances[digit, mix_component]
    
                # Calculate the posterior variance
                if b0 and lmb0 and samples is not None:
                    new_var = (b0 + samples*var/2) / (b0 * lmb0 + samples/2)
                    var = new_var

                #var = None

                me = means[digit, mix_component]
                #me = None
                penalty = None 

                #print(variances.shape)
                #print(llh_variances.shape)
                #print('prior vars:', variances[digit, mix_component,:,0])
                #print('data vars: ', llh_variances[digit, mix_component])

                #print('var:', var[:,:16].mean(), var[:,:(64-48)].min(), var[:,:16].max())
                #import sys; sys.exit(0)

                #print 'around:', means[digit, mix_component].mean()/var.mean()

                #print("var0:", var[0,0], var[1,0])

            
                if 0:
                    me = None
                    var = None
                    penalty = 100000.0

                #ag.plot.images(F)

                imdef, information = ag.stats.bernoulli_deformation(F, I, wavelet='db4', penalty=penalty, means=me, variances=var, start_level=0, last_level=3, debug_plot=debug_plot, gtol=0.1, maxiter=5)

                # Kill if cancelled
                imdef or sys.exit(0)

                new_cost = information['cost']
                # Update the cost 
                ag.info("{0:4.1f} --> {1:4.1f}".format(cost, new_cost))

                new_costs.append( (new_cost, digit, mix_component) )

                if new_cost < min_cost:
                    min_cost = new_cost
                    min_which = (digit, mix_component)


            # Reevaluate the surplus!
            if correct_label is not None:
                info['surplus_after'] = surplus(new_costs, correct_label)
                info['surplus_change'] = info['surplus_after'] - info['surplus_before']

    info['comp'] = min_which[1]

    return min_which[0], info


