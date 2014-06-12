

import numpy as np


def get_key_points(weights, suppress_radius=2, max_indices=np.inf, even=False): 
    indices = []
    #kern = detector.kernel_templates[k][m]
    #bkg = detector.fixed_spread_bkg[k][m]
    #eps = detector.settings['min_probability']
    #kern = np.clip(kern, eps, 1 - eps)
    #bkg = np.clip(bkg, eps, 1 - eps)
    #weights = np.log(kern / (1 - kern) * ((1 - bkg) / bkg))
    #print('Keypointing -------')
    #print('Even', even)


    suppress_radii = np.array([1, 2, 1])

    # TEMP{
    if 0:
        rotspread = 1
        ORI = 24 
        F = weights.shape[-1]
        between_feature_spreading = np.zeros((F, rotspread*2 + 1), dtype=np.int32)

        for f in range(F):
            thepart = f // ORI
            ori = f % ORI 
            for i in range(rotspread*2 + 1):
                between_feature_spreading[f,i] = thepart * ORI + (ori - rotspread + i) % ORI

    # }TEMP

    F = weights.shape[-1]

    rs = np.random.RandomState(0)

    # Add some noise, since we want each feature to have the same amount of features,
    # we must perturb the weights a bit since everyone is relying on the minimum amount.
    ww = weights
    absw_pos = np.maximum(0, ww)# + rs.normal(0, 0.00000001, size=weights.shape)
    absw_neg = -np.minimum(0, ww)# + rs.normal(0, 0.00000001, size=weights.shape)

    import scipy.stats
    #almost_zero = scipy.stats.scoreatpercentile(np.fabs(weights.ravel()), 20)
    almost_zero = 0

    indices_weights = []

    #supp = detector.settings.get('indices_suppress_radius', 4)
    for absw in [absw_pos, absw_neg]:
        while True: 
            if absw.max() <= almost_zero:
                break
            ii = np.unravel_index(np.argmax(absw), absw.shape)
            indices.append(ii) 
            indices_weights.append(absw[ii])


            absw[max(0, ii[0]-suppress_radius):ii[0]+suppress_radius+1, max(0, ii[1]-suppress_radius):ii[1]+suppress_radius+1,ii[2]] = 0.0
            #absw[max(0, ii[0]-suppress_radius):ii[0]+suppress_radius+1, max(0, ii[1]-suppress_radius):ii[1]+suppress_radius+1,between_feature_spreading[ii[2]]] = 0.0
            if 0:
                for j in range(3):
                    r = suppress_radii[j]
                    absw[max(0, ii[0]-r):ii[0]+r+1, max(0, ii[1]-r):ii[1]+r+1,between_feature_spreading[ii[2],j]] = 0.0

            if len(indices) >= max_indices:
                break

    # Sort indices by weight
    indices = np.asarray(indices, dtype=np.int32)
    II = np.argsort(indices_weights)[::-1]
    indices = indices[II]

    if not even:
        new_indices = indices
    #new_indices = []
    else: 
        Ls = np.bincount(indices[:,2], minlength=weights.shape[-1])
        L = int(np.median(Ls))

        print(('indices per features:', L))

        new_indices = np.zeros((L * F, 3), dtype=np.int32)

        for f in range(F):
            indf = indices[indices[...,2] == f]

        
            new_indices[L*f:L*f + min(L, indf.shape[0])] = indf[:L]

            # Fill up the rest with random indices that are not taken
            for l in range(int(indf.shape[0]), L):
                while True:
                    l0, l1 = rs.randint(weights.shape[0]), rs.randint(weights.shape[1])

                    # Check if it's in there already
                    if not (indices == [l0, l1, f]).all(1).any():
                        new_indices[L*f + l] = (l0, l1, f)
                        break

    #print('Indices:', len(new_indices))
    return new_indices

    if 0:
        Ls = np.bincount(indices[:,2], minlength=weights.shape[-1])
        min_L = np.min(Ls)
        #print min_L, np.max(Ls)
        #print indices_weights
        new_indices = []
        bins = np.zeros(weights.shape[-1])
        for index in indices:
            if bins[index[2]] < min_L:
                new_indices.append(index)
                bins[index[2]] += 1

        new_indices = np.array(new_indices, dtype=np.int32)

        # Sort
        new_indices = new_indices[np.argsort(new_indices[:,2])] 

        return new_indices
        
