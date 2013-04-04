from __future__ import division
from settings import load_settings

PLOT = False

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Experimentation")
    parser.add_argument('settings', metavar='<settings file>', type=argparse.FileType('r'), help='Filename of settings file')
    #parser.add_argument('--config', metavar='<config string>', nargs='+', type=str, help='Config string')
    parser.add_argument('--plot', action='store_true')

    args = parser.parse_args()
    settings_file = args.settings
    #configs = args.config
    configs = None
    PLOT = args.plot

    import matplotlib
    matplotlib.use('Agg')

import glob
import sys
import os
import gv
import numpy as np
import amitgroup as ag
import matplotlib.pylab as plt
from operator import itemgetter

if __name__ == '__main__':
    np.random.seed(0)

    settings = load_settings(settings_file)
    descriptor = gv.load_descriptor(settings)

def generate_random_patches(filenames, size, seed=0):
    print hash(tuple(filenames))
    print 'Getting backgrounds with seed ', seed
    randgen = np.random.RandomState(seed)
    for fn in filenames:
        img = gv.img.asgray(gv.img.load_image(fn))
        # Random position
        for l in xrange(100):
            x = randgen.randint(img.shape[0]-size[0]-1) 
            #y = 20+np.random.randint(2)*10#np.random.randint(img.shape[1]-size[1]-1)
            y = randgen.randint(img.shape[1]-size[1]-1)
            yield img[x:x+size[0], y:y+size[1]]

def get_bkg(filenames, N, seed=0):
    gen = generate_random_patches(filenames, (15, 15), seed=seed)
    tots = np.zeros(descriptor.num_parts)
    for i in xrange(N):
        img = gen.next()
        feats = descriptor.extract_features(img)
        tots += feats[1,1]
    return tots/N

# Some negatives
neg_filenames= sorted(glob.glob(os.path.join(os.environ['UIUC_DIR'], 'TrainImages', 'neg-*.pgm')))

def load_and_crop(fn):
    img, alpha = gv.img.load_image_binarized_alpha(fn)
    img0 = img[10:25,10:25]
    if alpha is None or alpha.mean() > 0.9:
        alpha0 = None
    else: 
        alpha0 = alpha[10:25,10:25]
    return alpha0, gv.img.asgray(img0)
    
    #return gv.img.sgray(img), alpha 
    #if img[...,3].mean() > 0.9:
    #    alpha = None
    #else:
    #    alpha = img[10:25,10:25,3]
    #return (alpha > 0.2), gv.img.asgray(img[10:25,10:25])

def composite(fg_img, bg_img, alpha):
    return fg_img * alpha + bg_img * (1 - alpha) 

def arrange_model_star(settings_config):
    return arrange_model(*settings_config)

def arrange_model(settings, config, offset=None, mods=None):
    print 'Arranging model'
    if offset is None:
        offset = settings['detector'].get('train_offset', 0)
    limit = settings['detector'].get('train_limit')
    if limit is not None:
        limit += offset
    files = sorted(glob.glob(settings['detector']['train_dir']))[offset:limit] * settings['detector'].get('duplicate', 1)
    alpha_and_images = map(load_and_crop, files)
    if alpha_and_images[0][0] is None:
        alpha = None
        all_alphas = None
    else:
        all_alphas = np.asarray(map(itemgetter(0), alpha_and_images))
        #all_alphas = np.asarray(map(lambda x: x[0], alpha_and_images))
        alpha = all_alphas[:,7-4:8+4,7-4:8+4].mean(axis=0)

    if 0 and PLOT and alpha is not None:
        plt.clf()
        ag.plot.images([alpha])
        plt.savefig('outs/alpha.png')
    #np.save('_alpha.npy', alpha) 
    
    images = np.asarray(map(itemgetter(1), alpha_and_images))

    edgy = ag.features.bedges(images, **settings['edges'])[0,1:-1,1:-1]

    if config.startswith('bkg'):
        seed = int(config[3:])
        neg_gen = generate_random_patches(neg_filenames, (15, 15), seed=seed)
        for i in xrange(len(images)):
            # Superimpose it onto the negative patch
            images[i] = neg_gen.next()
        
    elif config.startswith('sup'):
        seed = int(config[3:])
        neg_gen = generate_random_patches(neg_filenames, (15, 15), seed=seed)
        for i in xrange(len(images)):
            # Superimpose it onto the negative patch
            images[i] = composite(images[i], neg_gen.next(), all_alphas[i])
    

    #edges = ag.features.bedges(images, **settings['edges'])
    descriptor = gv.load_descriptor(settings) 

    #radii = settings['detector']['spread_radii']

    
    #feats = np.asarray(map(descriptor.extract_features, images))
    all_edges = ag.features.bedges(images, **settings['edges'])

    if 1:
        #alpha1 = np.load('_alpha.npy')
        if 0:
            sett = settings['edges'].copy()
            sett['radius'] = 1
            sett['spread'] = 'box'
            sett['k'] = 3
            aedges = ag.features.bedges(all_alphas[0].astype(np.float64), **sett)
            #aedges[...,[1,2,3]] = aedges[...,[0]]
            #aedges[5:7,-3:,:] = 1
            #import pdb; pdb.set_trace()
            #np.save('_edges.npy', aedges[1:-1,1:-1])

            plt.clf()
            ag.plot.images(np.rollaxis(aedges, 2), show=False)
            plt.savefig('outs/aedges.png')
        #if PLOT or 1:
        #    ag.plot.images(np.rollaxis(aedges, 2))
        #print all_edges.shape, aedges.shape
        #import pdb; pdb.set_trace()
        #all_edges *= (1-aedges)
        #no_feats = np.asarray(map(descriptor.extract_features, all_alphas))
        #feats &= (1-no_feats)

    edge_patch = all_edges[:,1:-1,1:-1]

    if mods is not None:
        #blackout = np.load('_blackout.npy')
        #blackin = np.load('_blackin.npy')

        mask = ~(mods.mean(axis=0).mean(axis=0) > 0.00001)
        
        #mask = ag.util.zeropad(~((blackout > 0) | (blackin > 0)), (1, 1, 0))
        #mask = ag.util.zeropad(((mods > 0.0001) | (blackin > 0.0001)), (1, 1, 0))
        #mask = ag.util.zeropad(~((blackin > 0)), (1, 1, 0))
        #edge_patch &= mask 
        

    if 0 and PLOT:
        edges = all_edges#ag.features.bedges(images, **settings['edges'])
        edges_ = np.rollaxis(edges, 3, start=1)
        print (np.prod(edges_.shape[:2]),) + edges_.shape[2:]
        pledges = edges_.reshape((np.prod(edges_.shape[:2]),) + edges_.shape[2:])

        #ag.plot.images([alpha])

        #print edges.shape
        plt.clf()
        ag.plot.images(pledges[:,1:-1,1:-1], subplots=edges_.shape[:2], show=False)
        plt.savefig('outs/edges-{0}.png'.format(config))


    feats = np.asarray(map(descriptor.extract_parts, edge_patch))
    #import pdb; pdb.set_trace()

    #import pdb; pdb.set_trace()
    
    return {
        'settings': settings, 
        'theta': feats[:,0,0].mean(axis=0), 
        'alpha': alpha.astype(np.bool),
        'edgy': edgy,
        'edges': edge_patch.astype(np.bool),
    }
    #return images

#print map(lambda x: np.sum(x[1]), models)

def weighted_choice(x, randgen=np.random):
    xcum = np.cumsum(x) 
    r = randgen.uniform(0, xcum[-1])
    return np.where(r < xcum)[0][0]

def weighted_choice_unit(x, randgen=np.random):
    xcum = np.cumsum(x) 
    r = randgen.rand()
    w = np.where(r < xcum)[0]
    if w.size == 0:
        return -1
    else:
        return w[0]
    return 

def get_probs(theta, f):
    if f == -1:
        return 0
    else:
        return theta[f]

def correct_model(model, bkg=None, model_bkg=None, seed=0, mods=None):
    settings = model['settings']
    feats = model['theta']
    alpha = model['alpha']
    edgy = model['edgy']
    descriptor = gv.load_descriptor(settings)
    N = settings['detector']['train_limit'] * settings['detector']['duplicate']# * 20
    num_features = feats.size
    part_counts = np.zeros(num_features)
    num_edges = 4 

    if alpha is None:
        alpha = np.ones((9, 9))
    p_alpha = alpha
    p_kernel = feats 
    if bkg is not None:
        good_back = p_back = bkg
    else:
        good_back = p_back = np.load('bkg2_nospread.npy')

    #ealpha = np.load('_edges.npy').astype(np.bool)

    Xs = []

    #blackout0 = np.load('_blackout.npy')
    #blackin0 = np.load('_blackin.npy')

    #bm = ag.stats.BernoulliMixture.load('_mix.npy') 
    #mods = np.load('_mods.npy') 

    neg_gen = generate_random_patches(neg_filenames, (15, 15), seed=seed)

    #theta = 1 - (1 - descriptor.parts)**(1/9)
    theta = descriptor.parts

    cumX = None

    for loop in xrange(N): 
        #if loop % 1000 == 0:
        #    print 'loop', loop
        randgen = np.random.RandomState(seed+loop)
        randgen2 = np.random.RandomState(seed+loop + 4)
        randgen3 = np.random.RandomState(seed+loop + 23)
        randgen4 = np.random.RandomState(seed+loop + 100)
        randgen5 = np.random.RandomState(seed+loop + 231)
        randgen6 = np.random.RandomState(seed+loop + 232)
        randgen7 = np.random.RandomState(seed+loop + 232)

        f_obj = weighted_choice_unit(p_kernel, randgen)
        probs_obj = get_probs(theta, f_obj)

        #import pdb; pdb.set_trace()
        f_bkg = weighted_choice_unit(good_back, randgen)
        probs_bkg = get_probs(theta, f_bkg) 

        if 0:
            if f_obj == -1 and f_bkg == -1:
                pass # Add nothing
            elif f_obj == -1 and f_bkg != -1:
                part_counts[f_bkg] += 1
            elif f_obj != -1 and f_bkg == -1:
                part_counts[f_obj] += 1 
        if 0:
            pass
        else:
            # Draw from the alpha
            #A = (randgen2.rand(*p_alpha.shape) < p_alpha).astype(np.uint8) 
            A = (randgen2.rand() < p_alpha)
            #A = (0.5 < p_alpha).astype(np.uint8)
            AA = A.reshape(A.shape + (1,)).astype(np.bool)

            #import pdb; pdb.set_trace()
            FIXED_OBJ = True

            if FIXED_OBJ:
                probs_mixed = ~AA * probs_bkg 
            else:
                probs_mixed = AA * probs_obj + ~AA * probs_bkg 
            #probs_mixed = probs_obj 

            if f_obj != -1 or f_bkg != -1:
            #if f_obj != -1:# or f_bkg != -1:

                #print probs_mixed.shape
                if not FIXED_OBJ:
                    X = (randgen3.rand(*probs_mixed.shape) < probs_mixed)

                else:
                    X = np.zeros_like(edgy, dtype=np.bool)
    
                    if 1:
                        Y = model_bkg['edges'][loop]
                        X |= ~AA & Y
                        
                        # What f_bkg is this?
                        f_bkg = np.argmax(descriptor.extract_parts(Y.astype(np.uint8))[0,0])
                    elif f_bkg != -1:
                        # Draw samples from the mixture components
                        X |= (randgen3.rand(*X.shape) < probs_mixed)
                        #X = (randgen3.rand() < probs_mixed).astype(np.uint8)
                        #X = (1 - AA) * ag.features.bedges(neg_gen.next(), **settings['edges'])[1:-1,1:-1] 
                    X |= edgy

                #X *= (1 - ealpha)
                #r = randgen4.uniform(0, 0.5)
                #mask = ~ealpha | ~(randgen4.rand(*X.shape) < 0.28)

                #print '----'
                #print np.rollaxis(mask, 2)
                #X &= mask 

                #X &= ((blackout0 > 0.0001) | (blackin0 > 0.0001)) 
    
                # Draw which blackout/in component
                if 0:
                    f_comp = weighted_choice_unit(bm.weights, randgen6)
                    assert f_comp >= 0  
                    blackout = bm.templates[f_comp,0]
                    blackin = bm.templates[f_comp,1]
                elif f_bkg != -1:
                    blackout = mods[f_bkg,0]
                    blackin = mods[f_bkg,1]
                    
                    mask = ~(randgen4.rand(*X.shape) < blackout)
                    mask2 = (randgen5.rand(*X.shape) < blackin)
                    #mask = ~(randgen4.rand() < blackout)
                    #import pdb; pdb.set_trace()
                    #Xmask = X & mask
                    Xmask2 = ~X & mask2
                    
                    X &= mask 
                    X |= Xmask2
                    #if randgen7.rand()>0.5:
                    #else:
                        #X &= mask 
                        #X |= mask2

                
                mask = ~(mods.mean(axis=0).mean(axis=0) > 0.00001)
                #X &= mask

                #X &= ~((blackout > 0) | (blackin > 0)) 
                #X &= ~((blackin > 0)) 

                
                # Now, do edge spreading!
                #import pdb; pdb.set_trace()
                #X = ag.features.bspread(X, spread=settings['edges']['spread'], radius=settings['edges']['radius'])

                #if PLOT:
                #    Xs.append(X)    
                if cumX is None:
                    cumX = X.astype(int)
                else:
                    cumX += X

                #print X
                
                part_counts += descriptor.extract_parts(X.astype(np.uint8))[0,0]

            if 0:
                if X.sum() >= settings['parts']['threshold']:

                    # Check which part this is most similar to
                    scores = np.apply_over_axes(np.sum, X * np.log(descriptor.parts) + (1 - X) * np.log(1 - descriptor.parts), [1, 2, 3]).ravel()
                    f_best = np.argmax(scores)
                    #f_best = np.argmax(np.apply_over_axes(np.sum, np.fabs(self.descriptor.parts - X), [1, 2, 3]).ravel())
                    part_counts[f_best] += 1
                
                    #import pdb; pdb.set_trace()
                    
                    #p = _integrate(integral_aa_log[mixcomp], i, j, i+istep, j+jstep)

            #if f_bkg != -1:
            #    part_counts[f_bkg] += 1

            #part_counts += descriptor.extract_parts(X)[0,0]
            # Or do it this way:
            #import pdb; pdb.set_trace()
            #feats = descriptor.extract_parts(X)
            #print f_best, feats

    new_feats = part_counts / N

    if PLOT:
        plt.clf()
        ag.plot.images(np.rollaxis(cumX, 2)/N)
        plt.savefig('outs/mean-cor.png')

    if PLOT and 0:
        plt.clf()
        Xs = np.asarray(Xs)
        Xs_ = np.rollaxis(Xs, 3, start=1)
        plXs = Xs_.reshape((np.prod(Xs_.shape[:2]),) + Xs_.shape[2:])
        ag.plot.images(plXs, subplots=Xs_.shape[:2], show=False)
        plt.savefig('outs/corrected.png')

    new_model = model.copy()
    new_model['theta'] = new_feats
    new_model['alpha'] = None
    return new_model

def test_model(offset, seed=1):
    global settings
    model_non = arrange_model(settings, 'none', offset=offset)
    model_bkg = arrange_model(settings, 'bkg'+str(seed), offset=offset)
    model_sup = arrange_model(settings, 'sup'+str(seed), offset=offset)

    num_features = model_non['theta'].size

    edges_non = model_non['edges'] 
    edges_bkg = model_bkg['edges']
    edges_sup = model_sup['edges'] 
    
    # Get coded parts from the sup model
    coded_parts_bkg = np.asarray(map(lambda x: descriptor.extract_parts(x.astype(np.uint8))[0,0], edges_bkg))

    coded_parts = np.asarray(map(lambda x: descriptor.extract_parts(x.astype(np.uint8))[0,0], edges_sup))

    #eqv12 = ~(edges_bkg ^ edges_sup)
    #eqv02 = ~(edges_non ^ edges_sup)
    #either = eqv12 | eqv02
    alpha = model_non['alpha']
    A = alpha.reshape(alpha.shape+(1,))

    #disappeared = (~eqv02 & edges_non)
    #disappeared = (edges_non ^ edges_sup) & edges_non
    disappeared_inside = A & edges_non & ~edges_sup
    disappeared_outside = ~A & (edges_non | edges_bkg) & ~edges_sup

    present_inside = A & edges_non
    present_outside = ~A & (edges_non | edges_bkg)

    # TODO: Add another term here for things outside the support. Will need the background.
    #appeared = alpha.reshape(alpha.shape+(1,)) & (~edges_non & edges_sup)
    appeared_inside = A & ~edges_non & edges_sup
    appeared_outside = ~edges_bkg & ~edges_non & edges_sup

    nonpresent_inside = A & ~edges_non
    nonpresent_outside = ~A & ~edges_bkg & ~edges_non

    disappeared = disappeared_inside | disappeared_outside
    appeared = appeared_inside | appeared_outside

    present = present_inside | present_outside
    nonpresent = nonpresent_inside | nonpresent_outside

    plt.clf()
    ag.plot.images(np.rollaxis(disappeared.mean(axis=0), 2))
    plt.savefig('outs/disappeared.png')

    plt.clf()
    ag.plot.images(np.rollaxis(appeared.mean(axis=0), 2))
    plt.savefig('outs/appeared.png')

    #import pdb; pdb.set_trace()

    import sys
    eps = sys.float_info.epsilon

    # Set up the blackin/out model
    mods = np.zeros((num_features, 2, 9, 9, 4)) # Don't hard code this
    for f in xrange(num_features):
        parts = coded_parts_bkg[:,f] == 1
        if parts.sum() > 0:
            mods[f,0] = disappeared[parts].sum(axis=0) / (present[parts].sum(axis=0) + eps)
            mods[f,1] = appeared[parts].sum(axis=0) / (nonpresent[parts].sum(axis=0) + eps)

    # Now, correct the model
    N = settings['detector']['duplicate'] * settings['detector']['train_limit']
    bkg = get_bkg(neg_filenames, N, seed=seed)

    model_cor = correct_model(model_non, bkg=bkg, model_bkg=model_bkg, seed=seed, mods=mods)

    model_sup = arrange_model(settings, 'sup'+str(seed), offset=offset, mods=mods)
    edges_sup = model_sup['edges'] 

    plt.clf()
    ag.plot.images(np.rollaxis(edges_sup.mean(axis=0), 2) )
    plt.savefig('outs/mean-sup.png')

    plt.clf()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.plot(model_sup['theta'], label='sup')
    plt.plot(model_cor['theta'], label='cor')
    plt.legend()
    print 'Saving final-{0}.png'.format(offset)
    plt.savefig('outs/final-{0}.png'.format(offset))

    diff = np.sum((model_cor['theta'] - model_sup['theta'])**2)/num_features * 1000000
    return diff

if __name__ == '__main__':
    ONE = True
    THREADING = True and not ONE
    if THREADING:
        from multiprocessing import Pool
        p = Pool(7)
        mymap = p.map
    else:
        mymap = map
    
    if not configs:
        offsets = 14 if ONE == False else 1
        scores = np.zeros(offsets)
        #def test_offset(offset):
            #return test_model(settings_objs[0], offset)
        #settings = load_settings(settings_objs[0])
        
        #print zip([settings_file]*offsets, range(offsets))
        scores = np.asarray(mymap(test_model, range(offsets)))
        #for offset in xrange(offsets):
            #scores[offset] = test_model(settings_objs[0], offset+4)

        print 'scores', scores
        print 'average', scores.mean()
        print 'std', scores.std()
    else:
        models = mymap(arrange_model_star, zip(settings_objs, configs))

         

        # Correct models with alpha
        print 'Starting'
        if 1:
            for i in xrange(len(models)):
                if configs[i].startswith('cor'): 
                    seed = int(configs[i][3:])
                    N = settings_objs[i]['detector']['duplicate'] * settings_objs[i]['detector']['train_limit']
                    bkg = get_bkg(neg_filenames, N, seed=seed)
                    models[i] = correct_model(models[i], bkg=bkg, seed=seed)

        if 0:
            settings, feats, alpha = models[0] 
            #print alpha.shape
            #import pdb; pdb.set_trace()
            if alpha is not None:
                plt.imshow(alpha, interpolation='nearest')
                plt.colorbar()
                plt.show()


        if len(models) >= 2:
            num_features = models[0]['theta'].size
            print np.sum((models[0]['theta'] - models[1]['theta'])**2)/num_features * 1000000, 'ppm'

        if PLOT or 1:
            plt.clf()
            plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
            for i, model in enumerate(models):
                print configs[i], np.where(model['theta'] > 0.1)
                plt.plot(model['theta'], label=configs[i])
            plt.legend()
            plt.savefig('outs/final.png')

        if 0:
            arr = None
            for i, (settings, model, alphas) in enumerate(models):
                if arr is None:
                    arr = model
                else:
                    arr += model 

            plt.plot(arr/len(models))
            plt.show()

            
        #import pdb; pdb.set_trace()
        #plt.imshow(models[0][0])
        #plt.show()
