from __future__ import division, print_function
import os
import argparse
import gv.datasets
import sys

def detect_raw(args):
    #logger.info('Entering')
    import os
    #print("entering detect_raw", os.getpid())
    detector, fileobj = args
    import textwrap
    import gv
    import amitgroup as ag
    import numpy as np

    detections = []
    img = gv.img.load_image(fileobj.path)
    grayscale_img = gv.img.asgray(img)
    
    tp = tp_fp = tp_fn = 0

    # Count tp+fn
    for bbobj in fileobj.boxes:
        if not bbobj.difficult:
            tp_fn += 1 

    #logger.info('Detecting Start')
    bbs = detector.detect(grayscale_img, fileobj=fileobj)
    #logger.info('Detecting Done')

    tp_fp += len(bbs)
    
    for bbobj in bbs:
        #print("{0:06d} {1} {2} {3} {4} {5}".format(fileobj.img_id, bbobj.confidence, int(bbobj.box[0]), int(bbobj.box[1]), int(bbobj.box[2]), int(bbobj.box[3])), file=fout)
        detections.append((bbobj.confidence, bbobj.scale, bbobj.score0, bbobj.score1, bbobj.plusscore, bbobj.correct, bbobj.mixcomp, bbobj.bkgcomp, fileobj.img_id, int(bbobj.box[1]), int(bbobj.box[0]), int(bbobj.box[3]), int(bbobj.box[2]), bbobj.index_pos[0], bbobj.index_pos[1]))
        #fout.flush()
        if bbobj.correct and not bbobj.difficult:
            tp += 1

    #print("exiting detect_raw", os.getpid())
    #logger.info('Exiting')
    return (tp, tp_fp, tp_fn, bbs, fileobj.img_id)

if __name__ == '__main__' and gv.parallel.main():
    parser = argparse.ArgumentParser(description='Test response of model')
    parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
    parser.add_argument('obj_class', metavar='<object class>', type=str, help='Object class')
    parser.add_argument('output', metavar='<output file>', type=argparse.FileType('wb'), help='Filename of output file')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--mini', action='store_true', default=False)
    parser.add_argument('--contest', type=str, choices=gv.datasets.contests(), default='voc-val', help='Contest to try on')
    parser.add_argument('--no-threading', action='store_true', default=False, help='Turn off threading')
    parser.add_argument('--log', type=str, default=None, help='Log to this directory name')
    parser.add_argument('--size', type=int, default=None, help='Use fixed size')
    parser.add_argument('--param', type=float, default=None)

    args = parser.parse_args()
    model_file = args.model
    obj_class = args.obj_class
    output_file = args.output
    limit = args.limit
    offset = args.offset
    mini = args.mini
    threading = not args.no_threading
    contest = args.contest
    logdir = args.log
    logging = args.log is not None

    if logging:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pylab as plt

    EDGE_TITLES = ['- Horizontal', '/ Diagonal', '| Vertical', '\\ Diagonal']

    import textwrap
    import gv
    import amitgroup as ag
    import numpy as np
    import os
    import itertools as itr

    if logging:
        try:
            os.mkdir(logdir)
        except OSError:
            print("Could not create folder {0}. Probably already exists.".format(logdir))
            sys.exit(1)

    def detect(fileobj):
        return detect_raw((detector, fileobj))

    detector = gv.Detector.load(model_file)
    # TODO: New
    detector.TEMP_second = True
    detector._param = args.param

    if args.size is not None:
        detector.settings['min_size'] = detector.settings['max_size'] = args.size

    #dataset = ['val', 'train'][mini]
    #dataset = ['val', 'train'][mini]
    #dataset = 'val'

    print("Loading files...")
    files, tot = gv.datasets.load_files(contest, obj_class)
    print("Done.")

    tot_tp = 0
    tot_tp_fp = 0
    tot_tp_fn = 0

    detections = []

    if mini:
        files = filter(lambda x: len(x.boxes) > 0, files)
    upper_limit = limit
    if upper_limit is not None:
        upper_limit += offset
    files = files[offset:upper_limit]

    all_kp_only_weights = None

    LOG_ALL = True 

    # Log the features and the model first
    if logging:
        # Log features
        if LOG_ALL:
            ag.info("Logging features...")

            d = detector.descriptor

            feature_dir = os.path.join(logdir, 'features')
            os.mkdir(feature_dir)

            # Visualize parts
            #plt.clf()
            plt.figure()
            ag.plot.images(d.visparts, show=False, zero_to_one=False)
            plt.savefig(os.path.join(feature_dir, 'visparts.png'))
            plt.close()

            # Show scores and counts of parts
            if 'scores' in d.extra and 'counts' in d.extra:
                #plt.clf()
                plt.figure()
                plt.plot(d.extra['scores'], label='Scores')
                plt.twinx() 
                plt.plot(d.extra['counts'], label='Counts')
                plt.legend(fontsize='xx-small', framealpha=0.2)
                plt.savefig(os.path.join(feature_dir, 'scores-and-counts.png'))
                plt.close()

            parts_dir = os.path.join(feature_dir, 'parts')
            os.mkdir(parts_dir)

            originals = d.extra.get('originals')
            for pi in xrange(d.num_features):
                #plt.clf()
                f = plt.figure()
                # Look inside view_bkg_stack.py for code to go here.            
                plt.subplot2grid((7,10), (0, 0), colspan=4, rowspan=4).set_axis_off()
                plt.imshow(d.visparts[pi], interpolation='nearest', cmap=plt.cm.gray) 

                for i in xrange(4):
                    plt.subplot2grid((7,10), (2* (i//2), 6+(i%2)*2), colspan=2, rowspan=2).set_axis_off()
                    plt.imshow(d.parts[pi,...,i], interpolation='nearest', vmin=0, vmax=1)
                    plt.title(EDGE_TITLES[i])
                    #if i == 3:

                if False and originals:
                    for i in xrange(min(20, len(originals))):
                        plt.subplot2grid((7,10), (5+i//10, i%10)).set_axis_off()
                        plt.imshow(originals[pi][i], interpolation='nearest', vmin=0, vmax=1, cmap=plt.cm.gray)
                        if i == 4:
                            plt.title('Original background parts')

                f.savefig(os.path.join(parts_dir, '{}.png'.format(pi)))
                plt.close()


        # Log model
        if LOG_ALL:
            all_kp_only_weights = []

            ag.info("Logging model...")
            model_dir = os.path.join(logdir, 'model')
            os.mkdir(model_dir)

             
            from view_mixtures import view_mixtures
            view_mixtures(detector, output_file=os.path.join(model_dir, 'mixtures.png'))

            svms = detector.extra.get('svms')
            
            for m in xrange(detector.num_mixtures):
                mixture_dir = os.path.join(model_dir, 'mixture{}'.format(m))
                os.mkdir(mixture_dir)

                mixture_kernel_dir = os.path.join(mixture_dir, 'kernel')
                os.mkdir(mixture_kernel_dir)

                base_weights = detector.weights(m)
                base_max_abs = np.fabs(base_weights).max()

                if svms is not None:
                    svm_weights = svms[m]['weights'].reshape(base_weights.shape)
                    svm_max_abs = np.fabs(svm_weights).max()
                else:
                    svm_weights = None

                # Visualize keypoints and weights with only keypoints
                kp = np.zeros(base_weights.shape, dtype=np.uint8)
                kp_mean_weights = np.zeros(base_weights.shape[:2])
                II = detector.indices[m]
                kp_weights = np.empty(len(II))
                kp_only_weights = np.zeros(base_weights.shape)
                for i, index in enumerate(II):
                    kp[tuple(index)] = 1 
                    kp_mean_weights[index[0],index[1]] += base_weights[tuple(index)]
                    kp_weights[i] = base_weights[tuple(index)]
                    kp_only_weights[tuple(index)] = base_weights[tuple(index)]

                all_kp_only_weights.append(kp_only_weights)
                kp_mean_weights /= len(II)
                kp_percentage = np.prod(kp_mean_weights.shape) / len(II)

                # Histogram of weights for base weights with keypoints
                plt.figure()
                plt.hist(kp_weights, 25)
                plt.title('Histogram of base weights over keypoints ({:.02f}% of features)'.format(kp_percentage))
                plt.savefig(os.path.join(mixture_dir, 'histogram-kp.png'))
                plt.close()

                # Histogram of weights for base weights
                plt.figure()
                plt.hist(base_weights.ravel(), 50)
                plt.title('Histogram of all base weights')
                plt.savefig(os.path.join(mixture_dir, 'histogram-base.png'))
                plt.close()

                # Histogram of weights for SVM
                if svms is not None:
                    plt.figure()
                    plt.hist(svm_weights.ravel(), 50)
                    plt.title('Histogram of SVM weights')
                    plt.savefig(os.path.join(mixture_dir, 'histogram-svm.png'))
                    plt.close()

                # Key point density
                plt.figure() 
                plt.imshow(kp.sum(axis=-1), interpolation='nearest', cmap=plt.cm.cool)
                plt.colorbar()
                plt.savefig(os.path.join(mixture_dir, 'kp-density.png'))
                plt.close()

                # Average part density 
                plt.figure()
                part_density = detector.kernel_templates[m].mean(axis=-1)
                plt.imshow(part_density, interpolation='nearest', vmin=0, vmax=1, cmap=plt.cm.cool)
                plt.savefig(os.path.join(mixture_dir, 'part-density.png'))
                plt.close()

                # Base weight density
                plt.figure()
                mean_weights = base_weights.mean(axis=-1)
                mean_weights_max_abs = np.fabs(mean_weights).max()
                plt.imshow(mean_weights, interpolation='nearest', vmin=-mean_weights_max_abs, vmax=mean_weights_max_abs, cmap=plt.cm.RdBu_r)
                plt.savefig(os.path.join(mixture_dir, 'mean-kernel-base.png'))
                plt.close()

                # Base weight density for key points only
                plt.figure()
                kp_mean_weights_max_abs = np.fabs(kp_mean_weights).max()
                plt.imshow(kp_mean_weights, interpolation='nearest', vmin=-kp_mean_weights_max_abs, vmax=kp_mean_weights_max_abs, cmap=plt.cm.RdBu_r)
                plt.savefig(os.path.join(mixture_dir, 'mean-kernel-kp.png'))
                plt.close()

                # SVM weight density
                if svms is not None:
                    plt.figure()

                    mean_svm_weights = svm_weights.mean(axis=-1)
                    mean_svm_weights_max_abs = np.fabs(mean_svm_weights).max()
                    plt.imshow(mean_svm_weights, interpolation='nearest', vmin=-mean_svm_weights_max_abs, vmax=mean_svm_weights_max_abs, cmap=plt.cm.RdBu_r)
                    plt.savefig(os.path.join(mixture_dir, 'mean-kernel-svm.png'))
                    plt.close()

                for f in xrange(detector.num_features):
                    fig, ((ax_vispart, ax_keypoints), (ax_base, ax_svm)) = plt.subplots(2, 2)
                    ax_vispart.set_axis_off()
                    ax_vispart.imshow(detector.descriptor.visparts[f], interpolation='nearest', cmap=plt.cm.gray)
                    ax_vispart.set_title('Part {}'.format(f))

                    ax_keypoints.set_axis_off()
                    ax_keypoints.imshow(kp[...,f], interpolation='nearest')
                    ax_keypoints.set_title('Key points')

                    ax_base.set_axis_off()
                    ax_base.imshow(base_weights[...,f], interpolation='nearest', vmin=-base_max_abs, vmax=base_max_abs, cmap=plt.cm.RdBu_r)
                    ax_base.set_title('Base detector weights')
                    if svms is not None: 
                        ax_svm.set_axis_off()
                        ax_svm.imshow(svm_weights[...,f], interpolation='nearest', vmin=-svm_max_abs, vmax=svm_max_abs, cmap=plt.cm.RdBu_r)
                        ax_svm.set_title('SVM weights')
                    fig.savefig(os.path.join(mixture_kernel_dir, 'part{}.png'.format(f)))
                    plt.close()


    #fout = open("detections.txt", "w")


    
    res = gv.parallel.imap_unordered(detect_raw, itr.izip(itr.cycle([detector]), files))


    tp_fn_dict = {}

    detections_dir = None
    positives_dir = None
    negatives_dir = None

    parts_help = np.zeros((2, detector.num_features))

    if logging:
        averages = [np.zeros((2,) + detector.kernel_templates[m].shape) for m in xrange(detector.num_mixtures)] 

        averages_counts = np.zeros((detector.num_mixtures, 2), dtype=int)

    colors = np.array(['b', 'r'])

    if logging:
        detections_dir = os.path.join(logdir, 'detections')
        positives_dir = os.path.join(detections_dir, 'positives')
        negatives_dir = os.path.join(detections_dir, 'negatives')

        os.mkdir(detections_dir)
        os.mkdir(positives_dir)
        os.mkdir(negatives_dir)

    for loop, (tp, tp_fp, tp_fn, bbs, img_id) in enumerate(res):
        tot_tp += tp
        tot_tp_fp += tp_fp
        tot_tp_fn += tp_fn
        tp_fn_dict[img_id] = tp_fn
        for bbobj in bbs:
            detections.append((bbobj.confidence, bbobj.scale, bbobj.score0, bbobj.score1, bbobj.plusscore, bbobj.correct, bbobj.mixcomp, bbobj.bkgcomp, img_id, int(bbobj.box[1]), int(bbobj.box[0]), int(bbobj.box[3]), int(bbobj.box[2]), bbobj.index_pos[0], bbobj.index_pos[1]))
        #detections.extend(dets)

        # Log all positives
        if logging:
            active_weights_bins = np.linspace(-8, 8, 50)
            svm_active_weights_bins = np.linspace(-0.01, 0.01, 50)
            svms = detector.extra.get('svms')

            sett = detector.descriptor.settings['bedges']
            unspread_sett = sett.copy()
            unspread_sett['radius'] = 0

            for bbobj in bbs:
                pos_ok = bbobj.correct
                neg_ok = not bbobj.correct and bbobj.score0 >= detector.extra['cascade_threshold']

                #if (bbobj.correct == True or (bbobj and not bbobj.difficult:
                if (pos_ok or neg_ok) and not bbobj.difficult:
                    if pos_ok:
                        directory = positives_dir 
                    else:
                        directory = negatives_dir 

                    # Output to file
                    fig, axarr = plt.subplots(3, 5, figsize=(23, 15))
                    if bbobj.image is not None and np.min(bbobj.image.shape) > 0:
                        ax_original = axarr[0,0]
                        ax_original.set_axis_off()
                        ax_original.imshow(bbobj.image, interpolation='nearest', cmap=plt.cm.gray)
                        ax_original.set_title('Original (img_id = {})'.format(bbobj.img_id))

                        # Extract edges and display
                        unspread_edges = ag.features.bedges(bbobj.image, **unspread_sett)
                        edges = ag.features.bedges(bbobj.image, **sett)
                        for e in xrange(4):
                            ax = axarr[0,1+e]
                            ax.set_axis_off()
                            ax.imshow(edges[...,e].astype(int) - edges[...,e+4].astype(int), interpolation='nearest', vmin=-1, vmax=1, cmap=plt.cm.RdBu)
                            ax.set_title(EDGE_TITLES[e])

                    # An info box of text
                    ax_text = axarr[1,0]
                    ax_text.set_axis_off()
                    ax_text.set_xlim((0, 10))
                    ax_text.set_ylim((0, 10))
                    s = textwrap.dedent("""
                        Base score: {base_score:.03f}
                        Final score: {score:.03f}
                        Scale: {scale:.02f}
                        Correct: {correct}
                        Overlap metric: {overlap:.02f}
                        Image ID: {img_id}
                        """.format(base_score=bbobj.score0, 
                               score=bbobj.confidence,
                               scale=bbobj.scale,
                               correct=bbobj.correct,
                               overlap=bbobj.overlap or 0,
                               img_id=bbobj.img_id))

                    ax_text.text(1, 6, s, bbox=dict(facecolor='white'))

                    # Matched mixture component
                    ax_mixture = axarr[2,0]
                    ax_mixture.set_axis_off()
                    ax_mixture.imshow(detector.support[bbobj.mixcomp], interpolation='nearest', cmap=plt.cm.gray)
                    ax_mixture.set_title('Best matched component')

                    if all_kp_only_weights is not None and bbobj.X is not None:
                        # Spatial keypoint weights
                        ax_kp_spatial = axarr[1,1]
                        ax_kp_spatial.set_axis_off()
                        kp_only_weights = all_kp_only_weights[bbobj.mixcomp]
                        active_weights = kp_only_weights * bbobj.X
                        kp_mean = active_weights.mean(axis=-1)
                        mm = 0.1
                        ax_kp_spatial.imshow(kp_mean, interpolation='nearest', vmin=-mm, vmax=mm, cmap=plt.cm.RdBu_r)
                        ax_kp_spatial.set_title('Keypoints spatial weights')

                        # Parts keypoint weights
                        ax_kp_parts = axarr[1,2]
                        parts_weights = np.apply_over_axes(np.mean, active_weights, [0, 1])[0,0]
                        ax_kp_parts.bar(np.arange(detector.num_features), parts_weights, color=colors[(parts_weights > 0).astype(int)])
                        ax_kp_parts.set_xlabel('Part')
                        ax_kp_parts.set_ylabel('Weight average')
                        ax_kp_parts.set_title('Keypoints parts weights')

                        # Active weights histogram
                        ax_kp_histogram = axarr[1,3]
                        nonzero_weights = active_weights.ravel()[active_weights.ravel() != 0]
                        ax_kp_histogram.hist(nonzero_weights, active_weights_bins, normed=True)
                        ax_kp_histogram.set_title('Keypoint active weights')
                        ax_kp_histogram.set_xlabel('Weight')
                        ax_kp_histogram.set_ylim((0, 1))

                        # Parts density
                        ax_parts_density = axarr[1,4]
                        ax_parts_density.imshow(bbobj.X.mean(axis=-1), interpolation='nearest', vmin=0, vmax=0.5)
                        ax_parts_density.set_title('Parts density')

                        # Now do some data aggregation stuff
                        if bbobj.score0 >= detector.extra['cascade_threshold']:
                            averages_counts[bbobj.mixcomp,bbobj.correct] += 1
                            averages[bbobj.mixcomp][bbobj.correct] += bbobj.X

                            #for f in xrange(detector.num_features):
                                #parts_help[bbobj.correct][f] += 
                            parts_help[bbobj.correct] += parts_weights

                    if svms is not None:
                        svm_weights = svms[bbobj.mixcomp]['weights'].reshape(detector.kernel_templates[bbobj.mixcomp].shape)
                        svm_max_abs = np.fabs(svm_weights).max()

                        # Spatial SVM weights
                        ax_svm_spatial = axarr[2,1]
                        ax_svm_spatial.set_axis_off()
                        active_weights = svm_weights * bbobj.X
                        svm_mean = active_weights.mean(axis=-1)
                        ax_svm_spatial.imshow(svm_mean, interpolation='nearest', vmin=-0.0001, vmax=0.0001, cmap=plt.cm.RdBu_r)
                        ax_svm_spatial.set_title('SVM spatial weights')

                        # Parts SVM weights
                        ax_svm_parts = axarr[2,2]
                        svm_parts_weights = np.apply_over_axes(np.mean, active_weights, [0, 1])[0,0]
                        ax_svm_parts.bar(np.arange(detector.num_features), svm_parts_weights, color=colors[(parts_weights > 0).astype(int)])
                        ax_svm_parts.set_xlabel('Part')
                        #ax_svm_parts.set_ylabel('Weight average')
                        ax_svm_parts.set_title('SVM parts weights')

                        # Active SVM histogram
                        ax_svm_histogram = axarr[2,3]
                        nonzero_weights = active_weights.ravel()[active_weights.ravel() != 0]
                        ax_svm_histogram.hist(nonzero_weights, svm_active_weights_bins, normed=True)
                        ax_svm_histogram.set_title('SVM active weights')
                        ax_svm_histogram.set_xlabel('Weight')
                        ax_svm_histogram.set_ylim((0, 800))

                    # Hide the rest
                    for ax in (axarr[ii] for ii in [(2,0), (1, 4), (2, 4)]):
                        ax.set_axis_off()

                    # Save image
                    fig.savefig(os.path.join(directory, 'base{score:05.02f}-final{final:06.02f}-{mixcomp}.png'.format(score=bbobj.score0, final=bbobj.confidence, mixcomp=bbobj.mixcomp)))
                    plt.close()
                            

                 
        # Get a snapshot of the current precision recall
        detarr = np.array(detections, dtype=[('confidence', float), ('scale', float), ('score0', float), ('score1', float), ('plusscore', float), ('correct', bool), ('mixcomp', int), ('bkgcomp', int), ('img_id', int), ('left', int), ('top', int), ('right', int), ('bottom', int), ('index_pos0', int), ('index_pos1', int)])
        detarr.sort(order='confidence')
        p, r = gv.rescalc.calc_precision_recall(detarr, tot_tp_fn)
        ap = gv.rescalc.calc_ap(p, r) 

        print("{ap:6.02f}% {loop} Testing file {img_id} (tp:{tp} tp+fp:{tp_fp} tp+fn:{tp_fn})".format(loop=loop, ap=100*ap, img_id=img_id, tp=tp, tp_fp=tp_fp, tp_fn=tp_fn))

        # If logging

        #per_file_dets.append(dets)


    if 0:
        from operator import itemgetter
        plt.clf()
        for i, file_dets in enumerate(per_file_dets):
            scores = map(itemgetter(0), file_dets)
            corrects = map(itemgetter(5), file_dets)
            colors = map(lambda x: ['r', 'g'][x], corrects)
            plt.scatter([i+1]*len(file_dets), scores, c=colors, s=50, alpha=0.75)

        plt.savefig('detvis.png')


    detections = np.array(detections, dtype=[('confidence', float), ('scale', float), ('score0', float), ('score1', float), ('plusscore', float), ('correct', bool), ('mixcomp', int), ('bkgcomp', int), ('img_id', int), ('left', int), ('top', int), ('right', int), ('bottom', int), ('index_pos0', int), ('index_pos1', int)])
    detections.sort(order='confidence')


    # Finalize some of the data aggregation and plot them
    if logging:
        parts_help /= averages_counts.mean(axis=0)[:,np.newaxis] + np.finfo(float).eps
        parts_net_help = parts_help[1] - parts_help[0]

        plt.figure(figsize=(20, 10))
        plt.bar(np.arange(detector.num_features), parts_net_help, color=colors[(parts_net_help > 0).astype(int)])
        plt.title('Parts net help')
        plt.xlabel('Part')
        plt.savefig(os.path.join(detections_dir, 'parts-net-help.png'))
        plt.close()

        np.save(os.path.join(detections_dir, 'parts_help.npy'), parts_help)

        plt.figure()
        plt.hist(parts_net_help, 50)
        plt.title('Histogram of net help')
        plt.xlabel('Part')
        plt.savefig(os.path.join(detections_dir, 'hist-net-help.png'))
        plt.close()

        for m in xrange(detector.num_mixtures):
            average = averages[m]
            average /= averages_counts[m,:,np.newaxis,np.newaxis,np.newaxis] + np.finfo(float).eps

            for c, name in enumerate(['neg', 'pos']):
                plt.figure()
                plt.imshow(average[c].mean(axis=-1), interpolation='nearest', vmin=-0.1, vmax=0.1, cmap=plt.cm.RdBu_r)
                plt.savefig(os.path.join(detections_dir, 'average-{0}-{1}.png'.format(m, name)))
                plt.close()


        # Plot score histograms
        from plot_dets import plot_detection_histograms 

        plot_detection_histograms(detections, detector, score_name='score0', 
                                  output_file=os.path.join(detections_dir, 'base-hist.png'))

        if svms is not None:
            good_detections = detections[detections['score0'] > detector.extra['cascade_threshold']]
            svm_detections = detections[detections['confidence'] > 50]
            plot_detection_histograms(good_detections, detector, score_name='score0', 
                                      output_file=os.path.join(detections_dir, 'good-hist.png'))
            plot_detection_histograms(svm_detections, detector, score_name='confidence2', 
                                      output_file=os.path.join(detections_dir, 'svm-hist.png'))


    p, r = gv.rescalc.calc_precision_recall(detections, tot_tp_fn)
    ap = gv.rescalc.calc_ap(p, r) 
    np.savez(output_file, detections=detections, tp_fn=tot_tp_fn, tp_fn_dict=tp_fn_dict, ap=ap, contest=contest, obj_class=obj_class)

    print('tp', tot_tp)
    print('tp+fp', tot_tp_fp)
    print('tp+fn', tot_tp_fn)
    print('----------------')
    #if tot_tp_fp:
    #    print('Precision', tot_tp / tot_tp_fp)
    #if tot_tp_fn:
    #    print('Recall', tot_tp / tot_tp_fn)
    print('AP {0:.2f}% ({1})'.format(100*ap, ap))
