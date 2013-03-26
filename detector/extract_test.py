
import gv
import amitgroup as ag

fileobjs, tot = gv.uiuc.load_testing_files()

i = 0
for fileobj in fileobjs:
    for bbobj in fileobj.boxes:
        bb = bbobj.box 
        im = gv.img.load_image(fileobj.path)

        imp = ag.util.zeropad(im, 50)
        
        im2 = imp[50+bb[0]:50+bb[2], 50+bb[1]:50+bb[3]] 
        print bb
        print im2.shape
        gv.img.save_image(im2, 'test-img/img-{0:03}.png'.format(i))
        i += 1
