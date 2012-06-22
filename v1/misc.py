import pylab as P
from amitgroup.features import bedges 
def plot_image_edges(image):
    edges = bedges(image)
    P.subplot(331)
    p = P.imshow(image, cmap=P.cm.gray)
    p.set_interpolation('nearest')
    for i in range(8):
        P.subplot(332+i)
        p = P.imshow(edges[:,:,i], cmap=P.cm.gray)
        p.set_interpolation('nearest')
    P.show() 


def plot_edges(edges):
    P.subplot(331)
    #p = P.imshow(image, cmap=P.cm.gray)
    #p.set_interpolation('nearest')
    for i in range(8):
        P.subplot(332+i)
        p = P.imshow(edges[:,:,i], cmap=P.cm.gray)
        p.set_interpolation('nearest')
    P.show() 
