#! /usr/bin/env python
from __future__ import division

from PyQt4 import QtGui, QtCore
from PyQt4.QtOpenGL import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import sys
import numpy as np
import gv
import amitgroup as ag

#im = gv.img.load_image('/Users/slimgee/Desktop/pic.png')

def numpy_to_qimage(img):
    assert img.ndim == 2, "We only need grayscale"
    x = (img.flatten()*255).astype(np.uint32)
    x |= (x << 8)
    x |= (x << 16)
    return QtGui.QImage(x, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB32)

def binary_map_to_qimage(img, color):
    assert img.ndim == 2, "We only need grayscale"
    x = (img.flatten()*color[2]).astype(np.uint32)
    x |= ((img.flatten()*color[1]).astype(np.uint32) << 8)
    x |= ((img.flatten()*color[0]).astype(np.uint32) << 16)
    x |= ((img.flatten()*color[3]).astype(np.uint32) << 24)
    return QtGui.QImage(x, img.shape[1], img.shape[0], QtGui.QImage.Format_ARGB32)

def unit_scale_to_qimage(img):
    data = img.flatten()
    x = (np.fabs(data)*255).astype(np.uint32) << 24

    x |= (data > 0) * 0xff0000
    x |= (data < 0) * 0x0000ff
    
    return QtGui.QImage(x, img.shape[1], img.shape[0], QtGui.QImage.Format_ARGB32)

class Viewer3DWidget(QGLWidget):

    def __init__(self, parent, descriptor, detector, img):
        #QGLWidget.__init__(self, parent)
        super(Viewer3DWidget, self).__init__(QGLFormat(QGL.SampleBuffers), parent)
        #self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        # self.setMinimumSize(500, 500)
        self.isPressed = False
        self.oldx = self.oldy = 0
        self.top_bar_height = 120
        self.orig_img = img
        self.descriptor = descriptor
        self._prepareFactor(1.0)
        self.view_edges = False
        self.edge_index = 0
        self.kernel_mode = False

        self.mixture_index = 0
        self.detector = detector
        self.num_mixtures = self.detector.num_mixtures
        self.kern = self.detector.prepare_kernels(None)

    def _prepareFactor(self, f):
        from skimage.transform import pyramid_reduce, pyramid_expand
        img = gv.img.resize_with_factor_new(self.orig_img, 1/f)

        img = gv.img.asgray(img)

        sett = self.descriptor.bedges_settings().copy()
        sett['preserve_size'] = True 
        sett2 = sett.copy()
        sett2['radius'] = 0

        self.unspread_edges = ag.features.bedges(img, **sett2)
        self.edges = ag.features.bedges(img, **sett)
        self.num_edges = self.edges.shape[-1]
        self.partprobs = self.descriptor.extract_partprobs_from_edges(self.edges)
        self.feats = self.partprobs.argmax(axis=-1)
        y = self.partprobs[...,1:]
        self.mn_partprobs = y[y>-np.inf].min()
        self.mx_partprobs = y[y>-np.inf].max()
        self.offset = ((img.shape[0]-self.partprobs.shape[0])//2,
                       (img.shape[1]-self.partprobs.shape[1])//2)
        self.selection = None
        self.im = numpy_to_qimage(img)

    @property
    def psize(self):
        return self.descriptor.patch_size

    @property
    def psize_full(self):
        # The +2 is for edges
        return tuple([self.descriptor.patch_size[i] + (self.descriptor.settings['bedges']['radius']+2)*2 for i in xrange(2)])

    def drawBackground(self, qp):
        qp.setPen(QtGui.QColor(0, 34, 3))
        qp.setBrush(self.gradient)
        #glPushAttrib(GL_ALL_ATTRIB_BITS)
        qp.drawRect(QtCore.QRect(0, 0, 20, 20))
        #glPopAttrib()
        pass

    def paintEvent(self, arg):
        #qp = None 
        qp = QtGui.QPainter(self)
        #qp.begin(self)
        #if qp:
        #self.drawBackground(qp)

        #glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        #self.paintGL()
        #self.drawCube()


        #qp.save()
        #qp.scale(2.0, 2.0)
    

        #qp.drawPixmap(self.rect(), self.im) 
        if self.view_edges:
            edge_im = numpy_to_qimage(self.edges[...,self.edge_index].astype(np.float64)/2 + self.unspread_edges[...,self.edge_index].astype(np.float64)/2)
            qp.drawImage(self.image_rect(), edge_im)
        else:
            qp.drawImage(self.image_rect(), self.im)

        trans = self.image_transform()

        if self.selection is not None and not self.view_edges:

            if self.kernel_mode:
                radii = self.detector.settings['spread_radii']
                psize = self.detector.settings['subsample_size']
                #bkg = self.detector.fixed_spread_bkg
                bkgs = self.detector.bkg_model(None, spread=True)
                bkg = bkgs[self.mixture_index]
                eps = self.detector.settings['min_probability']
            
                ksize = self.kern[self.mixture_index].shape

                kernel_rect = QtCore.QRect(self.selection[0]+self.offset[0], self.selection[1]+self.offset[1], ksize[0] * psize[0], ksize[1] * psize[1])
                qp.setPen(QtGui.QColor(250, 0, 0))
                qp.drawRect(trans.mapRect(kernel_rect))

                #X = self.feats[self.selection[0]:self.selection[0]+self.ksize[0], self.selection[1]:self.selection[1]+self.ksize[1]]
                partprobs = self.partprobs[self.selection[0]:self.selection[0]+ksize[0]*psize[0], self.selection[1]:self.selection[1]+ksize[1]*psize[1]]
                spread_parts = ag.features.spread_patches_new(partprobs.astype(np.float32), radii[0], radii[1], 0.0)
                Y = gv.sub.subsample(spread_parts, psize)
                
                kern = self.kern[self.mixture_index]

    
                kern = np.clip(kern, eps, 1-eps)
                bkg = np.clip(bkg, eps, 1-eps)
                a = np.log(kern / (1 - kern) * ((1-bkg)/bkg))

                print kern.shape, bkg.shape, Y.shape, a.shape
            
                if Y.shape == a.shape:
                    xcorr = (Y * a)
                    llh = xcorr.sum()
                    im = np.clip(xcorr.mean(axis=-1)/0.2, -1.0, 1.0)
                    #print im[10:15,10:15]
                    
                    qim = unit_scale_to_qimage(im)
                    rect = trans.mapRect(kernel_rect)
                    qp.drawImage(rect, qim)

                    # Print log-likelihood
                    qp.setPen(QtGui.QColor(0, 0, 0))
                    qp.setFont(QtGui.QFont('Decorative', 12))
                    Rst = (llh - self.detector.fixed_train_mean[self.mixture_index]) / self.detector.fixed_train_std[self.mixture_index]
                    qp.drawText(20, 120, "{0}".format(Rst))

                # Draw the current mixture component
                if self.detector.support is None:
                    viskern = self.kern[self.mixture_index].sum(axis=-1)
                else:
                    viskern = self.detector.support[self.mixture_index]
                qviskern = numpy_to_qimage(viskern)
                qp.drawImage(QtCore.QRect(10, 10, 100, 100), qviskern)
            
                 
            else:
                pass #print self.selection, self.offset, self.psize

                # Highlight the area of the part, and the total area of influence
                part_rect = QtCore.QRect(self.selection[0]+self.offset[0]-self.psize[0]//2, self.selection[1]+self.offset[1]-self.psize[1]//2, self.psize[0], self.psize[1])
                
                qp.setPen(QtGui.QColor(250, 0, 0))
                qp.drawRect(trans.mapRect(part_rect))

                part_rect_full = QtCore.QRect(self.selection[0]+self.offset[0]-self.psize_full[0]//2, self.selection[1]+self.offset[1]-self.psize_full[1]//2, self.psize_full[0], self.psize_full[1])
                
                qp.setPen(QtGui.QColor(250, 250, 0, 100))
                qp.drawRect(trans.mapRect(part_rect_full))

                # Draw part


                probs  = self.partprobs[self.selection]
                best = probs.argsort()[::-1]
                if best[0] > 0: 
                    qp.setPen(QtGui.QColor(50, 50, 50))
                    qp.setPen(QtGui.QColor(0, 0, 0))
                    qp.setFont(QtGui.QFont('Decorative', 12))
                    for i in xrange(3):
                        visim = self.descriptor.visparts[best[i]-1]
                        mn, mx = visim.min(), visim.max()
                        # TODO: Stretch it out?
                        if self.stretchGrayscale:
                            visim = (visim - mn) / (mx - mn)
                        qim = numpy_to_qimage(visim)
                        qp.drawImage(QtCore.QRect(10 + i * 110, 10, 100, 100), qim)

                        qp.drawText(10 + i * 110+3, 130-3, "{0}".format(best[i]-1))

                        # llh bar
                        score = (probs[best[i]] - self.mn_partprobs) / (self.mx_partprobs - self.mn_partprobs) 

                
                        #bar_rect = QtCore.QRect(10 + i * 110, 120, 100*score, 10)
                        #qp.drawRect(bar_rect)#, QtGui.QBrush(QtGui.QImage(255, 0, 0)))


                    x0 = 10 + 3 * 110 
                    x1 = self.rect().right() - 10 
                    # Draw the llh in descending order
                    llhs = probs[best][:20]
                    path = QtGui.QPainterPath()
                    if x1 > x0 + 30:
                        for i in xrange(len(llhs)):
                            xl = i / (len(llhs))
                            xr = (i+1) / (len(llhs))
                            y = (llhs[i] - self.mn_partprobs) /  (self.mx_partprobs - self.mn_partprobs)
                            pxl = int(x0 + xl * (x1 - x0)) 
                            pxr = int(x0 + xr * (x1 - x0)) - 1
                            py = 10 + 100 - y * 100 
                            if i < 3:
                                c = QtGui.QColor(255, 0, 0)
                            else:
                                c = QtGui.QColor(0, 0, 0)
                            qp.fillRect(QtCore.QRect(pxl, py, pxr-pxl, 110-py), c)

                            if 0:
                                if i == 0:
                                    path.moveTo(px, py)
                                else:
                                    path.lineTo(px, py)
                        #qp.setPen(QtGui.QPen(QtGui.QColor(79, 106, 25), 1, QtCore.Qt.SolidLine, QtCore.Qt.FlatCap, QtCore.Qt.MiterJoin))
                        qp.drawPath(path)

                
                    # Highlight other locations that were coded to this part 
                    fplus = best[0]
                    im = self.feats == fplus
                    im = ag.util.zeropad(im, self.offset) 
                    qim = binary_map_to_qimage(im, (255, 0, 0, 100))
                    qp.drawImage(self.image_rect(), qim)

                

        #self.drawText(qp)
        #if qp:

        if 0:
            qp.setPen(QtGui.QColor(250, 250, 250))
            qp.setFont(QtGui.QFont('Decorative', 16))
            qp.drawText(self.rect(), QtCore.Qt.AlignCenter, "HELLO there")

        
        #qp.end() 
        del qp

    def image_rect(self):
        rfull = self.rect()
        pad = 30
        r = QtCore.QRect(rfull.left() + pad, rfull.top() + self.top_bar_height + pad, rfull.width() - 2*pad, rfull.height() - self.top_bar_height - 2*pad)
        im_size = self.im.size()
        ratio = im_size.width() / im_size.height()
        ratio2 = r.width() / r.height()

        if ratio > ratio2:
            c = r.center().y()
            h = r.width() / ratio
            return QtCore.QRect(r.left(), c - h/2, r.width(), h)
        else:
            c = r.center().x()
            w = r.height() * ratio
            return QtCore.QRect(c - w/2, r.top(), w, r.height())

    def image_transform(self):
        ir = self.image_rect()
        im_size = self.im.size()
        return QtGui.QTransform(0, ir.width() / im_size.width(), ir.height() / im_size.height(), 0, ir.left(), ir.top())

    def point_in_image(self, pos):
        r = self.image_rect()
        return (int(self.im.height() * (pos.y() - r.top())/r.height()),
                int(self.im.width() * (pos.x() - r.left())/r.width()))

    def paintGL(self):
        self.makeCurrent()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()

        self.qglClearColor(QtGui.QColor(50, 50, 0))
        #glClearColor(0.0, 0.0, 0.0, 1.0)
        glClearDepth(1.0)

        glDepthFunc( GL_LEQUAL )
        glEnable( GL_DEPTH_TEST )
        glEnable( GL_CULL_FACE )
        glFrontFace( GL_CCW )
        glDisable( GL_LIGHTING )
        glShadeModel( GL_FLAT )

        # Set
        glViewport(0, 0, 300, 300)
        glEnable(GL_MULTISAMPLE)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        x = 3.0# * self.width() / self.height()
        glOrtho(-x, +x, -3.0, +3.0, -3.0, 3.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        #self.qglClearColor(QtGui.QColor(255, 255, 0))

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        glColor(1.0, 1.0, 1.0)
        glBegin(GL_LINE_STRIP)
        glVertex(-1,-1,-1)
        glVertex( 1,-1,-1)
        glVertex( 1, 1,-1)
        glVertex(-1, 1,-1)
        glVertex(-1,-1, 1)
        glVertex( 1,-1, 1)
        glVertex( 1, 1, 1)
        glVertex(-1, 1, 1)
        glEnd()
        glColor(1.0, 0.0, 0.0)
        glBegin(GL_LINES)
        glVertex( 0, 0, 0)
        glVertex( 1, 0, 0)
        glEnd()
        glColor(0.0, 1.0, 0.0)
        glBegin(GL_LINES)
        glVertex( 0, 0, 0)
        glVertex( 0, 1, 0)
        glEnd()
        glColor(0.0, 0.0, 1.0)
        glBegin(GL_LINES)
        glVertex( 0, 0, 0)
        glVertex( 0, 0, 1)
        glEnd()

        glFlush()

        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def resizeGL(self, widthInPixels, heightInPixels):
        glViewport(0, 0, widthInPixels, heightInPixels)

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClearDepth(1.0)
    
        glEnable(GL_MULTISAMPLE)

        # glMatrixMode(GL_PROJECTION)
        # glLoadIdentity()

#    def mouseMoveEvent(self, mouseEvent):
#        if int(mouseEvent.buttons()) != QtCore.Qt.NoButton :
#            # user is dragging
#            delta_x = mouseEvent.x() - self.oldx
#            delta_y = self.oldy - mouseEvent.y()
#            if int(mouseEvent.buttons()) & QtCore.Qt.LeftButton :
#                if int(mouseEvent.buttons()) & QtCore.Qt.MidButton :
#                    self.camera.dollyCameraForward( 3*(delta_x+delta_y), False )
#                else:
#                    self.camera.orbit(self.oldx,self.oldy,mouseEvent.x(),mouseEvent.y())
#            elif int(mouseEvent.buttons()) & QtCore.Qt.MidButton :
#                self.camera.translateSceneRightAndUp( delta_x, delta_y )
#            self.update()
#        self.oldx = mouseEvent.x()
#        self.oldy = mouseEvent.y()

    def select_point(self, index):
        if self.kernel_mode:
            # Kernel mode
            ii = (index[0]-self.offset[0], index[1]-self.offset[1])
            if ii[0] >= 0 and ii[0] < self.partprobs.shape[0] and \
               ii[1] >= 0 and ii[1] < self.partprobs.shape[1]:
                self.selection = ii 
            else:
                self.selection = None
        else:
            ii = (index[0]-self.offset[0], index[1]-self.offset[1])
            if ii[0] >= 0 and ii[0] < self.partprobs.shape[0] and \
               ii[1] >= 0 and ii[1] < self.partprobs.shape[1]:
                self.selection = ii 
            else:
                self.selection = None

    def mouseDoubleClickEvent(self, mouseEvent):
        pass#print "double click"

    def mousePressEvent(self, e):
        pos = self.point_in_image(e.pos())
        self.select_point(pos)
        self.update()
        self.isPressed = True

    def mouseMoveEvent(self, e):
        pos = self.point_in_image(e.pos())
        self.select_point(pos)
        self.update()

    def keyPressEvent(self, e):
        if self.selection is not None:
            if e.key() == QtCore.Qt.Key_E:
                self.toggleViewEdges()

            if e.key() == QtCore.Qt.Key_R:
                self.setEdgeIndex((self.edge_index + 1) % self.num_edges)

            if e.key() == QtCore.Qt.Key_K:
                self.setMixtureIndex((self.mixture_index + 1) % self.num_mixtures)

            if e.key() == QtCore.Qt.Key_W:
                self.toggleMode()

            sel = self.selection
            if e.key() == QtCore.Qt.Key_Left and sel[1] > 0:
                sel = (sel[0], sel[1]-1)
            elif e.key() == QtCore.Qt.Key_Right and sel[1] < self.partprobs.shape[1]-1:
                sel = (sel[0], sel[1]+1)
            if e.key() == QtCore.Qt.Key_Up and sel[0] > 0:
                sel = (sel[0]-1, sel[1])
            elif e.key() == QtCore.Qt.Key_Down and sel[0] < self.partprobs.shape[0]-1:
                sel = (sel[0]+1, sel[1])

            if sel != self.selection:
                self.selection = sel
                self.update()

    def mouseReleaseEvent(self, e):
        self.isPressed = False

    def setStretchCheck(self, value):
        self.stretchGrayscale = value
        self.update()

    def setFactor(self, f):
        self._prepareFactor(f)
        self.selection = None
        self.update()

    def setEdgeIndex(self, i):
        self.edge_index = i
        self.update()

    def setMixtureIndex(self, i):
        self.mixture_index = i
        self.update()

    def toggleViewEdges(self):
        self.view_edges = not self.view_edges
        self.update()

    def toggleMode(self):
        self.kernel_mode = not self.kernel_mode
        self.selection = None
        self.update()

class PythonQtOpenGLDemo(QtGui.QMainWindow):
    def __init__(self, descriptor, detector, img):
        QtGui.QMainWindow.__init__(self)
        self.setWindowTitle('Parts Viewer')
        self.statusBar().showMessage("Parts Viewer")

        exit = QtGui.QAction("Exit", self)
        exit.setShortcut("Ctrl+Q")
        exit.setStatusTip('Exit application')
        self.connect(exit, QtCore.SIGNAL('triggered()'), QtCore.SLOT('close()'))

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exit)

        self.setToolTip('This is a window, or <b>something</b>')

        viewer3D = Viewer3DWidget(self, descriptor, detector, img)
        self.viewer3D = viewer3D
        createButtons = True
        if createButtons:
            parentWidget = QtGui.QWidget()

            stretchCheck = QtGui.QCheckBox("Stretch grayscale")
            stretchCheck.setStatusTip("Stretch out the parts grayscale")
            self.connect(stretchCheck, QtCore.SIGNAL('clicked()'), self.buttonStretchCheckAction)

            factorText = QtGui.QLineEdit("1.0")
            #factorText.connect(
            self.connect(factorText, QtCore.SIGNAL('returnPressed()'), self.factorTextAction)

            edgeText = QtGui.QLineEdit("0")
            #factorText.connect(
            self.connect(edgeText, QtCore.SIGNAL('returnPressed()'), self.edgeTextAction)

            if 0:
                button1 = QtGui.QPushButton("Button 1")
                button1.setStatusTip('Button 1 does something')
                self.connect(button1, QtCore.SIGNAL('clicked()'), self.button1Action)
                button2 = QtGui.QPushButton("Button 2")
                button2.setToolTip('Button 2 does something else')
                self.connect(button2, QtCore.SIGNAL('clicked()'), self.button2Action)
            vbox = QtGui.QVBoxLayout()
            vbox.addStretch(0)
            vbox.setSizeConstraint(QtGui.QLayout.SetFixedSize)
            vbox.addWidget(stretchCheck)
            vbox.addWidget(factorText)
            if 0:
                vbox.addWidget(edgeText)

            self.stretchCheck = stretchCheck
            self.factorText = factorText
            self.edgeText = edgeText
            if 0:
                vbox.addWidget(button1)
                vbox.addWidget(button2)
            vbox.addStretch(1)
            viewer3D.setSizePolicy( QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding )
            hbox = QtGui.QHBoxLayout()
            hbox.addLayout(vbox)
            hbox.addWidget(viewer3D)

            parentWidget.setLayout(hbox)
            self.setCentralWidget(parentWidget)
        else:
            self.setCentralWidget(viewer3D)

        self.resize(800,500)
        self.buttonStretchCheckAction()

    if 0:
        def closeEvent(self, event):
            reply = QtGui.QMessageBox.question(self, "Confirmation",
                "Are you sure to quit?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
            if reply == QtGui.QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()

    def buttonStretchCheckAction(self):
        self.viewer3D.setStretchCheck(self.stretchCheck.checkState() > 0)

    def factorTextAction(self):
        f = float(self.factorText.text())
        self.viewer3D.setFactor(f)

    def edgeTextAction(self):
        i = int(self.edgeText.text())
        self.viewer3D.setEdgeIndex(i)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Parts diagnostic tool')
    parser.add_argument('parts', metavar='<parts file>', type=argparse.FileType('rb'), help='Parts file')
    parser.add_argument('image', metavar='<image file>', type=argparse.FileType('rb'), help='Image file')
    parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
    args = parser.parse_args()
    # app = QtGui.QApplication(['Python Qt OpenGL Demo'])
    parts_file = args.parts
    img_file = args.image
    model_file = args.model

    descriptor = gv.BinaryDescriptor.getclass('parts').load(parts_file)
    detector = gv.Detector.load(model_file) 
    img = gv.img.load_image(img_file)
    
    app = QtGui.QApplication(sys.argv)
    window = PythonQtOpenGLDemo(descriptor, detector, img)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

