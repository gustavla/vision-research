from __future__ import division, print_function, absolute_import

import os
import collada
import numpy as np

import pyglet
from pyglet.gl import * 
from PIL import Image
from copy import copy

import ctypes

from .shader import Shader

SHADERS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'shaders')

from collections import namedtuple

Camera = namedtuple('Camera', ['fr', 'at', 'up', 'aspect', 'fov', 'near', 'far', 'fixed_up', 'ortho'])


def _scale_matrix(s0, s1, s2):
    return np.matrix([[s0, 0, 0, 0],
                      [ 0,s1, 0, 0],
                      [ 0, 0,s2, 0],
                      [ 0, 0, 0, 1]], dtype=np.float64)

def _translation_matrix(t0, t1, t2):
    return np.matrix([[ 1, 0, 0, t0],
                      [ 0, 1, 0, t1],
                      [ 0, 0, 1, t2],
                      [ 0, 0, 0, 1]], dtype=np.float64)

def _rot_matrix_deg(axis, deg):
    return _rot_matrix(axis, deg / 180 * np.pi)

def _rot_matrix(axis, theta):
    s, c = np.sin(theta), np.cos(theta)
    if axis == 0:
        return np.matrix([[ 1, 0, 0],
                          [ 0, c,-s],
                          [ 0, s, c]])
    elif axis == 1:
        return np.matrix([[ c, 0, s],
                          [ 0, 1, 0],
                          [-s, 0, c]])
    elif axis == 2:
        return np.matrix([[ c,-s, 0],
                          [ s, c, 0],
                          [ 0, 0, 1]])
    else:
        raise ValueError("Invalid axis: {}".format(axis))

def mat3_to_mat4(mat3):
    mat4 = np.asmatrix(np.identity(4))
    mat4[:3,:3] = mat3
    return mat4

def _camera_calc_basis_vectors(camera):
    cz = camera.at - camera.fr
    cz /= np.linalg.norm(cz)

    cx = np.cross(cz, camera.up)
    cx /= np.linalg.norm(cx)

    cy = np.cross(cx, cz)

    return cx, cy, cz

def _camera_view_matrix(camera):
    """Construct a view matrix according to camera"""
    m = np.identity(4)
    view_matrix = np.identity(4)

    #GLfloat b[9], m[16];
    #cameraCalcBasisVectors(camera, b);
    cc = _camera_calc_basis_vectors(camera)

    #print(cc)

    for i in xrange(3):
        for j in xrange(3):
            m[i,j] = cc[i][j]

    #view_matrix = np.dot(_translation_matrix(*-camera.fr), view_matrix)
    view_matrix = _translation_matrix(*(-camera.fr))
    #print('view_matrix before')
    #print(view_matrix)
    view_matrix = np.dot(m, view_matrix)
    return view_matrix

def _camera_proj_matrix(camera):
    """Calculate width/height from fov/aspect"""

    proj_matrix = np.identity(4)

    # Set up projection matrix
    if camera.ortho:
        # Set width/height from the distance to the camera 'at' (as opposed to near plane)
        v = camera.fr - camera.at
        l = np.linalg.norm(v)
        height = 2.0 * l * np.tan(camera.fov / 2.0)
        width = camera.aspect * height

        span = camera.far - camera.near
        proj_matrix[0,0] = 2.0/width
        proj_matrix[1,1] = 2.0/height
        proj_matrix[2,2] = 2.0/span
        proj_matrix[2,3] = -(camera.far + camera.near)/span;
    else:
        #print("PROJ")
        height = 2.0*(camera.near) * np.tan(camera.fov / 2.0)
        width = camera.aspect * height

        span = (camera.far - camera.near)
        proj_matrix[0,0] = 2.0 * camera.near / width
        proj_matrix[1,1] = 2.0 * camera.near / height
        proj_matrix[2,2] = (camera.far + camera.near)/span
        proj_matrix[2,3] = -2.0 * camera.far * camera.near / span
        proj_matrix[3,2] = 1.0
        proj_matrix[3,3] = 0.0

    return proj_matrix

def VecF(*args):
    """Simple function to create ctypes arrays of floats"""
    return (GLfloat * len(args))(*args)

def _get_opengl_version():
    """Get the OpenGL minor and major version number"""
    versionString = glGetString(GL_VERSION)
    return ctypes.cast(versionString, ctypes.c_char_p).value

def _get_opengl_error():
    e = glGetError()
    if e != 0:
        errstr = gluErrorString(e)
        print('GL ERROR:', errstr)
        return errstr
    else:
        return None

def _check_opengl_error():
    ret = _get_opengl_error()
    if ret is not None:
        print(ret)
        import sys
        sys.exit(1)

class Renderer(object):
    def __init__(self, width, height, support='box'):

        # Custom colors
        self.colors = {}

        self.colors['body'] = dict(diffuse=(0.4, 0.2, 0.2, 1.0), env_amount=1.0, env_mult=True)
        self.colors['windows'] = dict(diffuse=(0.0, 0.0, 0.0, 0.8), env_amount=1.0, env_mult=None)

        self._bkg_tex_id = None
        self.cube_map_tex_ids = [] 
        self.cur_cube_map = 0

        self._fov = 45.0
        self._rotate_altitude = 0.0
        self._rotate_azimuth = 0.0
        self._rotate_out_of_plane = 0.0

        self.box_list = []
        self.models = []
        self.cur_model = 0

        self.env_amount = 0.3

        self.background_batch = None
        self.background_sprite = None

        self.extra_text = []

        self.dae_matrix = None
        self.box = None
        self.box_color = None

        self.hide = False

        self._cur_light = 0
        self._cur_light_seed = 0

        self.reset() 
        self.set_size(width, height)
        self.init_gl()
        self.set_projection('perspective')
        self.load_shaders()

        if support == 'box':
            self.load_unitbox()
        elif support == 'ground':
            self.load_support_lines()

        self.load_cube_map_texture('tex/texseam.jpg')
        self.load_cube_map_texture('tex/cube-sky.jpg')
        self.load_cube_map_texture('tex/cube-another.png')
        self.load_cube_map_texture('tex/cube-fullsky.png')
        self.load_cube_map_texture('tex/cube-fullsky2.jpg')

        self._render_shadow = True 

        if self._render_shadow:
            self._init_depth_texture()
        self.set_light(0)

    @property
    def rotate_altitude(self):
        return self._rotate_altitude

    @rotate_altitude.setter
    def rotate_altitude(self, value):
        self._rotate_altitude = np.clip(value, -90, 90)
        return self.rotate_altitude

    @property
    def rotate_azimuth(self):
        return self._rotate_azimuth

    @rotate_azimuth.setter
    def rotate_azimuth(self, value):
        self._rotate_azimuth = (value % 360) 
        return self.rotate_azimuth

    @property
    def rotate_out_of_plane(self):
        return self._rotate_out_of_plane

    @rotate_out_of_plane.setter
    def rotate_out_of_plane(self, value):
        self._rotate_out_of_plane = (value % 360) 
        return self.rotate_out_of_plane

    def set_rotation(self, rotate_altitude, rotate_azimuth, rotate_out_of_plane):
        self.rotate_altitude = rotate_altitude
        self.rotate_azimuth = rotate_azimuth
        self.rotate_out_of_plane = rotate_out_of_plane

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, value):
        self._fov = np.clip(value, 1, 180)
        return self._fov

    @property
    def principal_direction(self):
        """Returns the name of the principal direction"""
        if self.rotate_altitude > 45:
            return 'top'
        elif self.rotate_altitude < -45:
            return 'bottom'
        elif 315 <= self.rotate_azimuth or self.rotate_azimuth < 45:
            return 'front'
        elif 45 <= self.rotate_azimuth < 135:
            return 'left'
        elif 135 <= self.rotate_azimuth < 225:
            return 'back'
        elif 225 <= self.rotate_azimuth < 315:
            return 'right'

    def set_background_image(self, image):
        if self.background_sprite is not None:
            self.background_sprite.delete()

        if self._bkg_tex_id is not None:
            textures = (GLuint * 1)(self._bkg_tex_id)
            glDeleteTextures(1, textures) 

        #img = image 

        if 1:
            img = Image.open(image) 
            try:
                # get image meta-data
                # (dimensions) and data
                (ix, iy, tex_data) = (img.size[0], img.size[1], img.tostring("raw", "RGBA", 0, -1))
            except SystemError:
                # has no alpha channel,
                # synthesize one
                (ix, iy, tex_data) = (img.size[0], img.size[1], img.tostring("raw", "RGBX", 0, -1))
            # generate a texture ID
            tid = GLuint()
            glGenTextures(1, ctypes.byref(tid))
            tex_id = tid.value
            # make it current
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            # copy the texture into the
            # current texture ID
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_data)
            self._bkg_tex_id = tex_id

        #batch = pyglet.graphics.Batch()
        image0 = pyglet.image.load(image)
        self.background_sprite = pyglet.sprite.Sprite(image0)
        #self.background_batch = batch

    @property
    def num_lights(self):
        return 4

    @property
    def cur_light(self):
        return self._cur_light

    @property
    def cur_light_seed(self):
        return self._cur_light_seed

    def update_light(self):
        return self.set_light(self._cur_light, self._cur_light_seed)

    def set_light(self, light_index, light_seed=0):
        glEnable(GL_LIGHT0)
        rs = np.random.RandomState(light_seed)
        self._light_pos = (rs.uniform(-100, 100), rs.uniform(-100, 100), rs.uniform(30, 1000))
        sf = rs.uniform()
        crazy = rs.uniform() < 0.2 
        am = rs.uniform(0.10, 1.25) 
        df = rs.uniform(1.0, 2.0)
        self.env_amount = max(rs.uniform(0.0, np.sqrt(0.5))**2, 0)

        c = rs.uniform()

        if crazy:
            df *= 1.5
            am *= 0.75 
            sf *= 2

        pri = rs.uniform(0.5, 1.0)
        sec = rs.uniform(0.0, 0.5)

        if c <= 0.75:
            self.colors['body'] = dict(diffuse=(0.95, 0.95, 0.95, 1.0))
        elif c <= 0.8:
            self.colors['body'] = dict(diffuse=(sec, sec, pri, 1.0))
        elif c <= 0.85:
            self.colors['body'] = dict(diffuse=(pri, sec, sec, 1.0))
        elif c <= 0.90:
            self.colors['body'] = dict(diffuse=(sec, pri, sec, 1.0))
        else:
            self.colors['body'] = dict(diffuse=(sec, sec, sec, 1.0))
        self.colors['body']['env_amount'] = 0.35 
                
        b = rs.uniform()       
        if b <= 0.4:
            self.colors['windows'] = dict(diffuse=(0.0, 0.0, 0.0, 0.8), env_amount=0.8)
        elif b <= 0.7:
            self.colors['windows'] = dict(diffuse=(1.0, 1.0, 1.0, 0.5), env_amount=0.5)
        else:
            self.colors['windows'] = dict(diffuse=(0.0, 0.0, 0.0, 0.8), env_amount=0.1)

        if not self.cube_map_tex_ids:
            self.cur_cube_map = 0
        else:
            self.cur_cube_map = rs.randint(len(self.cube_map_tex_ids))
        if light_index == 0:
            glLightfv(GL_LIGHT0, GL_AMBIENT, VecF(0.7*am, 0.7*am, 0.7*am, 1.0))
            glLightfv(GL_LIGHT0, GL_DIFFUSE, VecF(0.4*df, 0.4*df, 0.4*df, 1.0))
            glLightfv(GL_LIGHT0, GL_SPECULAR, VecF(1.0*sf, 1.0*sf, 1.0*sf, 1.0))
        elif light_index == 1:
            glLightfv(GL_LIGHT0, GL_AMBIENT, VecF(0.4*am, 0.4*am, 0.4*am, 1.0))
            glLightfv(GL_LIGHT0, GL_DIFFUSE, VecF(0.6*df, 0.6*df, 0.6*df, 1.0))
            glLightfv(GL_LIGHT0, GL_SPECULAR, VecF(0.8*sf, 0.8*sf, 0.8*sf, 1.0))
        elif light_index == 2:
            glLightfv(GL_LIGHT0, GL_AMBIENT, VecF(0.4*am, 0.4*am, 0.4*am, 1.0))
            glLightfv(GL_LIGHT0, GL_DIFFUSE, VecF(0.4*df, 0.4*df, 0.4*df, 1.0))
            glLightfv(GL_LIGHT0, GL_SPECULAR, VecF(0.5*sf, 0.5*sf, 0.5*sf, 1.0))
        elif light_index == 3:
            glLightfv(GL_LIGHT0, GL_AMBIENT, VecF(0.3*am, 0.3*am, 0.3*am, 1.0))
            glLightfv(GL_LIGHT0, GL_DIFFUSE, VecF(0.3*df, 0.3*df, 0.3*df, 1.0))
            glLightfv(GL_LIGHT0, GL_SPECULAR, VecF(0.3*sf, 0.3*sf, 0.3*sf, 1.0))
        self._cur_light = light_index
        self._cur_light_seed = light_seed

        # TODO: This is an ugly fix
        #self._render_shadow_texture(np.asarray(self._light_pos)[[0,2,1]])
        if self._render_shadow:
            blur = max(rs.uniform(-5.0, 50.0), 0)
            opacity = np.clip(rs.uniform(0, 1.5), 0, 0.9)
            exponent = 1.0

            """
            if rs.randint(2) == 0:
                # This make the shadow smaller, which in many cases
                # is more realistic under sunlight
                blur *= rs.randint(2, 5) 
                exponent = rs.randint(3, 8) 
            """

            #print('blur', blur)
            #print('opacity', opacity)

            #blur = 20.0
            #opacity = 1.0
            #blur = 40.0
            self._render_shadow_texture(np.array([0.0, 5.0, 0.0]), blur=blur, opacity=opacity, exponent=exponent)

            #import pdb; pdb.set_trace()

            self._shadow_x = -self._light_pos[0] / 1500.0
            self._shadow_z = -self._light_pos[1] / 1500.0

    def next_light(self):
        self.set_light((self._cur_light + 1) % self.num_lights)

    def _load_shader(self, name):
        with open(os.path.join(SHADERS_DIR, '{}.vert'.format(name))) as f:
            vert = f.read()
        with open(os.path.join(SHADERS_DIR, '{}.frag'.format(name))) as f:
            frag = f.read()
#
        return Shader([vert], [frag])

    def load_shaders(self):
        self.shaders = {}

        #self.shaders['lambert'] = self._load_shader('shader')
        self.shaders['texture'] = self._load_shader('texture')
        self.shaders['texture-simple'] = self._load_shader('texture-simple')
        self.shaders['flat'] = self._load_shader('flat') 
        self.shaders['depth'] = self._load_shader('depth') 

    def reset(self):
        self.offset_2d = np.zeros(3)
        self.scale_2d = 1.0
        self.batch_list = []
        self.textures = {}

    def transform_dae(self, matrix):
        if self.dae is not None:
            M = np.asmatrix(matrix)
            for node in self.dae.scene.nodes:
                M2 = (M * node.matrix).astype(np.float32)
                node.transforms[:] = [collada.scene.MatrixTransform(np.asarray(M2).flatten())]
            self.dae.save()

    def load_unitbox(self):
        for f, diff_color in [(0.8, (1, 1, 0, 0.8)), (1.0, (1, 0, 0, 0.3))]:
            batch = pyglet.graphics.Batch()
            indices = np.array([
                0, 1, 
                1, 2,
                2, 3,
                3, 0,
                4, 5, 
                5, 6,
                6, 7,
                7, 4,
                0, 4,
                1, 5,
                2, 6,
                3, 7,])

            vertices = np.array([0.0, 0.0, 0.0,
                                 1.0, 0.0, 0.0,
                                 1.0, 1.0, 0.0,
                                 0.0, 1.0, 0.0,
                                 0.0, 0.0, 1.0,
                                 1.0, 0.0, 1.0,
                                 1.0, 1.0, 1.0,
                                 0.0, 1.0, 1.0])

            vertices -= 0.5
            vertices *= 2.0

            vertices *= f

            batch.add_indexed(int(indices.max()+1), 
                              GL_LINES,
                              None,
                              indices,
                              ('v3f/static', vertices)) 

            shader_prog = self.shaders['flat']

            self.box_list.append(
                (batch, shader_prog, None, '', dict(diffuse=diff_color)))

        # Do cross hairs
        for f, diff_color in [(1.5, (1, 0, 1, 0.8))]:
            batch = pyglet.graphics.Batch()
            indices = np.array([
                0, 1, 
                2, 3,
                4, 5])

            vertices = np.array([-1.0, 0.0, 0.0,
                                  1.0, 0.0, 0.0,
                                  0.0,-1.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0,-1.0,
                                  0.0, 0.0, 1.0])

            #vertices -= 0.5
            #vertices *= 2.0

            vertices *= f

            batch.add_indexed(int(indices.max()+1), 
                              GL_LINES,
                              None,
                              indices,
                              ('v3f/static', vertices)) 

            shader_prog = self.shaders['flat']

            self.box_list.append(
                (batch, shader_prog, None, '', dict(diffuse=diff_color)))

        for side in [-1, 1]:
            diff_color = (1, 1, 1, 1)
            batch = pyglet.graphics.Batch()
            indices = np.array([
                0, 1, 
                2, 3])

            py = side * 0.5
            pz = -0.13

            s = 0.01

            vertices = np.array([1.0, py-s, pz-s,
                                 1.0, py+s, pz-s,
                                 1.0, py+s, pz+s,
                                 1.0, py-s, pz+s])

            #vertices -= 0.5
            #vertices *= 2.0

            batch.add_indexed(int(indices.max()+1), 
                              GL_QUADS,
                              None,
                              indices,
                              ('v3f/static', vertices)) 

            shader_prog = self.shaders['flat']

            self.box_list.append(
                (batch, shader_prog, None, '', dict(diffuse=diff_color)))
             
    def load_support_lines(self):
        # Do cross hairs
        for flip, diff_color in [(1, (1, 1, 0, 0.8)), (-1, (1, 1, 0, 0.8))]:
            batch = pyglet.graphics.Batch()
            indices = np.array([
                0, 1, 
                2, 3, 
                4, 5])

            ss = 0.3 * flip
            hh = -0.2
            ll = 20.0

            vertices = np.array([ -ll,  ss, hh,
                                   ll,  ss, hh,
                                   ss, -ll, hh,
                                   ss,  ll, hh,
                                  0.0, 0.0,-ll,
                                  0.0, 0.0, ll])

            #vertices -= 0.5
            #vertices *= 2.0

            batch.add_indexed(int(indices.max()+1), 
                              GL_LINES,
                              None,
                              indices,
                              ('v3f/static', vertices)) 

            shader_prog = self.shaders['flat']

            self.box_list.append(
                (batch, shader_prog, None, '', dict(diffuse=diff_color)))


    def save_models(self):
        old_cur = self.cur_model
        for i, model in enumerate(self.models):
            self.set_model(i)
            #self.update_model()
            if 'filename' not in model:
                print('ERROR: Model read only')
            else:
                print('Saving', model['filename'])
                self.dae.write(model['filename'])
        print('Done.')

        self.set_model(old_cur)

    def update_model(self):
        """Incorporate working translation/scale into the model"""
    

        #for geom in self.dae.scene.objects('geometry'): 
            #geom.matrix[:] *= self.dae_matrix

        #print('updating here')
        #self.transform_dae(self.dae_matrix)
        #self.dae_matrix[:] = np.eye(4)

        #self.reload_dae()
        #print('dae M', self.dae_matrix)
        #self.transform_dae(_scale_matrix(0.005, 0.005, 0.020))
        #self.transform_dae(_scale_matrix(0.005, 0.005, 0.020))
        if self.num_models > 0 and not (self.dae_matrix == np.identity(4)).all():
            model = self.models[self.cur_model]
            model['batch_list'] = self.load_dae(model['dae'], transform=self.dae_matrix.copy())
        
        #for geom in 

        #glMultMatrixf(VecF(*self.dae_matrix.T.flatten()))


    def reload_dae(self):
        #self.reset()
        self.models[self.cur_model]['batch_list'] = self.load_dae(self.dae)
    
    def get_translation(self, index):
        if self.dae_matrix is None:
            return 0.0
        else:
            return self.dae_matrix[index,3]

    def set_translation(self, index, value):
        if self.dae_matrix is not None:
            self.dae_matrix[index,3] = value

    def get_offset_2d(self, index):
        return self.offset_2d[index] 

    def set_offset_2d(self, index, value):
        self.offset_2d[index] = value

    def get_scale_2d(self):
        return self.scale_2d

    def set_scale_2d(self, value):
        self.scale_2d = value

    def get_scale(self, index):
        if self.dae_matrix is None:
            return 1.0
        else:
            return self.dae_matrix[index,index]

    def set_scale(self, index, value):
        if self.dae_matrix is not None:
            self.dae_matrix[index,index] = value

    def set_dae_files(self, files, read_only=False):    
        self.collada_files = []
        self.models = []
        for fn in files:
            print('Loading model', fn)
            collada_file = collada.Collada(fn, ignore=[collada.DaeUnsupportedError,
                                                       collada.DaeBrokenRefError])

            #self.collada_files.append(collada_file)

            batch_list = self.load_dae(collada_file)
            info = dict(batch_list=batch_list)
            info['name'] = os.path.splitext(os.path.basename(fn))[0]
            if not read_only:
                info['filename'] = fn
                info['dae'] = collada_file
            self.models.append(info)

        self.set_model(0)

    @property
    def cur_model_name(self):
        if self.models:
            return self.models[self.cur_model]['name'] 
        else:
            return '(no model)'

    @property
    def num_models(self):
        return len(self.models)

    def set_model(self, index):
        self.update_model()

        self.cur_model = index
        if index >= len(self.models):
            index = 0

        if 'dae' in self.models[index]:
            self.dae = self.models[index]['dae']

        self.batch_list = self.models[index]['batch_list']

        self.update_light()

    def set_model_by_name(self, name):
        for i, model in enumerate(self.models):
            if name == model['name']: 
                self.cur_model = i
                break

    def next_model(self):
        self.set_model((self.cur_model + 1) % self.num_models)

    def prev_model(self):
        self.set_model((self.cur_model - 1) % self.num_models)

    def load_dae(self, dae, transform=None):

        #import pdb; pdb.set_trace()

        batch_list = []
        self.dae = dae


        if transform is None:
            transform = np.eye(4)
        transform = np.asmatrix(transform)

        self.transform_dae(transform)

        self.dae_matrix = np.eye(4)

        y_mn = np.inf

        if self.dae.scene is not None:
            for geom in self.dae.scene.objects('geometry'):
                for prim in geom.primitives():
                    colors = {}
                    #print('PRIM', prim.material.name)
                    mat = prim.material
                    diff_color = VecF(0.3, 0.3, 1.0, 1.0)
                    spec_color = None 
                    shininess = None
                    amb_color = None
                    tex_id = None
                    #print 'mat.effect.shadingtype', mat.effect.shadingtype
                    #shader_prog = self.shaders[mat.effect.shadingtype]
                    #shader_prog = self.shaders['lambert']
                    shader_prog = self.shaders['texture']
                    for prop in mat.effect.supported:
                        value = getattr(mat.effect, prop)
                        # it can be a float, a color (tuple) or a Map
                        # ( a texture )
                        if isinstance(value, collada.material.Map):
                            colladaimage = value.sampler.surface.image
                            #print 'image', colladaimage

                            # Accessing this attribute forces the
                            # loading of the image using PIL if
                            # available. Unless it is already loaded.
                            img = colladaimage.pilimage
                            if img: # can read and PIL available
                                #shader_prog = self.shaders['texture']
                                # See if we already have texture for this image
                                if self.textures.has_key(colladaimage.id):
                                    tex_id = self.textures[colladaimage.id]
                                else:
                                    # If not - create new texture
                                    try:
                                        # get image meta-data
                                        # (dimensions) and data
                                        (ix, iy, tex_data) = (img.size[0], img.size[1], img.tostring("raw", "RGBA", 0, -1))
                                    except SystemError:
                                        # has no alpha channel,
                                        # synthesize one
                                        (ix, iy, tex_data) = (img.size[0], img.size[1], img.tostring("raw", "RGBX", 0, -1))
                                    # generate a texture ID
                                    tid = GLuint()
                                    glGenTextures(1, ctypes.byref(tid))
                                    tex_id = tid.value
                                    # make it current
                                    glBindTexture(GL_TEXTURE_2D, tex_id)
                                    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE)
                                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
                                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                                    # copy the texture into the
                                    # current texture ID
                                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_data)

                                    self.textures[colladaimage.id] = tex_id
                            else:
                                print('  {0} = Texture {1}: (not available)'.format(
                                    prop, colladaimage.id))
                        else:
                            if prop == 'diffuse' and value is not None:
                                colors['diffuse'] = value 
                            elif prop == 'specular' and value is not None:
                                colors['specular'] = value
                            elif prop == 'ambient' and value is not None:
                                colors['ambient'] = value
                            elif prop == 'shininess' and value is not None:
                                colors['shininess'] = value

                    # use primitive-specific ways to get triangles
                    prim_type = type(prim).__name__
                    if prim_type == 'BoundTriangleSet':
                        triangles = prim
                    elif prim_type == 'BoundPolylist':
                        triangles = prim.triangleset()
                    else:
                        triangles = None

                    if triangles is not None:
                        triangles.generateNormals()
                        # We will need flat lists for VBO (batch) initialization

                        if 0:
                            v4 = np.concatenate([triangles.vertex, np.ones((triangles.vertex.shape[0], 1))], axis=1)


                            M = np.matrix([[1, 0, 0, 50.0],
                                           [0, 1, 0, 50.0],
                                           [0, 0, 1, 50.0],
                                           [0, 0, 0, 1]])

                            vv = (transform * v4.T).T
                            triangles.vertex[:] = np.asarray(vv[:,:3]) / np.asarray(vv[:,[3]])
                            #print(triangles.vertex[0])


                        vertices = (triangles.vertex[:,[0,2,1]] * np.array([[1, 1, -1]])).flatten().tolist()
                        y_mn = min(triangles.vertex[:,2].min(), y_mn)
                        #print('vertices', triangles.vertex[:,2].min())
                        batch_len = len(vertices)//3
                        indices = triangles.vertex_index.flatten().tolist()
                        normals = triangles.normal.flatten().tolist()

                        batch = pyglet.graphics.Batch()

                        # Track maximum and minimum Z coordinates
                        # (every third element) in the flattened
                        # vertex list

                        if tex_id is not None:
                            #print 'TEXTURE'

                            # This is probably the most inefficient
                            # way to get correct texture coordinate
                            # list (uv). I am sure that I just do not
                            # understand enough how texture
                            # coordinates and corresponding indexes
                            # are related to the vertices and vertex
                            # indicies here, but this is what I found
                            # to work. Feel free to improve the way
                            # texture coordinates (uv) are collected
                            # for batch.add_indexed() invocation.
                            uv = [[0.0, 0.0]] * batch_len
                            for t in triangles:
                                nidx = 0
                                texcoords = t.texcoords[0]
                                for vidx in t.indices:
                                    uv[vidx] = texcoords[nidx].tolist()
                                    nidx += 1
                            # Flatten the uv list
                            uv = [item for sublist in uv for item in sublist]

                            # Create textured batch
                            batch.add_indexed(batch_len, 
                                              GL_TRIANGLES,
                                              None,
                                              indices,
                                              ('v3f/static', vertices),
                                              ('n3f/static', normals),
                                              ('t2f/static', uv))
                        else:
                            #print 'creating colored batch'
                            #print batch_len
                            #print type(indices), indices
                            #print type(indices), vertices
                            #print normals

                            # Create colored batch
                            batch.add_indexed(batch_len, 
                                              GL_TRIANGLES,
                                              None,
                                              indices,
                                              ('v3f/static', vertices),#list(np.multiply(2.0, vertices))),
                                              ('n3f/static', normals))

                        # TODO: Temporary
                        #if diff_color[3] != 1.0:    
                            #print('diff_color', diff_color)


                        name = prim.material.name
                        #if n in self.colors:
                            #colors = self.colors[n]
                            #colors['diffuse'] = self.colors[n].get('diffuse')
                            #colors['env_amount'] = self.colors[n].get('env_amount')

                        # Append the batch with supplimentary
                        # information to the batch list
                        batch_list.append(
                            (batch, shader_prog, tex_id, name, colors))

        self._ground_y = y_mn                    
        print('y_mn', y_mn) 
        return batch_list                        


    def load_cube_map_texture(self, img_path):
        tid = GLuint()
        glGenTextures(1, ctypes.byref(tid))
        tex_id = tid.value
        self.cube_map_tex_ids.append(tex_id)
        glBindTexture(GL_TEXTURE_CUBE_MAP, tex_id)

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

        img = Image.open(img_path) 
        try:
            # get image meta-data
            # (dimensions) and data
            (ix, iy, tex_data) = (img.size[0], img.size[1], img.tostring("raw", "RGBA", 0, -1))
        except SystemError:
            # has no alpha channel,
            # synthesize one
            (ix, iy, tex_data) = (img.size[0], img.size[1], img.tostring("raw", "RGBX", 0, -1))

        offsets = [
            [ 2, 1 ],
            [ 0, 1 ],
            [ 1, 2 ],
            [ 1, 0 ],
            [ 1, 1 ],
            [ 3, 1 ],
        ]

        # Extract images

        side = img.size[0] // 4

        faceData = []
        for offset in offsets:
            bb = (offset[0]*side, offset[1]*side, (offset[0]+1)*side, (offset[1]+1)*side)
            img0 = img.crop(bb)
            faceData.append(img0)

        for i in xrange(len(faceData)):
            img_side = faceData[i]
            try:
                # get image meta-data
                # (dimensions) and data
                (ix, iy, tex_data) = (img_side.size[0], img_side.size[1], img_side.tostring("raw", "RGBA", 0, -1))
            except SystemError:
                # has no alpha channel,
                # synthesize one
                (ix, iy, tex_data) = (img_side.size[0], img_side.size[1], img_side.tostring("raw", "RGBX", 0, -1))

            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X+i, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_data)

        glGenerateMipmap(GL_TEXTURE_CUBE_MAP)
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
        

    def load_test_scene(self):
        self.dae = None
        self.dae_matrix = None
        batch_list = []

        from .sphere import xyz, norm, tex2, indices, gl_type

        batch = pyglet.graphics.Batch()

        batch.add_indexed(int(np.max(indices)+1),
                          gl_type[0],
                          None,
                          indices,
                          ('v3f/static', xyz),
                          ('n3f/static', norm),
                          ('t2f/static', tex2))


        shader_prog = self.shaders['texture']

        diff_color = (1.0, 0.0, 1.0, 1.0)

        batch_list.append(
            (batch, shader_prog, None, dict(diffuse=diff_color)))

        self.models = [dict(batch_list=batch_list, name='test scene')] 
        self.cur_model = 0
        return batch_list

    @property
    def projection(self):
        return self._proj_type

    def toggle_projection(self):
        x = ['perspective', 'ortho']
        self.set_projection(x[(x.index(self.projection)+1)%len(x)]) 

    def set_projection(self, proj_type):
        self._proj_type = proj_type

    def set_size(self, width, height):
        if height==0: 
            height=1
        self.width = width
        self.height = height
        # Override the default on_resize handler to create a 3D projection

    def init_gl(self):
        print('Running with OpenGL version:', _get_opengl_version())
        print('Initializing shaders...')

        # White transparent background
        glClearColor(0.5, 0.7, 1.0, 0.0) 
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_MULTISAMPLE)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_LIGHTING)

        glEnable(GL_TEXTURE_2D)


    def _init_depth_texture(self):
        self._depth_tex_size = 512 
        # Setup depth texture
        fb_id = GLuint(0)
        glGenFramebuffers(1, ctypes.byref(fb_id))
        self._depth_fb_id = fb_id.value
        glBindFramebuffer(GL_FRAMEBUFFER, self._depth_fb_id)

        # Attach color channel 
        color_tid = GLuint(0)
        if 0:
            glGenTextures(1, ctypes.byref(color_tid))
            color_tex_id = color_tid.value
            glBindTexture(GL_TEXTURE_2D, color_tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            #glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size, size, 0, GL_RGBA, GL_FLOAT, 0)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self._depth_tex_size, self._depth_tex_size, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0)
            #glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   #GL_TEXTURE_2D, color_tex_id, 0)
            glBindTexture(GL_TEXTURE_2D, 0)

            draw_buffers = (GLenum * 1)(GL_COLOR_ATTACHMENT0)

        depth_tex_id = 0 
        depth_tid = GLuint(0)
        if 0:
            glGenTextures(1, ctypes.byref(depth_tid))
            depth_tex_id = depth_tid.value
            glBindTexture(GL_TEXTURE_2D, depth_tex_id)

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 8)
            glGenerateMipmap(GL_TEXTURE_2D)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, self._depth_tex_size, self._depth_tex_size, 
                         0, GL_DEPTH_COMPONENT, GL_FLOAT, 0)

            #GLuint depthrenderbuffer
            if 0:
                depth_render_tid = GLuint(0)
                glGenRenderbuffers(1, ctypes.byref(depth_render_tid))
                glBindRenderbuffer(GL_RENDERBUFFER, depth_render_tid.value)
                glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, size, size)
                glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_render_tid.value)

                glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, depth_tex_id, 0)

                glBindRenderbuffer(GL_RENDERBUFFER, 0)
            

            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                   GL_TEXTURE_2D, depth_tex_id, 0)

            glBindTexture(GL_TEXTURE_2D, 0)
        elif 1:
            img = Image.open('tex/tex2.png') 
            try:
                # get image meta-data
                # (dimensions) and data
                (ix, iy, tex_data) = (img.size[0], img.size[1], img.tostring("raw", "RGBA", 0, -1))
            except SystemError:
                print("FALLING BACK")
                # has no alpha channel,
                # synthesize one
                (ix, iy, tex_data) = (img.size[0], img.size[1], img.tostring("raw", "RGBX", 0, -1))
            # generate a texture ID
            glGenTextures(1, ctypes.byref(depth_tid))
            depth_tex_id = depth_tid.value
            color_tex_id = depth_tex_id
            # make it current
            glBindTexture(GL_TEXTURE_2D, depth_tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            #glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
            #glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
            # copy the texture into the
            # current texture ID
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_data)
            glBindTexture(GL_TEXTURE_2D, 0)


        if 0:
            depth_render_tid = GLuint(0)
            glGenRenderbuffers(1, ctypes.byref(depth_render_tid))
            glBindRenderbuffer(GL_RENDERBUFFER, depth_render_tid.value)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, size, size)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_render_tid.value)
            self._depth_render_id = depth_render_tid.value

            glBindRenderbuffer(GL_RENDERBUFFER, 0)

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, color_tex_id, 0)


        _check_opengl_error()
        self._depth_tex_id = depth_tex_id
        self._color_tex_id = color_tex_id

        # Only draw depth buffer for this FBO
        #glDrawBuffer(GL_NONE)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print("ERROR: Depth buffer")

    def draw_background(self):
        if self.background_sprite:
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            a = self.width/self.height
            glDisable(GL_DEPTH_TEST)
            #glOrtho(-2*a, 2*a, -2, 2, -100, 100)
            glOrtho(0, self.width, 0, self.height, -1, 1) 
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glTranslatef(self.width/2 - self.background_sprite.width/2, self.height/2 - self.background_sprite.height/2, 0.0)

            glDisable(GL_LIGHTING)
            glEnable(GL_TEXTURE_2D)
            shader_prog = self.shaders['texture-simple']

            #glUseProgram(0)
            shader_prog.bind()
            #shader_prog.uniformf('diffuse', 1.0, 1.0, 1.0, 1.0)
            #shader_prog.uniformf('textureAmount', 1.0)
            #shader_prog.uniformf('envAmount', 0.0) 
            #glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, VecF(1.0, 1.0, 1.0, 1.0))
            #glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, VecF(0.0, 0.0, 0.0, 1.0))
            #glLightfv(GL_LIGHT0, GL_AMBIENT, VecF(1.0, 1.0, 1.0, 1.0))
            #glLightfv(GL_LIGHT0, GL_DIFFUSE, VecF(0.0, 0.0, 0.0, 1.0))
            #glLightfv(GL_LIGHT0, GL_SPECULAR, VecF(0.0, 0.0, 0.0, 1.0))

            #self._bind_cube_stuff(shader_prog, offset=1)

            #shader_prog.uniformf('textureAmount', 1.0)
            # We assume that the shader here is 'texture'
            tex_id = self._bkg_tex_id
            #tex_id = self.background_sprite._group.texture.id
            if 1:
                glActiveTexture(GL_TEXTURE0)
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, tex_id)
                shader_prog.uniformi('myTexture', 0)

            self.background_sprite.draw()
            #self.background_sprite._vertex_list.draw(GL_QUADS)

            if 0:
                pyglet.graphics.draw_indexed(4, pyglet.gl.GL_TRIANGLES,
                    [0, 1, 2, 0, 2, 3],
                    ('v2i', (100, 100,
                             150, 100,
                             150, 150,
                             100, 150)),
                    ('t2f', (0.0, 0.0,
                             1.0, 0.0,
                             1.0, 1.0,
                             0.0, 1.0))
                )

            shader_prog.unbind()


    def _bind_cube_stuff(self, shader_prog, offset=1):
        glActiveTexture(GL_TEXTURE0+offset)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.cube_map_tex_ids[self.cur_cube_map])
        shader_prog.uniformi('cubemapTexture', offset)
        #_check_opengl_error()

        shader_prog.uniformf('cameraDir', *self.camera_dir) 

        #_check_opengl_error()

    def update_colors(self, name, colors):
        new_colors = copy(colors)
        if name in self.colors:
            new_colors.update(self.colors[name])
        return new_colors

    def _render_objects(self, use_shader_prog=True): 
        prev_shader_prog = None
        if self.models:
            for batch, shader_prog, tex_id, name, colors in self.models[self.cur_model]['batch_list']:
                colors = self.update_colors(name, colors)
                diff_color = colors.get('diffuse')
                # Optimization to not make unnecessary bind/unbind for the
                # shader. Most of the time there will be same shaders for
                # geometries.
                if shader_prog != prev_shader_prog and use_shader_prog:
                    if prev_shader_prog is not None:
                        prev_shader_prog.unbind()
                    prev_shader_prog = shader_prog
                    shader_prog.bind()

                if self.hide and diff_color is not None:
                    diff_color = (diff_color[0], diff_color[1], diff_color[2], diff_color[3] * 0.2)

                if use_shader_prog:
                    #print 'colors', diff_color, spec_color, amb_color, shininess
                    if diff_color is not None:
                        #print('diff_color', diff_color)
                        #diff_color = (0, 0, 1, 1)
                        shader_prog.uniformf('diffuse', *diff_color)
                        #glUniform4f(glGetUniformLocation(shader_prog.handle, 'diffuse'), *diff_color)
                        #_get_opengl_error()
                        pass
                    if 'specular' in colors: 
                        shader_prog.uniformf('specular', *colors['specular'])
                        raise Exception('specular')
                    if 'ambient' in colors:
                        shader_prog.uniformf('ambient', *colors['ambient'])
                        raise Exception('ambient')
                    if 'shininess' in colors:
                        shader_prog.uniformf('shininess', colors['shininess'])
                        raise Exception('shininess')

                    if diff_color is not None:
                        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, VecF(*diff_color))
                    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, VecF(0.5, 0.5, 0.5, 1.0))
                    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 150)


                    if tex_id is not None:
                        shader_prog.uniformf('textureAmount', 1.0)
                        # We assume that the shader here is 'texture'
                        glActiveTexture(GL_TEXTURE0)
                        glEnable(GL_TEXTURE_2D)
                        #tex_id = self._bkg_tex_id
                        glBindTexture(GL_TEXTURE_2D, tex_id)
                        shader_prog.uniformi('myTexture', 0)
                    else:
                        shader_prog.uniformf('textureAmount', 0.0)

                    if 'env_amount' in colors:
                        amount = colors['env_amount']
                        if colors.get('env_mult'):
                             amount *= self.env_amount
                        shader_prog.uniformf('envAmount', amount)
                    else:
                        #shader_prog.uniformf('envAmount', self.env_amount) 
                        shader_prog.uniformf('envAmount', 0.0) 

                    self._bind_cube_stuff(shader_prog)


                batch.draw()
            if prev_shader_prog is not None and use_shader_prog:
                prev_shader_prog.unbind()


    def _render_shadow_texture(self, light_pos, blur=0.0, exponent=1.0, opacity=1.0):
        glViewport(0, 0, self._depth_tex_size, self._depth_tex_size) 
        glClearDepth(1.0)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        #glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        fov = 45.0
        gluPerspective(fov, 1.0, 0.1, 10000.0)

        light_pos = light_pos * 2.3 / np.linalg.norm(light_pos)

        camera = Camera(fr=light_pos,
                        at=np.array([0.0, 0.0, 0.0]),
                        up=np.array([1.0, 0.0, 0.0]),
                        aspect=1.0,
                        fov=fov,
                        near=0.01,
                        far=10.0,
                        fixed_up=False,
                        ortho=False)

        view_matrix = _camera_view_matrix(camera)
        proj_matrix = _camera_proj_matrix(camera)

        def reflect(I, N):
            return I - 2 * np.dot(N, I) * N

        np.set_printoptions(precision=2, suppress=True)

        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(VecF(*np.asarray(proj_matrix).astype(np.float32).T.flatten()))

        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(VecF(*np.asarray(view_matrix).astype(np.float32).T.flatten()))

        shader_prog = self.shaders['depth']
        shader_prog.bind()
        self._render_objects(use_shader_prog=False)
        shader_prog.unbind()


        # Render it to texture
        glBindTexture(GL_TEXTURE_2D, self._color_tex_id)
        glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 0, 0, self._depth_tex_size, self._depth_tex_size, 0);


        # Now blur the texture
        if blur > 0.0:
            glBindTexture(GL_TEXTURE_2D, self._color_tex_id)

            data = (ctypes.c_ubyte * (self._depth_tex_size * self._depth_tex_size * 4))()
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)

            sh = (self._depth_tex_size, self._depth_tex_size, 4)
            X = np.ctypeslib.as_array(data, shape=sh).reshape(sh)

            print('X', X.shape)
            #grayX = X[:,:,[0,3]]
            #alpha = X[:,:,3]

            #X[:] = 255 

            from skimage.filter import gaussian_filter

            alpha = X[...,3].astype(np.float64) / 255.0
            alpha[:] = gaussian_filter(alpha, blur)

            alpha[:] **= exponent

            X[...,[0,1,2]] = 0
            X[...,3] = (opacity * alpha * 255.0).astype(np.uint8)

            X[...,:3] = 0 

            # Now blur this S.O.B.

            new_data = np.ctypeslib.as_ctypes(X)
            
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self._depth_tex_size, self._depth_tex_size, 0, GL_RGBA, GL_UNSIGNED_BYTE, new_data)
            glBindTexture(GL_TEXTURE_2D, 0)

            #return 

    def render(self, osd=True, support_rendering=False):
        """Render batches created during class initialization"""

        # Render shadow texture once only
        #self._render_shadow_texture()

        def degsin(x):
            return np.sin(x * np.pi / 180)
        def degcos(x):
            return np.cos(x * np.pi / 180)

        if support_rendering:
            glClearColor(1.0, 1.0, 1.0, 1.0)
        else:
            glClearColor(0.5, 0.7, 1.0, 0.0) 
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glViewport(0, 0, self.width, self.height)

        self.camera_dir = np.array([1.0, 0.0, 0.0])

        if not support_rendering:
            self.draw_background()

        if 1:
            if self._proj_type == 'perspective':
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                # Translate across 2D
                #glTranslatef(*self.offset_2d)
                glTranslatef(self.offset_2d[0], self.offset_2d[1], 0.0)
                gluPerspective(self.fov, self.width / float(self.height), 0.1, 10000000.)

            elif self._proj_type == 'ortho':
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                a = self.width/self.height
                glOrtho(-2*a, 2*a, -2, 2, -100, 100)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glEnable(GL_DEPTH_TEST)

        z_offset = 4 / (2 * np.tan((self.fov/2) * np.pi/180))
        camera = Camera(fr=np.array([z_offset*degsin(-self.rotate_azimuth)*degcos(self.rotate_altitude), 
                                     z_offset*degsin(self.rotate_altitude), 
                                     z_offset*degcos(-self.rotate_azimuth)*degcos(self.rotate_altitude)]),
                        at=np.array([0.0, 0.0, 0.0]),
                        up=np.array([0.0, 1.0, 0.0]),
                        aspect=self.width/self.height,
                        fov=self.fov * np.pi / 180,
                        near=0.01,
                        far=100.0,
                        fixed_up=False,
                        ortho=False)

        view_matrix = mat3_to_mat4(_rot_matrix_deg(2, self.rotate_out_of_plane)) * _camera_view_matrix(camera)

        TM = _translation_matrix(self.offset_2d[0], self.offset_2d[1], 0.0)
        proj_matrix = TM * _camera_proj_matrix(camera)

        #print('view matrix')
        #print(view_matrix)
        #print('proj matrix')
        #print(proj_matrix)

        if 1:
            self.camera_dir = camera.at - camera.fr 
            self.camera_dir /= np.linalg.norm(self.camera_dir)
            #print('camera_dir', camera_dir)
            normal = np.array([0.0, -1.0, 1.0])
            normal /= np.linalg.norm(normal)
            def reflect(I, N):
                return I - 2 * np.dot(N, I) * N

            np.set_printoptions(precision=2, suppress=True)
            #print('e', self.camera_dir, normal, reflect(self.camera_dir, normal))

            glMatrixMode(GL_PROJECTION)
            #glLoadIdentity()
            glLoadMatrixf(VecF(*np.asarray(proj_matrix).astype(np.float32).T.flatten()))

            glMatrixMode(GL_MODELVIEW)
            #glLoadIdentity()
            glLoadMatrixf(VecF(*np.asarray(view_matrix).astype(np.float32).T.flatten()))
            #glRotatef(-90, 1.0, 0.0, 0.0)
        else:
            # Place the light far behind our object
            #z_offset = 0
            #z_offset -= 4 / (2 * np.tan((self.fov/2) * np.pi/180))
            #print('z_offset', z_offset)

            # Move the object deeper to the screen and rotate
            glTranslatef(0, 0, -z_offset)
            glRotatef(self.rotate_out_of_plane, 0.0, 0.0, 1.0)
            glRotatef(self.rotate_altitude, 1.0, 0.0, 0.0)
            glRotatef(self.rotate_azimuth, 0.0, 1.0, 0.0)
            #glRotatef(-90, 1.0, 0.0, 0.0)

        
        light_pos = VecF(*self._light_pos)
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos)

        glScalef(self.scale_2d, self.scale_2d, self.scale_2d)


        #self.set_light(self.cur_light, self.cur_light_seed)

        if osd:
            for batch, shader_prog, tex_id, name, colors in self.box_list:
                colors = self.update_colors(name, colors)
                shader_prog.bind()
                shader_prog.uniformf('diffuse', *colors['diffuse'])
                #shader_prog.uniformf('envAmount', 0.0)
                #self._bind_cube_stuff(shader_prog)
                batch.draw()
                shader_prog.unbind()

        #glScalef(0.01, 0.01, 0.01)
        #print('dae MM', self.dae_matrix)
        if self.dae_matrix is not None:
            glMultMatrixf(VecF(*self.dae_matrix.astype(np.float32).T.flatten()))

        # Draw batches (VBOs)
        #if not self.hide:

        if not support_rendering and self._render_shadow:
            shader_prog = self.shaders['texture-simple']
            shader_prog.bind()

            glActiveTexture(GL_TEXTURE0)
            glEnable(GL_TEXTURE_2D)
            glDisable(GL_DEPTH_TEST)
            #glEnable(GL_CULL_FACE)
            #glEnable(GL_MULTISAMPLE)

            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glBindTexture(GL_TEXTURE_2D, self._color_tex_id)
            shader_prog.uniformi('myTexture', 0)
            pyglet.graphics.draw_indexed(4, GL_QUADS,
                [0, 1, 2, 3],
                ('v3f/static', (self._shadow_x-1.3, self._ground_y, self._shadow_z-1.3,
                                self._shadow_x-1.3, self._ground_y, self._shadow_z+1.3,
                                self._shadow_x+1.3, self._ground_y, self._shadow_z+1.3,
                                self._shadow_x+1.3, self._ground_y, self._shadow_z-1.3)),
                ('t2f/static', (0.0, 1.0,
                                1.0, 1.0,
                                1.0, 0.0,
                                0.0, 0.0)))

            glBindTexture(GL_TEXTURE_2D, 0)

            shader_prog.unbind()

            glEnable(GL_DEPTH_TEST)

        #return

        if support_rendering:
            shader_prog = self.shaders['depth']
            shader_prog.bind()
            self._render_objects(use_shader_prog=False)
            shader_prog.unbind()
        else:
            self._render_objects()


        #self.shaders['texture'].bind()
        #glActiveTexture(GL_TEXTURE0)
        glEnable(GL_TEXTURE_2D)
        #glBindTexture(GL_TEXTURE_2D, tex_id)
        #shader_prog.uniformi('my_color_texture[0]', 0)

        #shader_prog = self.shaders['flat']
        #shader_prog.bind()
    
        if osd:
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            #glOrtho(-2, 2, -2, 2, -100, 100)
            glOrtho(0, self.width, 0, self.height, -1, 1) 
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            #glRotatef(-90, 1.0, 0.0, 0.0)
            glDisable(GL_DEPTH_TEST)


            if self.box is not None:
            
                shader_prog = self.shaders['flat']
                shader_prog.bind()

                glLineWidth(2.0)
                #glColor4f(1.0, 0.0, 0.0, 1.0)
                if self.box_color is not None:
                    box_color = self.box_color
                else:
                    box_color = (0.0, 1.0, 0.0, 1.0) 

                glUniform4f(glGetUniformLocation(shader_prog.handle, 'diffuse'), *box_color)
                def point(p):
                    return (self.width/2 - self.background_sprite.width/2 + p[0],
                            self.height/2 + self.background_sprite.height/2 - p[1])
                #x0 = self.width/2 - self.background_sprite.width/2 + self.box[1]
                #y0 = self.height/2 - self.background_sprite.height/2 + self.box[0]
                #x1 = x0 + 30
                #y1 = y0 + 30
                x0, y0 = point((self.box[1], self.box[0]))
                #x1, y1 = point(self.box[2:])
                x1, y1 = point((self.box[3], self.box[2])) 
                pyglet.graphics.draw(4, GL_LINE_LOOP, 
                    ('v2i', (int(x0), int(y0), 
                             int(x1), int(y0),
                             int(x1), int(y1), 
                             int(x0), int(y1))))

                shader_prog.unbind()

            def text(s, row):
                label = pyglet.text.Label(s,
                                          font_name='Arial',
                                          font_size=10,
                                          color=(1, 0, 1, 1),
                                          x=4, y=4 + 16*row) 

                #glUniform4f(glGetUniformLocation(shader_prog.handle, 'diffuse'), 1, 1, 1, 1) 
                #glColor3f(1.0, 1.0, 0.0)
                label.draw()

            text('Azimuth: {:.0f}'.format(self.rotate_azimuth), 0)
            text('Altitude: {:.0f}'.format(self.rotate_altitude), 1)
            text('Out-of-plane: {:.0f}'.format(self.rotate_out_of_plane), 2)
            text('FOV: {:.0f}'.format(self.fov), 3)
            text('Model: {0}/{1} ({2})'.format(self.cur_model+1, self.num_models, self.cur_model_name), 4)
            text('{}'.format(self.principal_direction), 5)
            text('Light: {0}'.format(self.cur_light), 6)

            for i, line in enumerate(self.extra_text):
                text(line, 7+i)

            text('Camera from: {0}'.format(camera.fr), 10)
            text('Light pos: {0}'.format(self._light_pos), 11)
        #shader_prog.unbind()

        #self.shaders['texture'].unbind()

    def delete_models(self):
        for tex_key, tex in self.textures.items():
            #import pdb; pdb.set_trace()
            tid = GLuint(tex)
            glGenTextures(1, ctypes.byref(tid))
            #tex.delete()
        #glDeleteTextures(len(self.textures), self.textures.values());



        

    def cleanup(self):
        pass
