
varying vec4 pos;

void main() {
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex; 
    pos = gl_Position;
}
