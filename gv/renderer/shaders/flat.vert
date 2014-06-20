
uniform vec4 diffuse;

void main() {
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex; 
}
