
varying vec4 pos;
uniform vec3 color;

void main() {
    //gl_FragColor.rgb = vec3((1.0+pos.z/pos.w)/2.0); 
    //gl_FragColor.a = 1.0;
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    //color = vec3(0.5);
}
