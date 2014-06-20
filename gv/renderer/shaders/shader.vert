
varying vec3 normal, lightDir0, eyeVec;
void main()
{
    normal = gl_NormalMatrix * gl_Normal;
    vec3 vVertex = vec3(gl_ModelViewMatrix * gl_Vertex);
    lightDir0 = vec3(gl_LightSource[0].position.xyz - vVertex);
    eyeVec = -vVertex;
    gl_Position = ftransform();
}
