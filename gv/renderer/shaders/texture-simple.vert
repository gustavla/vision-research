
varying vec3 normal, lightDir0, eyeVec;

//in vec3 tangent;
//out vec3 varVertTang;

void main()
{
	//normal = normalize(gl_NormalMatrix * gl_Normal);
	normal = gl_Normal; 

	vec3 vVertex = vec3(gl_ModelViewMatrix * gl_Vertex);

	lightDir0 = vec3(gl_LightSource[0].position.xyz - vVertex);
	eyeVec = -vVertex;

	gl_Position = ftransform();
    //gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex;
    gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0;
}
