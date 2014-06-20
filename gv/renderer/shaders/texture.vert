
varying vec3 normal, adjustedNormal, eyeVec;
uniform vec3 cameraDir;

//in vec3 tangent;
//out vec3 varVertTang;

void main()
{
	adjustedNormal = normalize(gl_NormalMatrix * gl_Normal);
	normal = gl_Normal; 

	vec3 vVertex = vec3(gl_ModelViewMatrix * gl_Vertex);

	//lightDir0 = vec3(gl_LightSource[0].position.xyz - vVertex);
	eyeVec = normalize(-vVertex);

	gl_Position = ftransform();
    //gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex;
    gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0;
}
