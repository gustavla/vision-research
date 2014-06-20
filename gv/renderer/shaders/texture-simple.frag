
//uniform float textureAmount;
//uniform vec4 diffuse, specular, ambient;
//uniform vec3 cameraDir;
varying vec3 normal, lightDir0, eyeVec;
uniform sampler2D myTexture; //0 = ColorMap

//uniform float envAmount;

void main (void)
{
    vec4 texColor = texture2D(myTexture, gl_TexCoord[0].st);
    gl_FragColor = texColor; 
    //gl_FragColor.rg += gl_TexCoord[0].st;
    //gl_FragColor.a = 1.0;
}
