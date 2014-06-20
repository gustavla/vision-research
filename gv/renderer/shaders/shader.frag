
uniform vec4 diffuse, specular, ambient;
uniform float shininess;
varying vec3 normal, lightDir0, eyeVec;

bool isNan(float val)
{
    return (val <= 0.0 || 0.0 <= val) ? false : true;
}

void main (void)
{
    vec4 light = gl_LightSource[0].ambient;

    vec3 lightDir0b = vec3(gl_LightSource[0].position.xyz + eyeVec);

    vec3 N = normalize(normal);
    vec3 L0 = normalize(lightDir0b);
    float lambertTerm0 = max(0.0, dot(N,L0));
    float sf = max(0.0, dot(N, L0));
    sf = pow(sf, gl_FrontMaterial.shininess);

    light += gl_LightSource[0].diffuse * lambertTerm0 + gl_LightSource[0].specular * sf; 

    light.a = 1.0;
    light.r = max(min(light.r, 1.0), 0.0);
    light.g = max(min(light.g, 1.0), 0.0);
    light.b = max(min(light.b, 1.0), 0.0);

    vec4 final_color = light;

    gl_FragColor = diffuse * final_color; 
    //gl_FragColor = vec4(N, 1.0);
}
