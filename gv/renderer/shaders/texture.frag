
uniform float Ka, Kd, Ks, shexp;
uniform float textureAmount;
uniform vec4 diffuse, specular, ambient;
uniform vec3 cameraDir;
varying vec3 normal, adjustedNormal, eyeVec;
uniform sampler2D myTexture; //0 = ColorMap
uniform samplerCube cubemapTexture;

uniform float envAmount;

// Phong lighting is applied to the first parameter
void applyPhongLighting(inout vec4 color,
                   in vec3 lightDir,
                   in vec3 vertNorm,
                   in vec3 cameraDir,
                   in vec4 ambient0,
                   in vec4 diffuse0,
                   in vec4 specular0,
                   in float shexp) {


    // Ambient
    vec4 ambient = color * ambient0;

    // Diffuse
    vec4 diffuse = color * diffuse0 * pow(max(0.0, dot(lightDir, vertNorm)), 0.4);

    // Specular - Lab prescribes the latter (the former is much nicer though)
    vec4 specular = specular0 * pow(max(0.0, dot(cameraDir, -reflect(lightDir, vertNorm))), shexp);
    //vec4 specular = lightColor * Ks * pow(max(0, dot(vertNorm, normalize(lightDir + cameraDir))), shexp);

    color = ambient + diffuse + specular;
}

void main (void)
{
	vec3 N = normalize(normal);
	vec3 AN = normalize(adjustedNormal);
    vec3 E = normalize(eyeVec);
    
	vec3 lightDir0 = vec3(gl_LightSource[0].position.xyz + E);
	vec3 L0 = normalize(lightDir0);
    vec3 R = reflect(-L0, N);

	float lambertTerm0 = max(0.0, dot(N,L0));

    vec4 texColor = texture2D(myTexture, gl_TexCoord[0].st);

    vec4 envColor;
    if (envAmount > 0.0) {
        vec3 c = textureCube(cubemapTexture, -reflect(cameraDir, N) * vec3(1.0, -1.0, 1.0)).rgb;
        //envColor = vec4(vec3((c.r + c.g + c.b) / 3.0), 1.0);
        envColor = vec4(c, 1.0);
    } else {
        envColor = vec4(1.0);
    }
    //vec4 envColor;
    vec4 final_color;

/*	final_color = (gl_FrontLightModelProduct.sceneColor * vec4(texColor.rgb,1.0)) +
		      gl_LightSource[0].ambient * vec4(texColor.rgb,1.0);*/

    vec4 materialColor = mix(mix(gl_FrontMaterial.diffuse, texColor, textureAmount), envColor, envAmount);
    //vec4 materialColor = mix(gl_FrontMaterial.diffuse, texColor, textureAmount);
    materialColor.a = gl_FrontMaterial.diffuse.a; 

	//final_color = ((gl_FrontLightModelProduct.sceneColor * vec4(texColor.rgb,1.0)) + 
	//	       gl_LightSource[0].ambient * vec4(texColor.rgb,1.0)) * textureAmount + 
    //           diffuse * (1.0 - textureAmount);


    final_color = materialColor * (gl_LightSource[0].ambient + gl_LightSource[0].diffuse * vec4(vec3(lambertTerm0), 1.0)) + gl_LightSource[0].specular * pow(lambertTerm0, 2.0);

    if (false) {
    float specular = pow( max(dot(R, E), 0.0),
                     gl_FrontMaterial.shininess );
    final_color += gl_LightSource[0].specular *
                   gl_FrontMaterial.specular *
                   specular;
    }
//    final_color = materialColor;

    float intensity = 1.0;

    final_color = materialColor; 

    vec4 dif = gl_LightSource[0].diffuse;
    //vec4 dif = mix(gl_LightSource[0].diffuse, gl_LightSource[0].diffuse * (envColor + vec4(0.5)), envAmount);
    //dif.rgb = vec3(1.0);
    //dif.a = 1.0;

    //gl_LightSource[0].specular
    applyPhongLighting(final_color, L0, AN, cameraDir, gl_LightSource[0].ambient,
                       dif, 
                       gl_LightSource[0].specular, gl_FrontMaterial.shininess);
 

    final_color.a = materialColor.a;
    //final_color.a = 1.0;
	gl_FragColor = final_color;

    //gl_FragColor.rgb = AN.xyz;
    //gl_FragColor.rgb = (reflect(cd, N) + 1.0)/2.0;
    //gl_FragColor.rgb = N;
}
