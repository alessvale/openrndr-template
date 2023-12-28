package utils

import org.openrndr.color.ColorRGBa
import org.openrndr.draw.*
import org.openrndr.math.Matrix44
import org.openrndr.math.Vector3

interface Material {
    var objectColor: ColorRGBa
    var ambient: Double
    var diffuse: Double
    var specular: Double

    fun getShadestyle(): ShadeStyle

}

open class PhongMaterial: Material {
    override var objectColor: ColorRGBa = ColorRGBa.WHITE
    override var ambient: Double = 0.1
    override var diffuse: Double = 0.0
    override var specular: Double = 0.0
    var shadows: Boolean = false
    var shadowSmooth: Double = 0.001
    var lightPos: Vector3 = Vector3.ZERO
    var shadowMap: ColorBuffer = colorBuffer(1, 1)
    var shadowStrength: Double = 0.5
    var vertexPre: String = """
                out vec4 FragPosLightSpace;
            """.trimIndent()
    var vertexTrans: String =  """
                FragPosLightSpace = vec4(0.0);
                if (p_shadows){
                    FragPosLightSpace = p_lightSpaceMatrix * x_modelMatrix * vec4(a_position, 1.0);
                }
            """.trimIndent()
    var fragPre: String =  """
                in vec4 FragPosLightSpace;
                float ShadowCalculation(vec4 fragPosLightSpace, sampler2D shadowMap, float bias, float smoothCoeff)
                    {
                      
                        vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
                        projCoords = projCoords * 0.5 + 0.5;
                        float currentDepth = clamp(projCoords.z, 0.0, 1.0);     
                        float shadow = 0.0;
                        for( float x = -3; x<=4; x++){
                            for( float y = -3; y<=4; y++){
                            vec2 coord = projCoords.xy + vec2(x,y) * smoothCoeff;
                            float closestDepth = 1.0;
                            if (coord.x >= 0 && coord.x <= 1.0 && coord.y >= 0.0 && coord.y <= 1.0){ 
                                closestDepth = texture(shadowMap, coord.xy).r; 
                            }
                            shadow += currentDepth - bias > closestDepth  ? 1.0/32.0 : 0.0;
                            }
                        }                       
                        return shadow;
                    }  
            """.trimIndent()
    var fragTrans: String = """         
                    vec3 lightDir = normalize(p_lightPos);
                    vec3 normal = normalize(v_worldNormal);
                    float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005); 
                    float shadow=0.0;
                    if (p_shadows){
                        shadow = ShadowCalculation(FragPosLightSpace, p_depthImg, bias, p_shadowSmooth);
                    }
                    float diffuse = max(dot(lightDir, normal), 0.0);
                    vec3 viewDir = normalize(v_viewPosition - v_worldPosition);
                    vec3 reflectDir = reflect(-lightDir, normal);
                    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
                    vec4 color = p_color;
                    x_fill.rgb = color.xyz * ((diffuse * p_diffStrength + spec * p_specStrength) * (1.0 - shadow * p_shadowStrength) + p_ambient) ;
                    x_fill.a = color.a;
                """.trimIndent()

    override fun getShadestyle(): ShadeStyle {
        val shade = shadeStyle {
            vertexPreamble = vertexPre
            vertexTransform = vertexTrans
            fragmentPreamble = fragPre
            fragmentTransform = fragTrans
        }
        shade.parameter("lightPos", lightPos)
        shade.parameter("lightSpaceMatrix", Matrix44.ZERO)
        shade.parameter("color", objectColor)
        shade.parameter("ambient", ambient)
        shade.parameter("diffStrength", diffuse)
        shade.parameter("specStrength", specular)
        shade.parameter("shadows", shadows)
        shade.parameter("shadowSmooth", shadowSmooth)
        shade.parameter("shadowStrength", shadowStrength)
        shade.parameter("depthImg", shadowMap)

        if (shadows) {
            val lightProjection = org.openrndr.math.transforms.ortho(-100.0, 100.0, -100.0, 100.0, -120.0, 120.0)
            val lightView =
                org.openrndr.math.transforms.lookAt(lightPos, Vector3(0.0, 0.0, 0.0), Vector3(0.0, 0.0, 1.0))
            val lightSpaceProjection = lightProjection * lightView
            shade.parameter("lightSpaceMatrix", lightSpaceProjection)
            shade.parameter("depthImg", shadowMap)
        }
        return shade
    }
}

open class Light: Material {
    override var objectColor: ColorRGBa = ColorRGBa.WHITE
    override var ambient: Double = 0.1
    override var diffuse: Double = 0.0
    override var specular: Double = 0.0
    var lightPos: Vector3 = Vector3.ZERO
    var vertexTrans = """
                    x_projectionMatrix = p_lightSpaceMatrix;                   
                """.trimIndent()
    var fragmentTrans = """
                    x_fill.rgb = vec3(gl_FragCoord.z);
                """.trimIndent()

    override fun getShadestyle(): ShadeStyle{
        val lightProjection = org.openrndr.math.transforms.ortho(-100.0, 100.0, -100.0, 100.0, -120.0, 120.0)
        val lightView = org.openrndr.math.transforms.lookAt(lightPos, Vector3(0.0, 0.0, 0.0), Vector3(0.0, 0.0, 1.0))
        val lightSpaceProjection = lightProjection * lightView

        val shadeLight = shadeStyle {
            vertexTransform = vertexTrans
            fragmentTransform = fragmentTrans
        }
        shadeLight.parameter("lightSpaceMatrix", lightSpaceProjection)
        return shadeLight
    }
}

class PhongMaterialInstanced(private val transformAttributeName: String, private val instancedColorBuffer: InstancedColors? = null): PhongMaterial(){

    override fun getShadestyle(): ShadeStyle {
        vertexTrans = "\n x_modelMatrix = x_modelMatrix * i_${transformAttributeName};\n x_modelNormalMatrix = x_modelNormalMatrix * i_${transformAttributeName};" + vertexTrans
        val shade =  super.getShadestyle()
        if (instancedColorBuffer != null){
            shade.vertexPreamble += "\n out vec4 instanceColor;"
            shade.vertexTransform += "\n instanceColor = b_color.color[gl_InstanceID];"
            shade.fragmentPreamble += "\n in vec4 instanceColor;"
            shade.fragmentTransform = shade.fragmentTransform!!.replace("p_color", "instanceColor")
            shade.buffer("color", instancedColorBuffer.getColors())
        }
        return shade
    }
}

class LightInstanced(private val transformAttributeName: String): Light(){

    override fun getShadestyle(): ShadeStyle {
        vertexTrans += "\n x_modelMatrix = x_modelMatrix * i_${transformAttributeName};\n x_modelNormalMatrix = x_modelNormalMatrix * i_${transformAttributeName};"
        return  super.getShadestyle()
    }
}

class InstancedColors(private val colorsList:List<ColorRGBa>) {
    private val colors: ShaderStorageBuffer = shaderStorageBuffer(shaderStorageFormat {
        member("color", BufferMemberType.VECTOR4_FLOAT, colorsList.size)
    })

    init {
        colors.put {
            colorsList.forEach {
                write(it.toVector4())
            }
        }
    }

    fun getColors():ShaderStorageBuffer{
        return colors
    }
}