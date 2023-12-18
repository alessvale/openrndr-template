package utils

import org.openrndr.color.ColorRGBa
import org.openrndr.draw.*
import org.openrndr.extra.noise.Random
import org.openrndr.extra.noise.simplex
import org.openrndr.extra.noise.uniform
import org.openrndr.math.Vector2
import org.openrndr.math.Vector3
import org.openrndr.shape.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sign
import kotlin.math.sin

/**
 * Generates a random walk path starting from a point
 */
fun Vector2.randomWalk(n: Int, step: Double = 1.0): List<Vector2> {
    var start = this
    val points = mutableListOf<Vector2>()
    points.add(start)
    (1 until n).forEach {
        //start += Random.point(Rectangle.fromCenter(Vector2.ZERO, step * 2.0, step * 2.0))
        start += Vector2.uniform(-step * 2.0, step * 2.0)
        points.add(start)
    }
    return points
}

/**
 * Constrains a 2d Vector to lie inside a rectangle
 */
fun Vector2.constrain(bound: Rectangle): Vector2 {
    var tmp = this
    if (tmp.x < bound.center.x - bound.width * 0.5) tmp = tmp.copy(x = bound.center.x - bound.width * 0.5)
    if (tmp.x > bound.center.x + bound.width * 0.5) tmp = tmp.copy(x = bound.center.x + bound.width * 0.5)
    if (tmp.y < bound.center.y - bound.height * 0.5) tmp = tmp.copy(x = bound.center.y - bound.height * 0.5)
    if (tmp.y > bound.center.y + bound.height * 0.5) tmp = tmp.copy(x = bound.center.y + bound.height * 0.5)
    return tmp
}

/**
 * Gets centroid of a ShapeContour
 */
fun ShapeContour.centroid(): Vector2 {

    var center = Vector2.ZERO
    var count = 0.0
    this.segments.forEach {
        center += it.start
        count += 1.0
    }
    return center / count
}

/**
 * Returns n points sampled from the black region in a b&w texture.
 */
fun ColorBuffer.samplePoints(n: Int): List<Vector2> {
    val tex = this
    val shadow = tex.shadow
    shadow.download()
    val points = mutableListOf<Vector2>()
    (0..n).forEach { _ ->
        var found = false
        var count = 0
        while ((!found) && (count < 1000)) {
            val p = Vector2.uniform(Rectangle(0.0, 0.0, tex.width * 1.0, tex.height * 1.0))
            val c = shadow[p.x.toInt(), p.y.toInt()]
            if (c.r == 0.0) {
                points.add(p)
                found = true
            } else {
                count += 1
            }
        }
    }
    return points
}

/**
 * Returns the vector in List<Vector2> most distant from p.
 */
fun List<Vector2>.mostDistant(p: Vector2): Vector2 {
    val distances = this.map { Pair(it, it.distanceTo(p)) }.filter { it.second != 0.0 }
    val idx = distances.indexOf(distances.maxBy { it.second })
    return distances[idx].first
}

/**
 * Performs sampling with weights
 */
fun <T> Collection<T>.randomWeighted(w: Collection<Double>): Pair<T, Int> {
    val items = this.toList()
    var weights = w.toList()
    if (items.size != weights.size) {
        throw Exception("Items size and weights size are different!")
    }
    val prefixSum = weights.sum()
    weights = weights.map { it * 1.0 / prefixSum }
    val t = Random.double(0.0, 1.0)
    var s = 0.0
    for (i in weights.indices) {
        val it = weights[i]
        s += it
        if (t < s) {
            return Pair(items[i], i)
        }
    }
    return Pair(items.last(), items.size - 1)
}


/**
 * Splits a rectangle
 */
fun Rectangle.split(how: Int, p: Double = 0.35): List<Rectangle> {
    val corner = this.corner
    val w = this.width
    val h = this.height
    val t = if (p < 0.5) Random.double(p, 1.0 - p) else Random.double(1.0 - p, p)

    val result = if (how == 1) {
        arrayListOf<Rectangle>(Rectangle(corner, t * w, h), Rectangle(corner + Vector2(t * w, 0.0), (1.0 - t) * w, h))
    } else {
        arrayListOf<Rectangle>(Rectangle(corner, w, t * h), Rectangle(corner + Vector2(0.0, t * h), w, (1.0 - t) * h))
    }
    return result
}

/**
 * Get geometry buffer from a list of line segments
 */
fun List<LineSegment>.getVertexBuffer(cols: List<ColorRGBa>): VertexBuffer {
    val lines = this
    val colors = cols.map { it.toVector4().xyz }
    if (lines.size != colors.size) {
        throw Exception("List of lines and colors must have the same size")
    }
    val geometry = vertexBuffer(vertexFormat {
        position(3)
        color(3)
    }, 2 * lines.size)
    geometry.put {
        (lines zip colors).forEach {
            write(it.first.start.x.toFloat(), it.first.start.y.toFloat(), 0.0.toFloat())
            write(it.second.x.toFloat(), it.second.y.toFloat(), it.second.z.toFloat())
            write(it.first.end.x.toFloat(), it.first.end.y.toFloat(), 0.0.toFloat())
            write(it.second.x.toFloat(), it.second.y.toFloat(), it.second.z.toFloat())
        }
    }
    return geometry
}

/**
 * Hatches a rectangle
 */
fun Rectangle.hatch(dir: Vector2, w: Double = 5.0): List<Segment> {
    val corner = if (dir.x > 0) this.corner else this.corner + Vector2.UNIT_X * this.width
    val l = (this.center - corner).length * 2.0
    val offset = Vector2(-dir.y, dir.x) * l * sign(dir.x)
    val steps = (offset.length / w).toInt()
    val segments = mutableListOf<Segment>()
    repeat(steps) {
        val t = it * 1.0 / (steps - 1)
        val seg = Segment(corner - dir * l + offset * t, corner + dir * l + offset * t)
        val inter = intersections(this.contour, seg.contour)
        if (inter.size == 2) {
            segments.add(Segment(inter[0].position, inter[1].position))
        }
    }
    return segments.toList()
}

/**
 * Splits a ShapeContour
 */
fun ShapeContour.half(): List<ShapeContour> {
    val pointA = this.position(Random.double(0.2, 0.4))
    val pointB = this.position(Random.double(0.6, 0.8))
    val sh = this.split(Segment(pointA, pointB).contour)
    return sh.map { it.close() }
}

/**
 * Implements uniform random sample with replacement
 */
fun <T> Collection<T>.uniformWithoutReplacement(k: Int): List<T> {
    val input = this.toMutableList()
    if ((input.size < k) || (k == 0)) {
        throw Exception("Items size and weights size are different!")
    }
    val output = mutableListOf<T>()
    while (output.size < k) {
        val item = input.withIndex().toList().random()
        output.add(item.value)
        input.removeAt(item.index)
    }
    return output.toList()
}

/**
 * Implement k-step cycling through a collection
 */
fun <T> Collection<T>.cycle(k: Int = 1, clockwise: Boolean = true): List<T> {
    var output = this.toList()
    (0 until k).forEach{
        output = if(clockwise) {output.takeLast(1) + output.dropLast(1)}
        else
        {output.drop(1) + output.take(1)}
    }
    return output.toList()
}

/**
 * Implement comparator function for 2D points ordering
 */
fun less(a:Vector2, b:Vector2):Int {

        if (a.x >= 0.0 && b.x < 0.0)
            return 1
        if (a.x  < 0.0 && b.x >= 0.0)
            return 0
        if ((a.x  == 0.0) && (b.x  == 0.0)) {
            if (a.y  >= 0.0 || b.y >= 0.0)
                return (a.y > b.y).compareTo(false)
            return (b.y > a.y).compareTo(false)
        }

// compute the cross product of vectors (center -> a) x (center -> b)
        val det = a.x * b.y  - b.x * a.y
        if (det < 0.0)
            return 1
        if (det > 0.0)
            return 0

// points a and b are on the same line from the center
// check which point is closer to the center
        val d1 =a.x * a.x + a.y * a.y
        val d2 =b.x * b.x + b.y * b.y
        return if(d1 > d2) 1 else 0
    }
/**
 * Implements clockwise sorting of a list of 2D points
 */
fun List<Vector2>.sortClockwise():List<Vector2>{
    if (this.size == 1) return this
    var cnt = Vector2.ZERO
    this.forEach {
        cnt += it / this.size.toDouble()
    }
    val points = this.map{it - cnt}
    return points.sortedWith { a, b -> less(a, b) * 2 - 1 }.map{it + cnt}
}
/**
 * Implements precomputed loopable simplex noise texture
 */
fun getNoiseTexture( noiseWidth:Int = 512, noiseHeight:Int = 512,  noiseDepth:Int = 128, freq: Double = 0.004 , seed: Int = 10):VolumeTexture {
    val tn = volumeTexture(noiseWidth, noiseHeight, noiseDepth)
    val buffer = ByteBuffer.allocateDirect(tn.width * tn.height * tn.format.componentCount * tn.type.componentSize)

    for (z in 0 until tn.depth) {
        for (y in 0 until tn.height) {
            for (x in 0 until tn.width) {
                val uVal = cos(z / noiseDepth.toDouble() * 2 * PI) * 0.5 + 0.5
                val wVal = sin(z / noiseDepth.toDouble() * 2 * PI) * 0.5 + 0.5
                val noiseValue = simplex(seed, x * freq, y * freq, uVal, wVal) * 0.5 + 0.5
                for (c in 0 until tn.format.componentCount) {
                    buffer.put((noiseValue * 255).toInt().toByte())
                }
            }
        }
        buffer.rewind()
        tn.write(z, buffer)
    }
    return tn
}
/**
 * Implements a simple feedback line which allows the insertion of a filter.
 */
class Feedback(private val w: Int, private val h: Int) {
    val cumulated = colorBuffer(w, h, type = ColorType.FLOAT32)
    private val rt = renderTarget(w, h) {
        colorBuffer(type = ColorType.FLOAT32)
    }

    private fun makeFrame(drawer: Drawer, current: ColorBuffer, feedback: Double = 0.8) {
        drawer.isolatedWithTarget(rt) {
            drawer.clear(ColorRGBa.BLACK)
            drawer.ortho()
            drawer.fill = ColorRGBa.WHITE
            drawer.stroke = null
            drawer.shadeStyle = shadeStyle {
                fragmentTransform = """
                    vec2 coords = c_screenPosition.xy/p_resolution;
                    coords.y = 1.0 - coords.y;
                    vec3 current = texture(p_current, coords).rgb;
                    vec3 prev = texture(p_previous, coords).rgb;
                    x_fill.rgb = current + p_feedback * prev;
                """.trimIndent()
                parameter("feedback", feedback)
                parameter("resolution", Vector2(w * 1.0, h * 1.0))
                parameter("current", current)
                parameter("previous", cumulated)
            }
            drawer.rectangle(Rectangle(0.0, 0.0, w * 1.0, h * 1.0))
        }
        rt.colorBuffer(0).copyTo(cumulated)
    }

    fun next(drawer: Drawer, current: ColorBuffer, feedback: Double = 0.8, filter: Filter? = null) {
        makeFrame(drawer, current, feedback)
        filter?.apply(cumulated, cumulated)
    }

    fun next(drawer: Drawer, current: ColorBuffer, feedback: Double = 0.8, filters: Array<Filter1to1>? = null) {
        makeFrame(drawer, current, feedback)
        filters?.let {
            it.forEach { filter ->
                filter.apply(cumulated, cumulated)
            }
        }
    }
}

/**
 * Creates index buffer from list of vertices.
 */

fun getBufferFromIndices(indices: List<Int>):IndexBuffer{
    val ib = indexBuffer(indices.size, IndexType.INT32)
    val bb = ByteBuffer.allocateDirect(indices.size * 4)
    bb.order(ByteOrder.nativeOrder())
    indices.forEach {
        bb.putInt(it)
    }
    bb.rewind()
    ib.write(bb)
    return ib
}

/**
 *
 */
fun sumOctave(
    num_iterations: Int,
    x: Double,
    y: Double,
    u:Double,
    v:Double,
    persistence: Double,
    scale: Double,
    low: Double,
    high: Double
): Double {
    var maxAmp: Double = 0.0
    var amp = 1.0
    var freq = scale
    var noise = 0.0

    for (i in 0 until num_iterations) {
        noise += (simplex(10, x * freq, y * freq, u, v) + 1.0)/2.0 * amp
        maxAmp += amp
        amp *= persistence
        freq *= 2.0
    }
    noise /= maxAmp
    noise = noise * (high - low) / 2 + (high + low) / 2

    return noise
}

fun getNoiseTextureBrown( noiseWidth:Int = 512, noiseHeight:Int = 512,  noiseDepth:Int = 128, freq: Double = 0.004 , seed: Int = 10):VolumeTexture {
    val tn = volumeTexture(noiseWidth, noiseHeight, noiseDepth)
    val buffer = ByteBuffer.allocateDirect(tn.width * tn.height * tn.format.componentCount * tn.type.componentSize)

    val scale = freq
    for (z in 0 until tn.depth) {
        for (y in 0 until tn.height) {
            for (x in 0 until tn.width) {
                val uVal = cos(z / noiseDepth.toDouble() * 2 * PI) * 0.5 + 0.5
                val wVal = sin(z / noiseDepth.toDouble() * 2 * PI) * 0.5 + 0.5
                val noiseValue = sumOctave(8, x.toDouble(), y.toDouble(), 0.5, scale, uVal, wVal, 0.0, 1.0)
                for (c in 0 until tn.format.componentCount) {
                    buffer.put((noiseValue * 255).toInt().toByte())
                }
            }
        }
        buffer.rewind()
        tn.write(z, buffer)
    }
    return tn
}

fun Drawer.getShadows(rt:RenderTarget, lightPos:Vector3, function:ShadeStyle.() -> Unit):ColorBuffer{

    val lightProjection = org.openrndr.math.transforms.ortho(-100.0, 100.0, -100.0, 100.0, -120.0, 120.0)
    val lightView = org.openrndr.math.transforms.lookAt(lightPos, Vector3(0.0, 0.0, 0.0), Vector3(0.0, 0.0, 1.0))
    val lightSpaceProjection = lightProjection * lightView

    val shadeLight = shadeStyle {
        vertexTransform = """
                    x_projectionMatrix = p_lightSpaceMatrix;                   
                """.trimIndent()

        fragmentTransform = """
                    x_fill.rgb = vec3(gl_FragCoord.z);
                """.trimIndent()
    }
    shadeLight.parameter("lightSpaceMatrix", lightSpaceProjection)

    isolatedWithTarget(rt){
        clear(ColorRGBa.BLACK)
        cullTestPass = CullTestPass.FRONT
        function(shadeLight)
    }
    return rt.colorBuffer(0)
}

/**
 * Gets a shadestyle which implements shadows
 */
fun getShadowPhong(lightPos: Vector3, shadowMap: ColorBuffer):ShadeStyle{
    val lightProjection = org.openrndr.math.transforms.ortho(-100.0, 100.0, -100.0, 100.0, -120.0, 120.0)
    val lightView = org.openrndr.math.transforms.lookAt(lightPos, Vector3(0.0, 0.0, 0.0), Vector3(0.0, 0.0, 1.0))
    val lightSpaceProjection = lightProjection * lightView

    val shade = shadeStyle {
        vertexPreamble = """
                out vec4 FragPosLightSpace;
            """.trimIndent()
        vertexTransform = """
                FragPosLightSpace = p_lightSpaceMatrix * x_modelMatrix * vec4(a_position, 1.0);
            """.trimIndent()
        fragmentPreamble = """
                in vec4 FragPosLightSpace;
                float ShadowCalculation(vec4 fragPosLightSpace, sampler2D shadowMap, float bias)
                    {
                        // perform perspective divide
                        vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
                        // transform to [0,1] range
                        projCoords = projCoords * 0.5 + 0.5;
                        // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
                        float closestDepth = 1.0;
                        if (projCoords.x >= 0 && projCoords.x <= 1.0 && projCoords.y >= 0.0 && projCoords.y <= 1.0){ 
                            closestDepth = texture(shadowMap, projCoords.xy).r; 
                        }
                        // get depth of current fragment from light's perspective
                        float currentDepth = clamp(projCoords.z, 0.0, 1.0);
                        // check whether current frag pos is in shadow
                        float shadow = 0.0;
                        for( float x = -3; x<=4; x++){
                            for( float y = -3; y<=4; y++){
                            vec2 coord = projCoords.xy + vec2(x,y) * 0.0015;
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
        fragmentTransform = """
             
                    vec2 uv = va_texCoord0.xy;
                    vec3 lightDir = normalize(p_lightPos);
                    vec3 normal = normalize(v_worldNormal);
                    float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005); 
                    float shadow = ShadowCalculation(FragPosLightSpace, p_depthImg, bias);
                    float d = max(dot(lightDir, normal), 0.0);
                    x_fill.rgb = p_color * (d * (1.0 - shadow) + p_ambient) ;
                """.trimIndent()
    }
    shade.parameter("lightPos", lightPos)
    shade.parameter("lightSpaceMatrix", lightSpaceProjection)
    shade.parameter("depthImg", shadowMap)
    shade.parameter("ambient", 0.1)
    return shade
}