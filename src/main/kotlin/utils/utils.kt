package utils

import org.openrndr.color.ColorRGBa
import org.openrndr.draw.*
import org.openrndr.extra.noise.Random
import org.openrndr.extra.noise.uniform
import org.openrndr.math.Vector2
import org.openrndr.shape.*
import java.util.*
import kotlin.math.sign


// Generates a random walk path starting from a point
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

// Constrains a 2d Vector to lie inside a rectangle
fun Vector2.constrain(bound: Rectangle): Vector2 {
    var tmp = this
    if (tmp.x < bound.center.x - bound.width * 0.5) tmp = tmp.copy(x = bound.center.x - bound.width * 0.5)
    if (tmp.x > bound.center.x + bound.width * 0.5) tmp = tmp.copy(x = bound.center.x + bound.width * 0.5)
    if (tmp.y < bound.center.y - bound.height * 0.5) tmp = tmp.copy(x = bound.center.y - bound.height * 0.5)
    if (tmp.y > bound.center.y + bound.height * 0.5) tmp = tmp.copy(x = bound.center.y + bound.height * 0.5)
    return tmp
}

// Gets centroid of a ShapeContour
fun ShapeContour.centroid(): Vector2? {
    if (this.closed) {
        var center = Vector2.ZERO
        var count = 0.0
        this.segments.forEach {
            center += it.start
            count += 1.0
        }
        return center / count
    }
    return null
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
            break
        }
    }
    return Pair(items.last(), items.size - 1)
}

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

fun ShapeContour.half(): List<ShapeContour> {
    val pointA = this.position(Random.double(0.2, 0.4))
    val pointB = this.position(Random.double(0.6, 0.8))
    val sh = this.split(Segment(pointA, pointB).contour)
    return sh.map { it.close() }
}

/**
 * Implement uniform random sample with replacement
 */
fun <T> Collection<T>.uniformWithoutReplacement(k: Int): List<T>{
    val input = this.toMutableList()
    if ((input.size < k) || (k  == 0)) {
        throw Exception("Items size and weights size are different!")
    }
    val output = mutableListOf<T>()
    while (output.size < k){
        val item = input.withIndex().toList().random()
        output.add(item.value)
        input.removeAt(item.index)
    }
    return output.toList()
}

/**
 * Implement k-step cycling through a collection
 */
fun <T> Collection<T>.cycle(k:Int = 1, clockwise:Boolean = true):List<T>{
    var output = this.toList()
    (0 until k).forEach{
        output = if(clockwise) {output.takeLast(1) + output.dropLast(1)}
        else
        {output.drop(1) + output.take(1)}
    }
    return output.toList()
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