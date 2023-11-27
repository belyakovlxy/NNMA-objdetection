package ai.onnxruntime.example.objectdetection

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.util.Collections

internal data class Result(
    var outputBitmap: Bitmap
) {}

internal class ObjectDetector(
) {

    fun detect(inputStream: InputStream, ortEnv: OrtEnvironment, ortSession: OrtSession): Result {
        // Step 1: convert image into byte array (raw image bytes)
        val rawImageBytes = inputStream.readBytes()

        // Step 2: get the shape of the byte array and make ort tensor
        val shape = longArrayOf(rawImageBytes.size.toLong())

        val inputTensor = OnnxTensor.createTensor(
            ortEnv,
            ByteBuffer.wrap(rawImageBytes),
            shape,
            OnnxJavaType.UINT8
        )
        inputTensor.use {
            // Step 3: call ort inferenceSession run
            val output = ortSession.run(Collections.singletonMap("image", inputTensor),
//                setOf("image_out","scaled_box_out_next")
                setOf("image_out")

            )


            // Step 4: output analysis
            output.use {
                val rawOutput = (output?.get(0)?.value) as ByteArray
                val outputImageBitmap = byteArrayToBitmap(rawOutput)

                // Step 5: set output result
                var result = Result(outputImageBitmap)
                return result
            }
        }
    }

    private fun byteArrayToBitmap(data: ByteArray): Bitmap {
        return BitmapFactory.decodeByteArray(data, 0, data.size)
    }



//    private fun getYV12(inputWidth: Int, inputHeight: Int, scaled: Bitmap): ByteArray? {
//        val argb = IntArray(inputWidth * inputHeight)
//        scaled.getPixels(argb, 0, inputWidth, 0, 0, inputWidth, inputHeight)
//        val yuv = ByteArray(inputWidth * inputHeight * 3 / 2)
//        encodeYV12(yuv, argb, inputWidth, inputHeight)
//        scaled.recycle()
//        return yuv
//    }
//
//    private fun encodeYV12(yuv420sp: ByteArray, argb: IntArray, width: Int, height: Int) {
//        val frameSize = width * height
//        var yIndex = 0
//        var uIndex = frameSize
//        var vIndex = frameSize + frameSize / 4
//        var a: Int
//        var R: Int
//        var G: Int
//        var B: Int
//        var Y: Int
//        var U: Int
//        var V: Int
//        var index = 0
//        for (j in 0 until height) {
//            for (i in 0 until width) {
//                a = argb[index] and -0x1000000 shr 24 // a is not used obviously
//                R = argb[index] and 0xff0000 shr 16
//                G = argb[index] and 0xff00 shr 8
//                B = argb[index] and 0xff shr 0
//
//                // well known RGB to YUV algorithm
//                Y = (66 * R + 129 * G + 25 * B + 128 shr 8) + 16
//                U = (-38 * R - 74 * G + 112 * B + 128 shr 8) + 128
//                V = (112 * R - 94 * G - 18 * B + 128 shr 8) + 128
//
//                // YV12 has a plane of Y and two chroma plans (U, V) planes each sampled by a factor of 2
//                //    meaning for every 4 Y pixels there are 1 V and 1 U.  Note the sampling is every other
//                //    pixel AND every other scanline.
//                yuv420sp[yIndex++] = (if (Y < 0) 0 else if (Y > 255) 255 else Y).toByte()
//                if (j % 2 == 0 && index % 2 == 0) {
//                    yuv420sp[uIndex++] = (if (V < 0) 0 else if (V > 255) 255 else V).toByte()
//                    yuv420sp[vIndex++] = (if (U < 0) 0 else if (U > 255) 255 else U).toByte()
//                }
//                index++
//            }
//        }
//    }
}