package com.example.chirpapp

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlin.math.PI
import kotlin.math.sin

class MainActivity : ComponentActivity() {

    private val sampleRate = 44100

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            val status = remember { mutableStateOf("Play Chirp") }

            Button(onClick = {
                status.value = "Playing…"
                // Do audio work off the UI thread
                lifecycleScope.launch(Dispatchers.Default) {
                    val ok = runCatching { playChirpSweep( startHz = 18000.0, endHz = 20000.0, seconds = 2 ) }.isSuccess
                    runOnUiThread {
                        status.value = if (ok) "Done — Play Again" else "Failed (check Logcat)"
                        if (!ok) Toast.makeText(this@MainActivity, "Audio error", Toast.LENGTH_SHORT).show()
                    }
                }
            }) {
                Text(status.value)
            }
        }
    }

    /** Play a linear frequency sweep using a streaming AudioTrack (no permissions needed) */
    private fun playChirpSweep(startHz: Double, endHz: Double, seconds: Int) {
        val totalSamples = seconds * sampleRate
        val shorts = ShortArray(totalSamples)

        // Generate sweep
        for (i in 0 until totalSamples) {
            val t = i.toDouble() / sampleRate
            val f = startHz + (endHz - startHz) * (t / seconds)
            val s = (sin(2.0 * PI * f * t) * 32767.0).toInt().toShort()
            shorts[i] = s
        }

        val format = AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setSampleRate(sampleRate)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
            .build()

        val attrs = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
            .build()

        val minBuf = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        ).coerceAtLeast(4096)

        val track = AudioTrack.Builder()
            .setAudioAttributes(attrs)
            .setAudioFormat(format)
            .setTransferMode(AudioTrack.MODE_STREAM) // more forgiving than STATIC
            .setBufferSizeInBytes(minBuf)
            .build()

        try {
            if (track.state != AudioTrack.STATE_INITIALIZED) error("AudioTrack init failed")
            track.play()

            // stream in chunks
            val byteBuf = ByteArray(minBuf)
            var i = 0
            while (i < shorts.size) {
                val shortsThisChunk = (minBuf / 2).coerceAtMost(shorts.size - i) // 2 bytes/short
                var j = 0
                for (k in 0 until shortsThisChunk) {
                    val s = shorts[i + k].toInt()
                    byteBuf[j++] = (s and 0xFF).toByte()
                    byteBuf[j++] = ((s shr 8) and 0xFF).toByte()
                }
                track.write(byteBuf, 0, j)
                i += shortsThisChunk
            }
        } finally {
            try { track.stop() } catch (_: Exception) {}
            track.release()
        }
    }
}
