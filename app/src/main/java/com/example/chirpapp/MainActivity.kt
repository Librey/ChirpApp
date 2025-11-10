package com.example.chirpapp

import android.Manifest
import android.content.pm.PackageManager
import android.media.*
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.*
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import kotlin.math.PI
import kotlin.math.sin
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier

class MainActivity : ComponentActivity() {

    private val sampleRate = 44100
    private val chirpStartHz = 18000.0
    private val chirpEndHz = 20000.0
    private val chirpDurationSeconds = 2
    private val sessionDurationSeconds = 10

    private val status = mutableStateOf("Request Permission")
    private var job: Job? = null

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                status.value = "Start Chirping"
                Toast.makeText(this, "Permission granted", Toast.LENGTH_SHORT).show()
            } else {
                status.value = "Permission Denied"
                Toast.makeText(
                    this,
                    "Microphone permission is required for this app to work",
                    Toast.LENGTH_LONG
                ).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            val currentStatus = remember { status }

            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Button(
                    onClick = {
                        when (currentStatus.value) {
                            "Start Chirping" -> {
                                startChirpAndRecord()
                                currentStatus.value = "Stop Chirping"
                            }
                            "Stop Chirping" -> {
                                stopChirpAndRecord()
                                currentStatus.value = "Start Chirping"
                            }
                            else -> {
                                requestMicrophonePermission()
                            }
                        }
                    }
                ) {
                    Text(currentStatus.value)
                }
            }
        }

        if (hasRecordPermission()) status.value = "Start Chirping"
    }

    private fun hasRecordPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED

    private fun requestMicrophonePermission() {
        requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
    }

    private fun startChirpAndRecord() {
        if (job?.isActive == true) return
        job = lifecycleScope.launch(Dispatchers.Default) {
            val recordedAudio = ByteArrayOutputStream()
            val chirpSamples = generateChirpSamples()

            val recorder = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                AudioRecord.getMinBufferSize(
                    sampleRate,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT
                )
            )

            val track = AudioTrack.Builder()
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                        .build()
                )
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(sampleRate)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build()
                )
                .setTransferMode(AudioTrack.MODE_STREAM)
                .setBufferSizeInBytes(chirpSamples.size)
                .build()

            recorder.startRecording()
            track.play()

            launch {
                while (isActive) {
                    track.write(chirpSamples, 0, chirpSamples.size)
                }
            }

            val buffer = ByteArray(2048)
            val sessionEnd = System.currentTimeMillis() + (sessionDurationSeconds * 1000)
            while (System.currentTimeMillis() < sessionEnd && isActive) {
                val read = recorder.read(buffer, 0, buffer.size)
                if (read > 0) recordedAudio.write(buffer, 0, read)
            }

            recorder.stop()
            recorder.release()
            track.stop()
            track.release()

            saveAsPcm(recordedAudio.toByteArray())
        }
    }

    private fun stopChirpAndRecord() {
        job?.cancel()
    }

    private fun generateChirpSamples(): ByteArray {
        val totalSamples = chirpDurationSeconds * sampleRate
        val chirpShorts = ShortArray(totalSamples)

        for (i in 0 until totalSamples) {
            val t = i.toDouble() / sampleRate
            val freq = chirpStartHz + (chirpEndHz - chirpStartHz) * (t / chirpDurationSeconds)
            val amplitude = (sin(2 * PI * freq * t) * 32767.0).toInt()
                .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
            chirpShorts[i] = amplitude.toShort()
        }

        return ByteArray(chirpShorts.size * 2).apply {
            for (i in chirpShorts.indices) {
                val s = chirpShorts[i].toInt()
                this[i * 2] = (s and 0xFF).toByte()
                this[i * 2 + 1] = ((s shr 8) and 0xFF).toByte()
            }
        }
    }

    // 🔽 Updated Function: Save raw PCM data instead of WAV
    private fun saveAsPcm(data: ByteArray) {
        val fileName = "chirp_reflection_${System.currentTimeMillis()}.pcm"
        val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        val file = File(downloadsDir, fileName)

        FileOutputStream(file).use { out ->
            out.write(data)
        }

        Log.i("ChirpApp", "✅ PCM file saved to ${file.absolutePath} (${data.size} bytes)")
    }
}
