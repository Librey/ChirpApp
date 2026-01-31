package com.example.chirpapp

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.media.*
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.*
import java.io.File
import java.io.FileOutputStream
import kotlin.math.PI
import kotlin.math.sin

class MainActivity : ComponentActivity() {

    private val sampleRate = 44100
    private val chirpStartHz = 18000.0
    private val chirpEndHz = 20000.0
    private val chirpDurationSeconds = 2

    private val status = mutableStateOf("Request Permission")
    private var job: Job? = null
    private var recordingStartTime = 0L

    // Recording settings
    private val labelState = mutableStateOf(TextFieldValue("eating"))
    private val distanceState = mutableStateOf(TextFieldValue("30"))
    private val angleState = mutableStateOf(TextFieldValue("120"))

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
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    ChirpAppUI()
                }
            }
        }

        if (hasRecordPermission()) status.value = "Start Chirping"
    }

    @Composable
    fun ChirpAppUI() {
        val currentStatus = remember { status }
        val label = remember { labelState }
        val distance = remember { distanceState }
        val angle = remember { angleState }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = "ChirpApp Settings",
                style = MaterialTheme.typography.headlineMedium,
                modifier = Modifier.padding(bottom = 24.dp)
            )

            // Label input
            OutlinedTextField(
                value = label.value,
                onValueChange = { label.value = it },
                label = { Text("Label (e.g., eating, idle, drinking)") },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 12.dp),
                enabled = currentStatus.value == "Start Chirping"
            )

            // Distance input
            OutlinedTextField(
                value = distance.value,
                onValueChange = { distance.value = it },
                label = { Text("Distance (cm)") },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 12.dp),
                enabled = currentStatus.value == "Start Chirping"
            )

            // Angle input
            OutlinedTextField(
                value = angle.value,
                onValueChange = { angle.value = it },
                label = { Text("Angle (degrees)") },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 24.dp),
                enabled = currentStatus.value == "Start Chirping"
            )

            // Main button
            Button(
                onClick = {
                    when (currentStatus.value) {
                        "Start Chirping" -> {
                            // Validate inputs
                            if (label.value.text.isBlank()) {
                                Toast.makeText(
                                    this@MainActivity,
                                    "Please enter a label",
                                    Toast.LENGTH_SHORT
                                ).show()
                                return@Button
                            }
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
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp)
            ) {
                Text(
                    text = currentStatus.value,
                    style = MaterialTheme.typography.bodyLarge
                )
            }

            // Info text
            if (currentStatus.value == "Stop Chirping") {
                Text(
                    text = "Recording in progress...",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.padding(top = 16.dp)
                )
            }
        }
    }

    private fun hasRecordPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED

    private fun requestMicrophonePermission() {
        requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
    }

    @SuppressLint("MissingPermission")
    private fun startChirpAndRecord() {
        if (job?.isActive == true) return

        recordingStartTime = System.currentTimeMillis()

        job = lifecycleScope.launch(Dispatchers.IO) {
            val label = labelState.value.text.trim()
            val distance = distanceState.value.text.trim()
            val angle = angleState.value.text.trim()
            val timestamp = System.currentTimeMillis()

            // Generate filename with settings
            val fileName = "chirp_${label}_${distance}cm_${angle}deg_$timestamp.pcm"

            val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            val file = File(downloadsDir, fileName)
            val fileOutputStream = FileOutputStream(file)

            val chirpSamples = generateChirpSamples()

            val recorder = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                AudioFormat.CHANNEL_IN_STEREO,
                AudioFormat.ENCODING_PCM_16BIT,
                AudioRecord.getMinBufferSize(
                    sampleRate,
                    AudioFormat.CHANNEL_IN_STEREO,
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

            val writerJob = launch {
                while (isActive) {
                    track.write(chirpSamples, 0, chirpSamples.size)
                }
            }

            try {
                val buffer = ByteArray(4096)
                while (isActive) {
                    val read = recorder.read(buffer, 0, buffer.size)
                    if (read > 0) {
                        fileOutputStream.write(buffer, 0, read)
                    }
                }
            } catch (e: Exception) {
                Log.e("ChirpApp", "Error during recording", e)
            } finally {
                withContext(NonCancellable) {
                    writerJob.cancelAndJoin()

                    try {
                        recorder.stop()
                    } catch (e: IllegalStateException) {
                        e.printStackTrace()
                    }
                    recorder.release()

                    try {
                        track.stop()
                    } catch (e: IllegalStateException) {
                        e.printStackTrace()
                    }
                    track.release()

                    try {
                        fileOutputStream.flush()
                        fileOutputStream.close()

                        // Calculate actual recording duration
                        val durationSeconds = (System.currentTimeMillis() - recordingStartTime) / 1000

                        Log.i("ChirpApp", "✅ PCM file saved to ${file.absolutePath}")
                        Log.i("ChirpApp", "Recording duration: ${durationSeconds}s")

                        // Show success message
                        withContext(Dispatchers.Main) {
                            Toast.makeText(
                                this@MainActivity,
                                "Saved: $fileName (${durationSeconds}s)",
                                Toast.LENGTH_LONG
                            ).show()
                        }
                    } catch (e: Exception) {
                        Log.e("ChirpApp", "Error saving file", e)
                    }
                }
            }
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
}