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
import androidx.compose.foundation.selection.selectable
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

    private val status = mutableStateOf("Request Permission")
    private var job: Job? = null
    private var recordingStartTime = 0L

    // Recording settings
    private val labelState = mutableStateOf(TextFieldValue("chips"))
    private val distanceState = mutableStateOf(TextFieldValue("30"))
    private val angleState = mutableStateOf(TextFieldValue("120"))

    // NEW: Setting selection (1 or 2)
    private val selectedSetting = mutableStateOf(2)  // Default: Setting 2

    // NEW: Serving counter
    private val servingCounter = mutableStateOf(1)
    private var lastLabel = "chips"  // Track label changes

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                status.value = "Start Recording"
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

        if (hasRecordPermission()) status.value = "Start Recording"
    }

    @Composable
    fun ChirpAppUI() {
        val currentStatus = remember { status }
        val label = remember { labelState }
        val distance = remember { distanceState }
        val angle = remember { angleState }
        val setting = remember { selectedSetting }
        val serving = remember { servingCounter }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = "ChirpApp - Pilot Study",
                style = MaterialTheme.typography.headlineMedium,
                modifier = Modifier.padding(bottom = 24.dp)
            )

            // NEW: Setting Selection
            Text(
                text = "Chirp Setting:",
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 8.dp)
            )

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 16.dp)
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .selectable(
                            selected = (setting.value == 1),
                            onClick = {
                                if (currentStatus.value == "Start Recording") {
                                    setting.value = 1
                                }
                            }
                        )
                        .padding(vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    RadioButton(
                        selected = (setting.value == 1),
                        onClick = {
                            if (currentStatus.value == "Start Recording") {
                                setting.value = 1
                            }
                        },
                        enabled = currentStatus.value == "Start Recording"
                    )
                    Text(
                        text = "Setting 1 (500ms chirp, 250ms gap)",
                        modifier = Modifier.padding(start = 8.dp)
                    )
                }

                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .selectable(
                            selected = (setting.value == 2),
                            onClick = {
                                if (currentStatus.value == "Start Recording") {
                                    setting.value = 2
                                }
                            }
                        )
                        .padding(vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    RadioButton(
                        selected = (setting.value == 2),
                        onClick = {
                            if (currentStatus.value == "Start Recording") {
                                setting.value = 2
                            }
                        },
                        enabled = currentStatus.value == "Start Recording"
                    )
                    Text(
                        text = "Setting 2 (1000ms chirp, 500ms gap)",
                        modifier = Modifier.padding(start = 8.dp)
                    )
                }
            }

            Divider(modifier = Modifier.padding(vertical = 16.dp))

            // Label input
            OutlinedTextField(
                value = label.value,
                onValueChange = {
                    label.value = it
                    // Reset counter if label changed
                    if (it.text.trim() != lastLabel) {
                        serving.value = 1
                        lastLabel = it.text.trim()
                    }
                },
                label = { Text("Food Label (e.g., chips, carrots)") },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 12.dp),
                enabled = currentStatus.value == "Start Recording"
            )

            // Distance input
            OutlinedTextField(
                value = distance.value,
                onValueChange = { distance.value = it },
                label = { Text("Distance (cm)") },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 12.dp),
                enabled = currentStatus.value == "Start Recording"
            )

            // Angle input
            OutlinedTextField(
                value = angle.value,
                onValueChange = { angle.value = it },
                label = { Text("Angle (degrees)") },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 16.dp),
                enabled = currentStatus.value == "Start Recording"
            )

            // NEW: Serving Counter Display
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 24.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            ) {
                Text(
                    text = "Current Serving: #${String.format("%02d", serving.value)}",
                    style = MaterialTheme.typography.titleLarge,
                    modifier = Modifier.padding(16.dp)
                )
            }

            // Main button
            Button(
                onClick = {
                    when (currentStatus.value) {
                        "Start Recording" -> {
                            // Validate inputs
                            if (label.value.text.isBlank()) {
                                Toast.makeText(
                                    this@MainActivity,
                                    "Please enter a food label",
                                    Toast.LENGTH_SHORT
                                ).show()
                                return@Button
                            }
                            startChirpAndRecord()
                            currentStatus.value = "Stop Recording"
                        }
                        "Stop Recording" -> {
                            stopChirpAndRecord()
                            // Auto-increment serving counter
                            serving.value += 1
                            currentStatus.value = "Start Recording"
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

            // Status text
            if (currentStatus.value == "Stop Recording") {
                Text(
                    text = "Recording serving #${String.format("%02d", serving.value)}...",
                    style = MaterialTheme.typography.bodyMedium,
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
            val setting = selectedSetting.value
            val serving = String.format("%02d", servingCounter.value)
            val timestamp = System.currentTimeMillis()

            // NEW: Filename with serving number
            val fileName = "chirp_${label}_${distance}cm_${angle}deg_s${serving}_set${setting}_$timestamp.pcm"

            val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            val file = File(downloadsDir, fileName)
            val fileOutputStream = FileOutputStream(file)

            // NEW: Generate chirps based on selected setting
            val chirpSamples = generateChirpSamplesWithGaps(setting)

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

                        val durationSeconds = (System.currentTimeMillis() - recordingStartTime) / 1000

                        Log.i("ChirpApp", "✅ Saved: $fileName")
                        Log.i("ChirpApp", "Duration: ${durationSeconds}s, Serving: $serving, Setting: $setting")

                        withContext(Dispatchers.Main) {
                            Toast.makeText(
                                this@MainActivity,
                                "Saved serving #$serving (${durationSeconds}s)",
                                Toast.LENGTH_SHORT
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

    // NEW: Generate chirps with gaps based on setting
    private fun generateChirpSamplesWithGaps(setting: Int): ByteArray {
        // Setting parameters
        val (chirpDurationMs, gapDurationMs) = when (setting) {
            1 -> Pair(500, 250)   // Setting 1: 500ms chirp, 250ms gap
            2 -> Pair(1000, 500)  // Setting 2: 1000ms chirp, 500ms gap
            else -> Pair(1000, 500)
        }

        val chirpDurationSamples = (chirpDurationMs * sampleRate) / 1000
        val gapDurationSamples = (gapDurationMs * sampleRate) / 1000
        val cycleSamples = chirpDurationSamples + gapDurationSamples

        // Generate one chirp + gap cycle
        val cycleShorts = ShortArray(cycleSamples)

        // Generate chirp part (first chirpDurationSamples)
        for (i in 0 until chirpDurationSamples) {
            val t = i.toDouble() / sampleRate
            val freq = chirpStartHz + (chirpEndHz - chirpStartHz) * (t / (chirpDurationMs / 1000.0))
            val amplitude = (sin(2 * PI * freq * t) * 32767.0).toInt()
                .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
            cycleShorts[i] = amplitude.toShort()
        }

        // Gap part is already 0 (silence) from ShortArray initialization

        // Convert to ByteArray
        return ByteArray(cycleShorts.size * 2).apply {
            for (i in cycleShorts.indices) {
                val s = cycleShorts[i].toInt()
                this[i * 2] = (s and 0xFF).toByte()
                this[i * 2 + 1] = ((s shr 8) and 0xFF).toByte()
            }
        }
    }
}