package com.example.chirpapp

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.pm.PackageManager
import android.media.*
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.TextRange
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.*
import java.io.File
import java.io.OutputStream
import kotlin.math.PI
import kotlin.math.sin

class MainActivity : ComponentActivity() {

    // Audio params
    private val sampleRate = 44100
    private val chirpStartHz = 18000.0
    private val chirpEndHz = 20000.0

    // FIXED protocol
    private val fixedSetting = 2                 // Setting 2 only
    private val continuousSec = 30f              // ✅ 30s only

    // UI / state
    private val status = mutableStateOf("Request Permission")
    private var job: Job? = null

    private val labelState = mutableStateOf(TextFieldValue("chips"))
    private val distanceState = mutableStateOf(TextFieldValue("30"))
    private val angleState = mutableStateOf(TextFieldValue("120"))

    private val servingCounter = mutableStateOf(1)
    private var lastLabel = "chips"

    // Optional IRB naming toggle + fields
    private var useIrbNaming by mutableStateOf(false)
    private val institutionState = mutableStateOf(TextFieldValue("1"))   // 1=GSU, 2=SDSU, 3=TAMUSA
    private val participantIdState = mutableStateOf(TextFieldValue("001"))
    private val categoryCodeState = mutableStateOf(TextFieldValue("0"))  // 0=idle, 1–10

    // Timestamp markers
    @Volatile private var startUnixMs: Long = 0L
    @Volatile private var stopEatingMarked: Boolean = false
    @Volatile private var stopEatingUnixMs: Long = -1L
    @Volatile private var bytesWrittenSoFar: Long = 0L

    // ✅ Countdown timer (seconds remaining)
    private val remainingSeconds = mutableStateOf(0)

    // Permission
    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                status.value = "Start Recording"
                Toast.makeText(this, "Permission granted", Toast.LENGTH_SHORT).show()
            } else {
                status.value = "Permission Denied"
                Toast.makeText(this, "Microphone permission is required", Toast.LENGTH_LONG).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) { ChirpAppUI() }
            }
        }

        if (hasRecordPermission()) status.value = "Start Recording"
    }

    @Composable
    private fun ChirpAppUI() {

        val currentStatus = remember { status }
        val label = remember { labelState }
        val distance = remember { distanceState }
        val angle = remember { angleState }
        val serving = remember { servingCounter }

        val isRecording = currentStatus.value == "Recording..."
        val scrollState = rememberScrollState()

        Column(
            modifier = Modifier
                .fillMaxSize()
                .safeDrawingPadding()
                .padding(24.dp)
        ) {

            // =======================
            // SCROLLABLE CONTENT AREA
            // =======================
            Column(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .verticalScroll(scrollState),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {

                Text(
                    text = "ChirpApp - Study Capture",
                    style = MaterialTheme.typography.headlineMedium,
                    modifier = Modifier.padding(bottom = 16.dp)
                )

                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 12.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.secondaryContainer
                    )
                ) {
                    Text(
                        text = "Fixed: Setting 2 (1000ms chirp, 500ms gap) • Recording: 30s • Stereo 16‑bit PCM @ 44.1kHz",
                        style = MaterialTheme.typography.titleSmall,
                        modifier = Modifier.padding(12.dp)
                    )
                }

                // IRB toggle
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 8.dp)
                ) {
                    Checkbox(
                        checked = useIrbNaming,
                        onCheckedChange = { if (!isRecording) useIrbNaming = it },
                        enabled = !isRecording
                    )
                    Text("Use IRB filename (W_XXX_Y_ZZ.pcm)")
                }

                // IRB fields
                if (useIrbNaming) {

                    OutlinedTextField(
                        value = institutionState.value,
                        onValueChange = { institutionState.value = it },
                        label = { Text("Institution ID (1=GSU, 2=SDSU, 3=TAMUSA)") },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(bottom = 6.dp),
                        enabled = !isRecording
                    )

                    OutlinedTextField(
                        value = participantIdState.value,
                        onValueChange = { participantIdState.value = it },
                        label = { Text("Participant ID (XXX, e.g., 019)") },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(bottom = 6.dp),
                        enabled = !isRecording
                    )

                    // ✅ Digits-only AND editable (cursor stays at end)
                    OutlinedTextField(
                        value = categoryCodeState.value,
                        onValueChange = { tfv ->
                            val digitsOnly = tfv.text.filter { it.isDigit() }
                            categoryCodeState.value = TextFieldValue(
                                text = digitsOnly,
                                selection = TextRange(digitsOnly.length)
                            )
                        },
                        label = { Text("Category Code (0=idle, 1–10)") },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(bottom = 8.dp),
                        enabled = !isRecording
                    )
                }

                // Pilot / logging fields (kept)
                OutlinedTextField(
                    value = label.value,
                    onValueChange = {
                        label.value = it
                        if (it.text.trim() != lastLabel) {
                            serving.value = 1
                            lastLabel = it.text.trim()
                        }
                    },
                    label = { Text("Food Label (e.g., chips, carrots)") },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 6.dp),
                    enabled = !isRecording
                )

                OutlinedTextField(
                    value = distance.value,
                    onValueChange = { distance.value = it },
                    label = { Text("Distance (cm)") },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 6.dp),
                    enabled = !isRecording
                )

                OutlinedTextField(
                    value = angle.value,
                    onValueChange = { angle.value = it },
                    label = { Text("Angle (degrees)") },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 12.dp),
                    enabled = !isRecording
                )

                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 16.dp),
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

                // ✅ Recording status + countdown timer
                if (isRecording) {
                    Text(
                        text = "Recording… auto‑stops at 30s",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.padding(top = 8.dp)
                    )

                    Text(
                        text = "Time left: ${remainingSeconds.value} s",
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.padding(top = 4.dp)
                    )

                    val progress = (remainingSeconds.value.toFloat() / continuousSec)
                        .coerceIn(0f, 1f)

                    LinearProgressIndicator(
                        progress = progress,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 6.dp)
                    )

                    if (stopEatingMarked) {
                        Text(
                            text = "Stop‑eating marked: tail saved as *_idleTail.pcm",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.secondary,
                            modifier = Modifier.padding(top = 6.dp)
                        )
                    }
                }

                Spacer(Modifier.height(8.dp))
            }

            // =======================
            // FIXED BOTTOM BUTTONS
            // =======================
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .imePadding()
            ) {

                Button(
                    onClick = {
                        when (currentStatus.value) {
                            "Start Recording" -> {
                                if (label.value.text.isBlank()) {
                                    Toast.makeText(
                                        this@MainActivity,
                                        "Please enter a food label",
                                        Toast.LENGTH_SHORT
                                    ).show()
                                    return@Button
                                }
                                startChirpAndRecordFixed()
                                currentStatus.value = "Recording..."
                            }
                            else -> requestMicrophonePermission()
                        }
                    },
                    enabled = (
                            currentStatus.value == "Start Recording" ||
                                    currentStatus.value == "Request Permission" ||
                                    currentStatus.value == "Permission Denied"
                            ),
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(56.dp)
                ) {
                    Text(if (currentStatus.value == "Recording...") "Recording..." else currentStatus.value)
                }

                Spacer(Modifier.height(12.dp))

                OutlinedButton(
                    onClick = { markStopEating() },
                    enabled = isRecording && !stopEatingMarked,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(52.dp)
                ) {
                    Text(if (!stopEatingMarked) "Mark Stop‑Eating (save tail as idle)" else "Stop‑Eating Marked")
                }
            }
        }
    }

    private fun hasRecordPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED

    private fun requestMicrophonePermission() {
        requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
    }

    private fun markStopEating() {
        stopEatingMarked = true
        stopEatingUnixMs = System.currentTimeMillis()
        Toast.makeText(this, "Marked stop-eating. Tail will be saved as idle PCM.", Toast.LENGTH_SHORT).show()
    }

    @SuppressLint("MissingPermission")
    private fun startChirpAndRecordFixed() {
        if (job?.isActive == true) return

        // reset markers
        startUnixMs = System.currentTimeMillis()
        stopEatingMarked = false
        stopEatingUnixMs = -1L
        bytesWrittenSoFar = 0L

        // ✅ start countdown at 30
        remainingSeconds.value = continuousSec.toInt()

        job = lifecycleScope.launch(Dispatchers.IO) {

            // ✅ timer job updates UI state once per second (on Main)
            val timerJob = launch(Dispatchers.Main) {
                while (isActive && remainingSeconds.value > 0) {
                    delay(1000)
                    remainingSeconds.value -= 1
                }
            }

            val label = labelState.value.text.trim()
            val distance = distanceState.value.text.trim()
            val angle = angleState.value.text.trim()
            val serving = String.format("%02d", servingCounter.value)
            val timestamp = startUnixMs

            val baseName = if (useIrbNaming) {
                val W = institutionState.value.text.trim().ifEmpty { "1" }
                val XXX = participantIdState.value.text.trim().padStart(3, '0')
                val rawY = categoryCodeState.value.text.trim()
                val yInt = rawY.toIntOrNull() ?: 0
                val Y = yInt.coerceIn(0, 10).toString().padStart(2, '0')
                val ZZ = serving
                "${W}_${XXX}_${Y}_${ZZ}"
            } else {
                "chirp_${label}_${distance}cm_${angle}deg_c30_s${serving}_set${fixedSetting}_$timestamp"
            }

            val mainName = "$baseName.pcm"
            val idleTailName = "${baseName}_idleTail.pcm"
            val metaName = "${baseName}_meta.json"

            val (mainOut, mainUri) = createDownloadOutput(mainName, "application/octet-stream")
            val (idleOut, idleUri) = createDownloadOutput(idleTailName, "application/octet-stream")
            val (metaOut, metaUri) = createDownloadOutput(metaName, "application/json")

            val chirpSamples = generateChirpSamplesWithGaps(fixedSetting)
            saveReferenceChirp(chirpSamples, "chirp_reference_set${fixedSetting}.pcm")


            val minBuf = AudioRecord.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_IN_STEREO,
                AudioFormat.ENCODING_PCM_16BIT
            )

            val recorder = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                AudioFormat.CHANNEL_IN_STEREO,
                AudioFormat.ENCODING_PCM_16BIT,
                minBuf
            )

            val trackMinBuf = AudioTrack.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_16BIT
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
                .setBufferSizeInBytes(maxOf(trackMinBuf, chirpSamples.size))
                .build()

            recorder.startRecording()
            track.play()

            val chirpWriterJob = launch {
                while (isActive) {
                    track.write(chirpSamples, 0, chirpSamples.size)
                }
            }

            var totalWritten = 0L
            val bytesPerFrame = 2 /*bytes*/ * 2 /*stereo*/
            val targetBytes = (continuousSec * sampleRate * bytesPerFrame).toLong()

            try {
                val buffer = ByteArray(4096)

                while (isActive) {
                    val read = recorder.read(buffer, 0, buffer.size)
                    if (read > 0) {
                        mainOut.write(buffer, 0, read)
                        totalWritten += read
                        bytesWrittenSoFar = totalWritten

                        if (stopEatingMarked) {
                            idleOut.write(buffer, 0, read)
                        }

                        if (totalWritten >= targetBytes) break
                    }
                }
            } catch (e: Exception) {
                Log.e("ChirpApp", "Error during recording", e)
            } finally {
                withContext(NonCancellable) {
                    try { chirpWriterJob.cancelAndJoin() } catch (_: Exception) {}

                    try { recorder.stop() } catch (_: Exception) {}
                    recorder.release()

                    try { track.stop() } catch (_: Exception) {}
                    track.release()

                    try { mainOut.flush(); mainOut.close() } catch (_: Exception) {}
                    try { idleOut.flush(); idleOut.close() } catch (_: Exception) {}

                    val endUnixMs = System.currentTimeMillis()
                    val metaJson = """
                    {
                      "baseName": "$baseName",
                      "mainPcm": "$mainName",
                      "idleTailPcm": "$idleTailName",
                      "sampleRate": $sampleRate,
                      "channels": 2,
                      "bitsPerSample": 16,
                      "chirpSetting": $fixedSetting,
                      "durationSecFixed": 30,
                      "startUnixMs": $startUnixMs,
                      "endUnixMs": $endUnixMs,
                      "stopEatingUnixMs": $stopEatingUnixMs,
                      "note": "idleTailPcm contains audio after stopEatingUnixMs (if marked)."
                    }
                    """.trimIndent()

                    try {
                        metaOut.write(metaJson.toByteArray())
                        metaOut.flush()
                        metaOut.close()
                    } catch (_: Exception) {}

                    finalizeDownload(mainUri)
                    finalizeDownload(idleUri)
                    finalizeDownload(metaUri)

                    // ✅ stop timer + reset remainingSeconds
                    try { timerJob.cancel() } catch (_: Exception) {}
                    withContext(Dispatchers.Main) {
                        remainingSeconds.value = 0
                        Toast.makeText(
                            this@MainActivity,
                            "Saved: $mainName (+ $idleTailName)",
                            Toast.LENGTH_LONG
                        ).show()

                        servingCounter.value += 1
                        status.value = "Start Recording"
                    }
                }
            }
        }
    }
    private fun saveReferenceChirp(samples: ByteArray, filename: String) {
        try {
            val (out, uri) = createDownloadOutput(filename, "application/octet-stream")
            out.write(samples)
            out.flush()
            out.close()
            finalizeDownload(uri)
            Log.d("ChirpApp", "Reference chirp saved: $filename")
        } catch (e: Exception) {
            Log.e("ChirpApp", "Failed to save reference chirp", e)
        }
    }


    private fun createDownloadOutput(displayName: String, mimeType: String): Pair<OutputStream, Uri> {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, displayName)
                put(MediaStore.MediaColumns.MIME_TYPE, mimeType)
                put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/ChirpApp")
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
            val uri = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                ?: throw IllegalStateException("Failed to create MediaStore entry for $displayName")
            val out = contentResolver.openOutputStream(uri)
                ?: throw IllegalStateException("Failed to open output stream for $displayName")
            Pair(out, uri)
        } else {
            val dir = File(getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS), "ChirpApp")
            if (!dir.exists()) dir.mkdirs()
            val file = File(dir, displayName)
            val uri = Uri.fromFile(file)
            val out = file.outputStream()
            Pair(out, uri)
        }
    }

    private fun finalizeDownload(uri: Uri) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            try {
                val values = ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }
                contentResolver.update(uri, values, null, null)
            } catch (_: Exception) {}
        }
    }

    private fun generateChirpSamplesWithGaps(setting: Int): ByteArray {
        val (chirpDurationMs, gapDurationMs) = when (setting) {
            1 -> 500 to 250
            2 -> 1000 to 500
            else -> 1000 to 500
        }

        val chirpSamples = (chirpDurationMs * sampleRate) / 1000
        val gapSamples = (gapDurationMs * sampleRate) / 1000
        val totalSamples = chirpSamples + gapSamples

        val cycleShorts = ShortArray(totalSamples)

        for (i in 0 until chirpSamples) {
            val t = i.toDouble() / sampleRate
            val freq = chirpStartHz + (chirpEndHz - chirpStartHz) * (t / (chirpDurationMs / 1000.0))
            val amp = (sin(2 * PI * freq * t) * 32767.0).toInt()
                .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
            cycleShorts[i] = amp.toShort()
        }

        // PCM 16-bit little-endian (mono for playback)
        return ByteArray(cycleShorts.size * 2).apply {
            for (i in cycleShorts.indices) {
                val s = cycleShorts[i].toInt()
                this[i * 2] = (s and 0xFF).toByte()
                this[i * 2 + 1] = ((s shr 8) and 0xFF).toByte()
            }
        }
    }
}