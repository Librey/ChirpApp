package com.example.chirpapp

import android.Manifest
//import android.content.Context
import android.content.pm.PackageManager
import android.media.*
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresPermission
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
//import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.*
import java.io.File
import java.io.FileOutputStream
import kotlin.math.PI
import kotlin.math.sin
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier


class MainActivity : ComponentActivity() {

    private val sampleRate = 44100
//    private val chirpStartHz = 18000.0
//    private val chirpEndHz = 20000.0
// New, audible values
    private val chirpStartHz = 400.0  // A low-mid pitch
    private val chirpEndHz = 4000.0 // A high-mid pitch

    private val chirpSeconds = 2

    // State for the UI
    private val status = mutableStateOf("Request Permission")
    private var job: Job? = null // To control the periodic task

    // Activity Result Launcher for RECORD_AUDIO permission
    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                status.value = "Start Chirping"
                Toast.makeText(this, "Permission granted", Toast.LENGTH_SHORT).show()
            } else {
                status.value = "Permission Denied"
                Toast.makeText(this, "Microphone permission is required for this app to work", Toast.LENGTH_LONG).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            val currentStatus = remember { status }

            // Use a Box to center its content
            Box(
                modifier = Modifier.fillMaxSize(), // Make the Box fill the entire screen
                contentAlignment = Alignment.Center // Align content (the Button) to the center
            )
            {
                Button(
                    onClick = {
                        when (currentStatus.value) {
                            "Start Chirping" -> {
                                startPeriodicChirps()
                                currentStatus.value = "Stop Chirping"
                            }
                            "Stop Chirping" -> {
                                stopPeriodicChirps()
                                currentStatus.value = "Start Chirping"
                            }
                            else -> {
                                // Initial state is to request permission
                                requestMicrophonePermission()
                            }
                        }
                    }
                )
                {
                    Text(currentStatus.value)
                }
            }
        }

        // Set initial button state based on current permission status
        if (hasRecordPermission()) {
            status.value = "Start Chirping"
        }
    }

    private fun hasRecordPermission() = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

    private fun requestMicrophonePermission() {
        requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
    }

    private fun startPeriodicChirps() {
        if (job?.isActive == true) return // Already running
        status.value = "Chirping..."
        // Launch a repeating job
        job = lifecycleScope.launch(Dispatchers.Default) {
            while (isActive) {
                Log.d("ChirpApp", "Starting a new chirp-and-record cycle.")
                val ok = runCatching {
                    playAndRecord()
                }.isSuccess


                withContext(Dispatchers.Main) {
                    if (!ok) {
                        Toast.makeText(this@MainActivity, "Cycle failed (check Logcat)", Toast.LENGTH_SHORT).show()
                    }
                }
                // Wait for 5 seconds before the next cycle
                delay(5000)
            }
        }
    }

    private fun stopPeriodicChirps() {
        job?.cancel()
        status.value = "Start Chirping"
        Log.d("ChirpApp", "Periodic chirping stopped by user.")
    }

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    private suspend fun playAndRecord() {
        val totalSamples = chirpSeconds * sampleRate
        val playbackMinBuf = AudioTrack.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val recordMinBuf = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)

        // Configure player
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
            .setBufferSizeInBytes(playbackMinBuf)
            .build()

        // Configure recorder
        val recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC, // Use the standard microphone input
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            recordMinBuf
        )


        val recordedData = ByteArray(totalSamples * 2) // 2 bytes per sample (16-bit)
        var totalBytesRead = 0

        try {
            if (track.state != AudioTrack.STATE_INITIALIZED) error("AudioTrack init failed")
            if (recorder.state != AudioRecord.STATE_INITIALIZED) error("AudioRecord init failed")

            // Generate chirp data on the fly
            val chirpShorts = ShortArray(totalSamples)
            for (i in 0 until totalSamples) {
                val t = i.toDouble() / sampleRate
                val f = chirpStartHz + (chirpEndHz - chirpStartHz) * (t / chirpSeconds)
                chirpShorts[i] = (sin(2.0 * PI * f * t) * 32767.0).toInt().toShort()
            }

            // Start playing and recording
            recorder.startRecording()
            track.play()

            // Main loop: write chirp data to speaker and read reflection from mic
            var samplesWritten = 0
            val writeChunkSize = (playbackMinBuf / 2).coerceAtMost(chirpShorts.size) // in shorts
            val writeByteBuffer = ByteArray(writeChunkSize * 2)
            val readByteBuffer = ByteArray(recordMinBuf)

            while (samplesWritten < chirpShorts.size) {
                // Write a chunk to the speaker
                val shortsToWrite = (chirpShorts.size - samplesWritten).coerceAtMost(writeChunkSize)
                for (i in 0 until shortsToWrite) {
                    val s = chirpShorts[samplesWritten + i].toInt()
                    writeByteBuffer[i * 2] = (s and 0xFF).toByte()
                    writeByteBuffer[i * 2 + 1] = ((s shr 8) and 0xFF).toByte()
                }
                track.write(writeByteBuffer, 0, shortsToWrite * 2)

                // Read a chunk from the microphone
                val bytesRead = recorder.read(readByteBuffer, 0, readByteBuffer.size)
                if (bytesRead > 0 && totalBytesRead + bytesRead <= recordedData.size) {
                    System.arraycopy(readByteBuffer, 0, recordedData, totalBytesRead, bytesRead)
                    totalBytesRead += bytesRead
                }
                samplesWritten += shortsToWrite
            }

            // After chirp finishes, keep recording for a moment to capture reflections
            delay(200) // e.g., 200ms
            val bytesRead = recorder.read(readByteBuffer, 0, readByteBuffer.size)
            if (bytesRead > 0 && totalBytesRead + bytesRead <= recordedData.size) {
                System.arraycopy(readByteBuffer, 0, recordedData, totalBytesRead, bytesRead)
                totalBytesRead += bytesRead
            }


            Log.d("ChirpApp", "Playback and recording finished. Total bytes recorded: $totalBytesRead")
            saveAsWav(recordedData.sliceArray(0 until totalBytesRead))

        } finally {
            // Clean up resources
            if (track.playState == AudioTrack.PLAYSTATE_PLAYING) track.stop()
            track.release()
            if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) recorder.stop()
            recorder.release()
        }
    }

    private fun saveAsWav(data: ByteArray) {
        val fileName = "chirp_reflection_${System.currentTimeMillis()}.wav"
        val file = File(getExternalFilesDir(null), fileName) // App-specific external storage

        FileOutputStream(file).use { out ->
            // WAV header
            val channels = 1
            val bitDepth = 16
            val byteRate = sampleRate * channels * bitDepth / 8
            val totalDataLen = data.size.toLong()
            val totalWavLen = totalDataLen + 36

            val header = byteArrayOf(
                'R'.code.toByte(), 'I'.code.toByte(), 'F'.code.toByte(), 'F'.code.toByte(), // RIFF
                (totalWavLen and 0xff).toByte(),
                (totalWavLen shr 8 and 0xff).toByte(),
                (totalWavLen shr 16 and 0xff).toByte(),
                (totalWavLen shr 24 and 0xff).toByte(),
                'W'.code.toByte(), 'A'.code.toByte(), 'V'.code.toByte(), 'E'.code.toByte(), // WAVE
                'f'.code.toByte(), 'm'.code.toByte(), 't'.code.toByte(), ' '.code.toByte(), // fmt chunk
                16, 0, 0, 0, // 16 for PCM
                1, 0, // PCM
                channels.toByte(), 0, // Mono
                (sampleRate and 0xff).toByte(),
                (sampleRate shr 8 and 0xff).toByte(),
                (sampleRate shr 16 and 0xff).toByte(),
                (sampleRate shr 24 and 0xff).toByte(),
                (byteRate and 0xff).toByte(),
                (byteRate shr 8 and 0xff).toByte(),
                (byteRate shr 16 and 0xff).toByte(),
                (byteRate shr 24 and 0xff).toByte(),
                (channels * bitDepth / 8).toByte(), 0, // block align
                bitDepth.toByte(), 0, // bits per sample
                'd'.code.toByte(), 'a'.code.toByte(), 't'.code.toByte(), 'a'.code.toByte(), // data chunk
                (totalDataLen and 0xff).toByte(),
                (totalDataLen shr 8 and 0xff).toByte(),
                (totalDataLen shr 16 and 0xff).toByte(),
                (totalDataLen shr 24 and 0xff).toByte()
            )
            out.write(header)
            out.write(data)
        }
        Log.i("ChirpApp", "Successfully saved WAV file to ${file.absolutePath}")
    }
}
