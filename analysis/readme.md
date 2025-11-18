# SenseEat Chirp Analysis

Hey team 

This folder has the Python script and a few example recordings we’ll use to analyze the ultrasonic chirp reflections from the app.

SETUP

1. Make sure you’ve got Python 3 installed.
   Check with:

   ```bash
   python --version
   ```

2. Install the needed libraries (you only do this once):

   ```bash
   pip install numpy matplotlib scipy
   ```
HOW TO RUN

1. Drop your `.wav` or `.pcm` recording in this same folder.
2. Open the terminal in VS Code or right inside the folder.
3. Run:

   bash
   python signal_analysis.py
   

The script will generate four graphs:

time_domain.png → raw waveform
fft_signal.png → frequency content
filtered_signal.png → ultrasonic range (18–20 kHz)
spectrogram.png → frequency vs time view

All graphs save automatically in this folder.

NOTES

The script filters out everything below **18 kHz**, so no voice or background sound shows up.
To test a new recording, just update the `filename` at the top of the script.
`.pcm` files are raw audio, but the script already handles them fine.
Keeps the analysis clean, private, and focused strictly on the chirp signal.