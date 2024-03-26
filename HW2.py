# DAAP Homework 2
# students and students code:
# Chiara Lunghi     233195
# Alice Portentoso  232985

import numpy as np
import wave
import math
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn.cluster import KMeans
import seaborn as sns

window_size = 512
fft_size = window_size
hop_size = 256 # 0.5 * 512
frame_size = 1024

# Load the audio file and generate a NumPy array that contains its normalized samples
def load_signal(filename):

    with wave.open(filename, 'rb') as f:

        # Get audio parameters
        global num_sample, rate
        num_sample = f.getparams().nframes
        rate = f.getparams().framerate

        # Read frames and convert into array
        frames = f.readframes(num_sample)
        signal = np.frombuffer(frames, dtype=np.int16)

        # Normalize the signal
        signal = signal / np.float32(32767)

    return signal

# Calculate STFT
def stft(signal):

    global window
    window = np.hanning(window_size)
    global num_frames
    num_frames = 1 + int(np.floor((len(signal) - window_size) / hop_size))

    padded_signal_length = (num_frames - 1) * hop_size + window_size
    padded_signal = np.append(signal, np.zeros(padded_signal_length))

    # Calculate STFT matrix: frames array in frequency
    stft_matrix = np.empty((num_frames, fft_size), dtype=np.complex64)
    for i in range(num_frames):
        segment = padded_signal[i * hop_size:i * hop_size + window_size]
        windowed_segment = segment * window #156
        stft_matrix[i,:] = np.fft.fft(windowed_segment, axis=0, n=fft_size)
    return stft_matrix

# ISTFT
def istft(X):
    # Apply IFFT for each window of frequency
    x_frames = np.zeros((num_frames - 1) * hop_size + fft_size)
    for i in range(num_frames):
        x_frame = np.fft.ifft(X[i,:]).real
        x_frames[i * hop_size: i * hop_size + fft_size] += x_frame[:fft_size]
    return x_frames

# Feature vectors
def f_vect(signal_1_stft, signal_2_stft):
    f_vector = []

    for i in range(num_frames):
        global A1_frame, A2_frame, P_frame
        if (i == 50):
            A1_frame = []
            A2_frame = []
            P_frame = []
        for f in range(fft_size):

            # Compute the power spectrogram of the mixture signals y1, y2
            y1_power = np.abs(signal_1_stft[i][f])**2
            y2_power = np.abs(signal_2_stft[i][f])**2

            # Create the vector starting from its components
            A1 = np.abs(signal_1_stft[i][f])/np.sqrt((y1_power + y2_power)) #dim window_size * num_frames
            A2 = np.abs(signal_2_stft[i][f])/np.sqrt((y1_power + y2_power))
            P = (1/(2 * math.pi))*np.angle(signal_2_stft[i][f]/signal_1_stft[i][f])
            f_vector.append(np.array([A1, A2, P]))
            if (i == 50):
                A1_frame.append(A1)
                A2_frame.append(A2)
                P_frame.append(P)
    return f_vector

# Plot of a log-amplitude spectrogram
def plot_spect(estimated_signal, name_estimated, true_signal, name_true):

    # Denormalized signal
    for i in range(len(true_signal)):
        true_signal[i] = true_signal[i] * np.float32(32767)

    plt.figure(figsize=(12, 6))

    # Plot spectogram of calculated separated signal
    plt.subplot(2, 1, 1)
    plt.title('Spectrogram of ' + name_estimated)
    plt.specgram(estimated_signal, NFFT=window_size, Fs=rate, noverlap=hop_size, window=window, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')

    # Plot spectogram of original sound
    plt.subplot(2, 1, 2)
    plt.title('Spectrogram of ' + name_true)
    plt.specgram(true_signal, NFFT=window_size, Fs=rate, noverlap=hop_size, window=window, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')

    plt.tight_layout()
    plt.show()

# Plot of a binary mask in black and white
def plot_bm(mask_1, name_mask_1, mask_2, name_mask_2, mask_3, name_mask_3):

    plt.figure(figsize=(9,9))

    plt.subplot(3, 1, 1)
    plt.imshow(np.transpose(mask_1), cmap='gray')
    plt.title('Plot of ' + name_mask_1)
    plt.xlabel('Frame')
    plt.ylabel('Frequency bin')
    plt.gca().set_aspect('auto')

    plt.subplot(3, 1, 2)
    plt.imshow(np.transpose(mask_2), cmap='gray')
    plt.title('Plot of ' + name_mask_2)
    plt.xlabel('Frame')
    plt.ylabel('Frequency bin')
    plt.gca().set_aspect('auto')

    plt.subplot(3, 1, 3)
    plt.imshow(np.transpose(mask_3), cmap='gray')
    plt.title('Plot of ' + name_mask_3)
    plt.xlabel('Frame')
    plt.ylabel('Frequency bin')
    plt.gca().set_aspect('auto')

    plt.tight_layout()
    plt.show()

# Plot of density plot and correspective histograms
def d_plot_and_hist(vect_1, vect_2, name_vect_1, name_vect_2):

    sns.set_style('white')
    sns.jointplot(x=vect_1, y=vect_2, kind='kde', space=0.6, fill=True, cmap='Blues')
    plt.title('Plot of ' + name_vect_1 + ' and ' +  name_vect_2)
    plt.show()


#### MAIN ####

# Take the mixture signals
y1 = load_signal('y1.wav')
y2 = load_signal('y2.wav')
# Take the true results only to use them in the final graphs
s1 = load_signal('s1.wav')
s2 = load_signal('s2.wav')
s3 = load_signal('s3.wav')

# Signals in time-frequency:
y1_stft = stft(y1)
y2_stft = stft(y2)
s1_stft = stft(s1)
s2_stft = stft(s2)
s3_stft = stft(s3)

# Vector of feature vectors (3D)
f_vector = f_vect(y1_stft, y2_stft)

# Clustering: kmeans
kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(f_vector)
labels = kmeans.labels_

# Reshape labels in 2D
labels_2D = labels.reshape(len(y1_stft),fft_size)

# Compute binary Mask
binary_mask_1 = np.zeros((num_frames,fft_size))
binary_mask_2 = np.zeros((num_frames,fft_size))
binary_mask_3 = np.zeros((num_frames,fft_size))

for i in range(num_frames):
    for f in range(fft_size):
        if labels_2D[i][f] == 0:
            binary_mask_1[i][f] = 1
        elif labels_2D[i][f] == 1:
            binary_mask_2[i][f] = 1
        elif labels_2D[i][f] == 2:
            binary_mask_3[i][f] = 1

# Calculate the separated signals by multiplying binary mask and y1, and do istft
output1 = istft(np.multiply(binary_mask_1, y1_stft))
output2 = istft(np.multiply(binary_mask_2, y1_stft))
output3 = istft(np.multiply(binary_mask_3, y1_stft))

# Denormalize signals
for i in range(len(output1)):
    output1[i] = output1[i] * np.float32(32767)
    output2[i] = output2[i] * np.float32(32767)
    output3[i] = output3[i] * np.float32(32767)

# Save final signal
wavfile.write("Lunghi_Portentoso_s1_hat.wav", rate, output1.astype(np.int16))
wavfile.write("Lunghi_Portentoso_s2_hat.wav", rate, output2.astype(np.int16))
wavfile.write("Lunghi_Portentoso_s3_hat.wav", rate, output3.astype(np.int16))

#PLOTS

# Plot spectrograms with differences between estimate and true signals
plot_spect(output1, "s1 estimate", s1, "s1 true")
plot_spect(output2, "s2 estimate", s2, "s2 true")
plot_spect(output3, "s3 estimate", s3, "s3 true")

# Binary Mask
plot_bm(binary_mask_1, "Binary mask 1", binary_mask_2, "Binary mask 2", binary_mask_3, "Binary mask 3")

# Density plots and correspective histograms of the 3 masks
d_plot_and_hist(A1_frame, A2_frame, 'A1', 'A2')
d_plot_and_hist(A1_frame, P_frame, 'A1', 'P')
d_plot_and_hist(A2_frame, P_frame, 'A2', 'P')
