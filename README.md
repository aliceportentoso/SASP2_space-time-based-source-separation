### Music and Acoustic Engineering - Politecnico di Milano 
## Sound Analysis, Synthesis and Processing 2023 - Homework 2
## Space-Time-Based Source Separation
<br>
For this homework assignment, you will be exploring the topic of acoustic source separation.<br>
You will be provided with two mixture signals 𝑦1(𝑛) and 𝑦2(𝑛), corresponding to files y1.wav and y2.wav, respectively. These signals have been acquired using with 𝑀 = 2 ideal microphones placed at 𝑑 = 9 cm from one another. Each microphone captures the mixture of 𝑁 = 3 speech signals, i.e., 𝑠1(𝑛), 𝑠2(𝑛), and 𝑠3(𝑛), having a sampling frequency of 8 kHz. 
<br><br>
The speakers are located at 𝜃1 = 30°, 𝜃2 = 85°, and 𝜃3 = −40°, respectively. These directions of arrival (DOAs) are referred to the
normal direction to the line passing through the two microphones (similarly to the ULA configuration.) All speakers are located at 75 cm from the reference microphone, i.e., microphone 1. Each microphone is characterized by a i.i.d. Gaussian self-noise, which is uncorrelated with the source signals. For this homework, we
will also assume that i) reverberation is absent, ii) the sensor noise has a standard deviation of 𝜎 = 10−3 , and iii) the speed of sound is 𝑐 = 340 m/s.
<br> <br>
You will implement source separation by applying binary masking to the short-time Fourier transform (STFT) of the mixture signal 𝑦1(𝑛). To design the mask, you will first extract a feature vector for each of the time-frequency locations (𝑚, 𝜔𝑘) of the STFT. Then, you will apply a clustering algorithm (e.g., 𝑘𝑘-means) with 𝑘 = 𝑁 = 3 clusters. Namely, the cluster index associated to the pair (𝑚, 𝜔𝑘) will determine whether the corresponding time-frequency location in the binary mask 𝑀𝑀ℓ(𝑚𝑚, 𝜔𝜔𝑘𝑘) should be set to either 1 or 0. In other words, this corresponds to
<br>
𝑀ℓ(𝑚, 𝜔𝑘) = 1 if (𝑚, 𝜔𝑘) belongs to cluster ℓ, 0 otherwise.
<br><br>
Having designed the three binary masks, the estimate of the source signals can be retrieved by taking the Hadamard product between the corresponding mask and the STFT of the reference microphone signal, i.e., 𝑌1(𝑚, 𝜔𝑘).
For this homework, we recommend the following 3-dimesional feature vector, but you are encouraged to experiment with other types of space-time features.
<br><br>
As a reference, you will be provided with the source signals (s1.wav, s2.wav, s3.wav), as well as exemplary separated sources (separated_1.wav, separated_2.wav, separated_3.wav). Please note that these files are only for reference purposes and should not be used in implementing the source separation algorithm.<br>
Please provide and be ready to discuss the following plots:<br>
- log-amplitude spectrograms for the mixture signals (subplot). <br>
- log-amplitude spectrograms of the true and estimated source signals (subplots).<br>
- Binary masks, in black and white (subplot).<br>
- Density plot each pair of features, i.e., [𝐴1(𝑚, 𝜔𝑘), 𝐴2(𝑚, 𝜔𝑘)], [𝐴1(𝑚, 𝜔𝑘), 𝑃(𝑚, 𝜔𝑘)], [𝐴2(𝑛, 𝜔𝑘), 𝑃(𝑛, 𝜔𝑘)].<br>
- Accompany the density plots with the histograms of the individual features 𝐴1(𝑚, 𝜔𝑘), 𝐴2(𝑚, 𝜔𝑘), 𝑃(𝑚, 𝜔𝑘). Make sure to label the axes correctly, e.g., by expressing time in seconds and frequency in Hertz.
<br><br>
You are tasked to implement the short-time Fourier transform yourself. Do not use the built-in stft/istft functions from, e.g., MATLAB, scipy, or librosa—just to name a few. On the contrary, feel free to use whichever ready-made implementation of the clustering algorithm is available to you (e.g., scikit-learn). While there is no specific programming language required for this assignment, we strongly recommend using either MATLAB or Python. Lastly, you are expected to explain and justify all your design choices and refer to the course materials if necessary.
