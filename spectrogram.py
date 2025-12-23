import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# funkcja do przygotowania danych
def prepareData(audio, frameSize, hopSize) :
    signal, sr = librosa.load(audio)
    signalSTFT = librosa.stft(signal, n_fft=frameSize, hop_length=hopSize) # stft dla sygnału
    signalPower = np.abs(signalSTFT) ** 2 # gęstość mocy dla sygnału
    signalPowerLog = librosa.power_to_db(signalPower) # gęstość mocy w skali logarytmicznej

    return signalSTFT, signalPowerLog, sr

# funkcja do rysowania spektrogramu
def drawSpectrogram(data, sr, hopSize) :
    plt.figure(figsize = (20, 10))
    librosa.display.specshow(
        data,
        sr=sr,
        hop_length=hopSize,
        x_axis='time',
        y_axis='log'
    )
    plt.colorbar(format="%+2.f")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Spectrogram")
    # plt.savefig("spectrograms/spectrogram.png")
    plt.show()

# analiza widma w danej chwili czasu
def spectrumAtTimeAnalysis(stft, time, frameSize, hopSize, samples) :
    moment = int(time * samples / hopSize)
    moment = min(moment, stft.shape[1] - 1)

    # częstotliwości w danej chwili czasu
    spectrum = np.abs(stft[:, moment])
    frs = librosa.fft_frequencies(sr=samples, n_fft=frameSize)

    # print(frs.size)

    return frs, spectrum

# rysowanie widma w danej chwili czasu
def drawSpectrumAtTime(frs, spectrum) :
    plt.plot(frs, spectrum)
    plt.xlabel("Frequency [Hz]")
    plt.title("Magnitude")
    plt.grid()
    plt.show()

def conductFullAnalysis(audio, frameSize, hopSize, time) :
    stft, stftLog, samples = prepareData(audio, frameSize, hopSize)
    frs, spectrum = spectrumAtTimeAnalysis(stft, time, frameSize, hopSize, samples)
    drawSpectrogram(stftLog, samples, hopSize)
    drawSpectrumAtTime(frs, spectrum)

conductFullAnalysis("signal/dynamic/sunflower_64_linear.wav", 128, 64, 5.0)