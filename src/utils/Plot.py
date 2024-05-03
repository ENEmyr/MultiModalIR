import io
import cv2
import matplotlib.pyplot as plt
import numpy as np
import wave
import seaborn as sns


def buf_figure_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    figure.savefig(buf, format="png", dpi=180)
    plt.close(figure)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)

    return img


def spectrogram_grid(titles: list, audios: list):
    """Return a 5x5 grid of images as a matplotlib figure."""
    figure = plt.figure(figsize=(10, 10))
    for i in range(25):
        # Start next subplot.
        with wave.open(audios[i], "r") as wav_file:
            metadata = wav_file.getparams()
            sr = metadata.nframes
            frames = wav_file.readframes(sr)
        signal = np.frombuffer(frames, dtype=np.int16)
        plot = plt.subplot(5, 5, i + 1, title=titles[i])
        plt.subplots_adjust(wspace=0.2, hspace=0.5)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plot.set_xlabel("Time")
        plot.set_ylabel("Frequency")
        plot.specgram(signal, NFFT=1024, Fs=sr, noverlap=900)
    return figure


def image_grid(titles: list, images: list):
    """Return a 5x5 grid of images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(10, 10))
    for i in range(25):
        # Start next subplot.
        plt.subplot(5, 5, i + 1, title=titles[i])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])

    return figure


def plot_confusion_matrix(cm, labels):
    figure = plt.figure(figsize=(len(labels), len(labels)))
    ax = plt.subplot()
    sns.heatmap(
        cm, annot=True, fmt="g", ax=ax
    )  # annot=True to annotate cells, ftm='g' to disable scientific notation
    # labels, title and ticks
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix")
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    return figure
