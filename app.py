from flask import Flask, render_template, request, send_from_directory
from moviepy.editor import VideoFileClip
import os
import cv2
import torch
import torchaudio
from torchvision.io import read_video
from torch.utils.data import Dataset
from transformers import ViTFeatureExtractor, ViTModel
from transformers import Wav2Vec2Model
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image
import torch.nn.functional as F
from torch import nn
import torch.optim as optim


class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.audio_net = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.classifier = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size + self.audio_net.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes))


    def forward(self, images, audio):


        # Assuming images is of shape (batch_size, num_frames, channels, height, width)
        batch_size, num_frames, channels, height, width = images.shape
        images = images.view(batch_size * num_frames, channels, height, width)

        # Reshape audio for Wav2Vec2
        batch_size, *_ = audio.shape
        audio = audio.view(batch_size, -1)

        # Process images and audio
        x1 = self.vit(images).pooler_output
        x2 = self.audio_net(audio).last_hidden_state.mean(dim=1)

        # Make the tensors contiguous and flatten them
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        # Average the features across the frames for the video
        x1 = x1.view(batch_size, num_frames, -1).mean(dim=1)

        # Concatenate and classify
        x = torch.cat((x1, x2), dim=1)
        return self.classifier(x)

# Load your trained model
model = FusionModel(num_classes=2)
model.load_state_dict(torch.load(r"C:\Users\cyril\PycharmProjects\flaskProject\late_fusion_transformer_model1.pth"))
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(device)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def process_input():

    if request.method == 'POST':
        file = request.files.get('videoUpload')
        if file:
            file.save("upload_video/video.mp4")
            video = VideoFileClip("upload_video/video.mp4")
            audio = video.audio
            audio.write_audiofile("upload_video/video.wav")

            # Load the video and audio
            video, audio, _ = read_video(r"C:\Users\cyril\PycharmProjects\flaskProject\upload_video\video.mp4", pts_unit='sec', output_format='TCHW')
            waveform, sample_rate = torchaudio.load(r"C:\Users\cyril\PycharmProjects\flaskProject\upload_video\video.wav")

            # Preprocess the video
            # Normalize the video frames
            video = video / 224.0

            # Preprocess the audio
            # If your audio data is stereo, you might want to convert it to mono
            if audio.shape[0] == 2:
                audio = torch.mean(audio, dim=0, keepdim=True)
            # Normalize the audio waveform
            audio = torchaudio.transforms.Vad(sample_rate)(waveform)
            # If your model expects a specific length, you might need to pad or truncate the waveform
            if audio.shape[1] < sample_rate:
                audio = torch.nn.functional.pad(audio, (0, sample_rate - audio.shape[1]))
            elif audio.shape[1] > sample_rate:
                audio = audio[:, :sample_rate]
            audio = audio.reshape(1, -1)

            # Reshape the tensors for model input
            video = torch.nn.functional.interpolate(video.squeeze(0), size=(224, 224), mode='bilinear')
            video = video.unsqueeze(0)  # Add batch dimension
            audio = audio.unsqueeze(0)  # Add batch dimension

            # Move tensors to the correct device
            print(device)
            video = video.to(device)
            audio = audio.to(device)


            # Make predictions
            with torch.no_grad():  # Disable gradient calculation for prediction
                outputs = model(video, audio)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probs, dim=1).item()
                classes = ['real','fake']
                predicted_class_name = classes[predicted_class]
                print("probs", probs)

            print(f"Predicted class: {predicted_class_name}")
            result = "The Video is "+predicted_class_name
            torch.cuda.empty_cache()

            return render_template('index.html', result=result)

        else:
            return "No file was submitted with the form."

    else:
        # Print an error message
        return "Error uploading file."



if __name__ == '__main__':
    app.run(debug=True)
