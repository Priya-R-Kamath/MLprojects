import speechbrain as sb
import torchaudio
from IPython.display import Audio
from speechbrain.dataio.dataio import read_audio
from speechbrain.pretrained import SpeakerRecognition
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.pretrained import SepformerSeparation as separator
from scipy.io import wavfile

model = separator.from_hparams(
    source="speechbrain/sepformer-wsj02mix", savedir='pretrained_models/sepformer-wsj02mix')

model_path = input("Path to audio file")
est_sources = model.separate_file(path=model_path)

wavfile.write('output1.wav', rate=8000,
              data=est_sources[:, :, 0].detach().cpu().squeeze().numpy())
Audio(est_sources[:, :, 0].detach().cpu().squeeze(), rate=8000)

wavfile.write('output2.wav', rate=8000,
              data=est_sources[:, :, 1].detach().cpu().squeeze().numpy())
Audio(est_sources[:, :, 1].detach().cpu().squeeze(), rate=8000)
