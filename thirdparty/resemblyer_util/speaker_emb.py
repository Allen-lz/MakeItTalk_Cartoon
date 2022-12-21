from resemblyzer import preprocess_wav, VoiceEncoder
import numpy as np
import torch


def get_spk_emb(audio_file_dir, segment_len=960000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 声音编码
    resemblyzer_encoder = VoiceEncoder(device=device)

    wav = preprocess_wav(audio_file_dir)
    l = len(wav) // segment_len  # segment_len = 16000 * 60
    l = np.max([1, l])
    all_embeds = []
    for i in range(l):
        # 一段一段的推理, 第一段要是不满一段就将其全部塞进去, 强行作为一段, 后期不满一段的就直接舍弃
        mean_embeds, cont_embeds, wav_splits = resemblyzer_encoder.embed_utterance(
            wav[segment_len * i:segment_len * (i + 1)], return_partials=True, rate=2)
        all_embeds.append(mean_embeds)
    all_embeds = np.array(all_embeds)
    mean_embed = np.mean(all_embeds, axis=0)

    return mean_embed, all_embeds



if __name__ == '__main__':
    m, a = get_spk_emb(r'E:\audio2face\MakeItTalk\examples\M6_04_16k.wav')
    print('Mean Speaker embedding:', m.shape)  # (256,)
    print('All Speaker embedding:', a.shape)   # (1, 256)