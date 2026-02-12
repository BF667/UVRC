import os, torch, yaml
from urllib.parse import quote

class IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, False)


def tuple_constructor(loader, node):
    # Load the sequence of values from the YAML node
    values = loader.construct_sequence(node)
    # Return a tuple constructed from the sequence
    return tuple(values)

# Register the constructor with PyYAML
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple',
tuple_constructor)



def conf_edit(config_path, chunk_size, overlap):
    with open(config_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    # handle cases where 'use_amp' is missing from config:
    if 'use_amp' not in data.keys():
      data['training']['use_amp'] = True

    data['audio']['chunk_size'] = chunk_size
    data['inference']['num_overlap'] = overlap

    if data['inference']['batch_size'] == 1:
      data['inference']['batch_size'] = 2

    print("Using custom overlap and chunk_size values:")
    print(f"overlap = {data['inference']['num_overlap']}")
    print(f"chunk_size = {data['audio']['chunk_size']}")
    print(f"batch_size = {data['inference']['batch_size']}")

    with open(config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, Dumper=IndentDumper, allow_unicode=True)

def download_file(url):
    # Encode the URL to handle spaces and special characters
    encoded_url = quote(url, safe=':/')

    path = 'ckpts'
    os.makedirs(path, exist_ok=True)
    filename = os.path.basename(encoded_url)
    file_path = os.path.join(path, filename)

    if os.path.exists(file_path):
        print(f"File '{filename}' already exists at '{path}'.")
        return

    try:
        response = torch.hub.download_url_to_file(encoded_url, file_path)
        print(f"File '{filename}' downloaded successfully")
    except Exception as e:
        print(f"Error downloading file '{filename}' from '{url}': {e}")





model =  [
  'VOCALS-MelBand-Roformer (by KimberleyJSN)',
  'VOCALS-MelBand-Roformer (by Becruily)',
  'VOCALS-MelBand-Roformer voc_Fv5 (by Gabox)',
  'VOCALS-MelBand-Roformer voc_Fv4 (by Gabox)', 
  'VOCALS-MelBand-Roformer voc_Fv6 experimental (by Gabox)',
  'VOCALS-MelBand-Roformer voc_Fv7 beta 3 (by Gabox)',
  'VOCALS-MelBand-Roformer voc_Fv7 beta 2 (by Gabox)', 
  'VOCALS-MelBand-Roformer voc_Fv7 beta (by Gabox)', 
  'VOCALS-MelBand-Roformer voc_Fv3 (by Gabox)', 
  'VOCALS-Melband-Roformer BigBeta6 (by unwa)',
  'VOCALS-Melband-Roformer BigBeta6X (by unwa)', 
  'VOCALS-MelBand-Roformer voc_gabox2 (by Gabox)', 
  'VOCALS-MelBand-Roformer Kim FT 2 Bleedless (by Unwa)',
  'VOCALS-MelBand-Roformer Kim FT 2 (by Unwa)', 
  'VOCALS-Mel-Roformer FT 3 Preview (by unwa)', 
  'VOCALS-MelBand-Roformer Kim FT (by Unwa)', 
  'VOCALS-Melband-Roformer BigBeta5e (by unwa)', 
  'VOCALS-Mel-Roformer BigBeta4 (by unwa)', 
  'VOCALS-BS-Roformer Resurrection (by unwa)',
  'VOCALS-BS-Roformer Revive 3e (by unwa)',
  'VOCALS-BS-Roformer Revive 2 (by unwa)',
  'VOCALS-BS-Roformer Revive (by unwa)',
  'VOCALS-BS-RoformerLargev1 (by unwa)', 
  'VOCALS-BS-Roformer_1297 (by viperx)', 
  'VOCALS-BS-Roformer_1296 (by viperx)', 
  'VOCALS-InstVocHQ', 
  'INST-Mel-Roformer v1e (by unwa)', 
  'INST-Mel-Roformer v1e+ (by unwa)', 
  'INST-BS-Roformer Resurrection (by unwa)', 
  'INST-Mel-Roformer INSTV8B (by Gabox)', 
  'INST-MelBand-Roformer Inst_Fv8 v2 (by Gabox)',
  'INST-MelBand-Roformer (by Becruily)',
  'INST-Mel-Roformer INSTV7 (by Gabox)', 
  'INST-MelBand-Roformer Inst_Fv4 (by Gabox)',
  'INST-Mel-Roformer Neo_InstVFX (by neoculture)', 
  'INST-Mel-Roformer INSTFVX (by Gabox)', 
  'INST-Mel-Roformer INSTV7N (by Gabox)', 
  'INST-Mel-Roformer INSTFV7Z (by Gabox)',
  'INST-Mel-Roformer INSTV6N (by Gabox)',
  'INST-Mel-Roformer INSTV5 (by Gabox)', 
  'INST-Mel-Roformer INSTV6 (by Gabox)', 
  'INST-Mel-Roformer inst_gabox3 (by Gabox)', 
  'INST-Mel-Roformer Metal Model Preview (by Mesk)', 
  'INST-VOC-Mel-Roformer deux (by Becruily)',
  'INST-VOC-Mel-Roformer a.k.a. duality (by unwa)', 
  'INST-VOC-Mel-Roformer a.k.a. duality v2 (by unwa)', 
  'INST-Mel-Roformer v1 (by unwa)', 
  'INST-Mel-Roformer v1+ (by unwa)',
  'INST-Mel-Roformer v2 (by unwa)',
  'INST-Mel-Roformer Rifforge (for metal; by mesk)',
  'KARAOKE-BS-Roformer (by anvuew)', 
  'KARAOKE-MelBand-Roformer (by becruily)', 
  'KARAOKE-MelBand-Roformer (by aufr33 & viperx)', 
  'OTHER-BS-Roformer_1053 (by viperx)',
  '6STEMS-BS-Roformer-SW', 
  '4STEMS-MelBand-Roformer_Large (by Amane)',
  '4STEMS-SCNet_XL_MUSDB18 (by ZFTurbo)',
  '4STEMS-SCNet_Large (by starrytong)', 
  '4STEMS-BS-Roformer_MUSDB18 (by ZFTurbo)',
  '4STEMS-SCNet_MUSDB18 (by starrytong)',
  'CROWD-REMOVAL-MelBand-Roformer (by aufr33)',
  'VOCALS-VitLarge23 (by ZFTurbo)',
  'CINEMATIC-BandIt_Plus (by kwatcharasupat)', 
  'CINEMATIC-BandIt_v2 multi (by kwatcharasupat)', 
  'DRUMSEP-MDX23C_DrumSep_5stem_new (by jarredou)',
  'DRUMSEP-MDX23C_DrumSep_6stem (by aufr33 & jarredou)',
  'DE-REVERB-MDX23C (by aufr33 & jarredou)', 
  'DE-REVERB-MelBand-Roformer aggr./v2/19.1729 (by anvuew)',
  'DE-REVERB-BS-Roformer dereverb_room mono (by anvuew)',
  'DE-REVERB-MelBand-Roformer mono 20.4029 (by anvuew)', 
  'DE-REVERB-Echo-MelBand-Roformer (by Sucial)', 
  'LEAD-VOCAL-DE-REVERB-MelBand-Roformer (by Gabox)', 
  'DENOISE-MelBand-Roformer-1 (by aufr33)', 
  'DENOISE-MelBand-Roformer-2 (by aufr33)', 
  'DEBLEED-MelBand-Roformer (by unwa/97chris)',
  'DENOISE-DEBLEED-MelBand-Roformer (by Gabox)',
  'VOCALS-Male Female-BS-RoFormer Male Female Beta 7_2889 (by aufr33)',
  'PHANTOM-CENTER-HTDemucs (by wesleyr36)', 
  'GUITAR-MelBand-Roformer (by becruily)'
]


if export_format.startswith('flac'):
    flac_file = True
    pcm_type = export_format.split(' ')[1]
else:
    flac_file = False
    pcm_type = None

if model == 'VOCALS-InstVocHQ':
    model_type = 'mdx23c'
    config_path = 'ckpts/config_vocals_mdx23c.yaml'
    start_check_point = 'ckpts/model_vocals_mdx23c_sdr_10.17.ckpt'
    download_file('https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_vocals_mdx23c.yaml')
    download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt')

elif model == 'VOCALS-MelBand-Roformer (by Becruily)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_vocals_becruily.yaml'
    start_check_point = 'ckpts/mel_band_roformer_vocals_becruily.ckpt'
    download_file('https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/config_vocals_becruily.yaml')
    download_file('https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/mel_band_roformer_vocals_becruily.ckpt')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'INST-MelBand-Roformer (by Becruily)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_instrumental_becruily.yaml'
    start_check_point = 'ckpts/mel_band_roformer_instrumental_becruily.ckpt'
    download_file('https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/config_instrumental_becruily.yaml')
    download_file('https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/mel_band_roformer_instrumental_becruily.ckpt')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-MelBand-Roformer (by KimberleyJSN)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_vocals_mel_band_roformer_kj.yaml'
    start_check_point = 'ckpts/MelBandRoformer.ckpt'
    download_file('https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml')
    download_file('https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-MelBand-Roformer voc_Fv5 (by Gabox)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/voc_gabox.yaml'
    start_check_point = 'ckpts/voc_fv5.ckpt'
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_fv5.ckpt')
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-MelBand-Roformer voc_gabox2 (by Gabox)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/voc_gabox.yaml'
    start_check_point = 'ckpts/voc_gabox2.ckpt'
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox2.ckpt')
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-MelBand-Roformer voc_Fv4 (by Gabox)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/voc_gabox.yaml'
    start_check_point = 'ckpts/voc_fv4.ckpt'
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml')
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_fv4.ckpt')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-MelBand-Roformer voc_Fv6 experimental (by Gabox)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/voc_gabox.yaml'
    start_check_point = 'ckpts/voc_fv6.ckpt'
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml')
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_fv6.ckpt')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-MelBand-Roformer voc_Fv7 beta 3 (by Gabox)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/voc_gabox.yaml'
    start_check_point = 'ckpts/vocfv7beta3.ckpt'
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml')
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/experimental/vocfv7beta3.ckpt')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-MelBand-Roformer voc_Fv7 beta 2 (by Gabox)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/voc_gabox.yaml'
    start_check_point = 'ckpts/vocfv7beta2.ckpt'
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml')
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/experimental/vocfv7beta2.ckpt')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-MelBand-Roformer voc_Fv7 beta (by Gabox)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/voc_gabox.yaml'
    start_check_point = 'ckpts/vocfv7beta1.ckpt'
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml')
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/experimental/vocfv7beta1.ckpt')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-MelBand-Roformer voc_Fv3 (by Gabox)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/voc_gabox.yaml'
    start_check_point = 'ckpts/voc_Fv3.ckpt'
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml')
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_Fv3.ckpt')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-MelBand-Roformer Kim FT (by Unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_kimmel_unwa_ft.yaml'
    start_check_point = 'ckpts/kimmel_unwa_ft.ckpt'
    download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml')
    download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft.ckpt')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-MelBand-Roformer Kim FT 2 (by Unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_kimmel_unwa_ft.yaml'
    start_check_point = 'ckpts/kimmel_unwa_ft.ckpt'
    download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml')
    download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft2.ckpt')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-MelBand-Roformer Kim FT 2 Bleedless (by Unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_kimmel_unwa_ft.yaml'
    start_check_point = 'ckpts/kimmel_unwa_ft2_bleedless.ckpt'
    download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml')
    download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft2_bleedless.ckpt')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-Mel-Roformer FT 3 Preview (by unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_kimmel_unwa_ft.yaml'
    start_check_point = 'ckpts/kimmel_unwa_ft3_prev.ckpt'
    download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft3_prev.ckpt')
    download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-BS-Roformer_1297 (by viperx)':
    model_type = 'bs_roformer'
    config_path = 'ckpts/model_bs_roformer_ep_317_sdr_12.9755.yaml'
    start_check_point = 'ckpts/model_bs_roformer_ep_317_sdr_12.9755.ckpt'
    download_file('https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml')
    download_file('https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-BS-Roformer_1296 (by viperx)':
    model_type = 'bs_roformer'
    config_path = 'ckpts/model_bs_roformer_ep_368_sdr_12.9628.yaml'
    start_check_point = 'ckpts/model_bs_roformer_ep_368_sdr_12.9628.ckpt'
    download_file('https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt')
    download_file('https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_368_sdr_12.9628.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-BS-RoformerLargev1 (by unwa)':
    model_type = 'bs_roformer'
    config_path = 'ckpts/config_bsrofoL.yaml'
    start_check_point = 'ckpts/BS-Roformer_LargeV1.ckpt'
    download_file('https://huggingface.co/jarredou/unwa_bs_roformer/resolve/main/BS-Roformer_LargeV1.ckpt')
    download_file('https://huggingface.co/jarredou/unwa_bs_roformer/raw/main/config_bsrofoL.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-BS-Roformer Resurrection (by unwa)':
    model_type = 'bs_roformer'
    config_path = 'ckpts/BS-Roformer-Resurrection-Config.yaml'
    start_check_point = 'ckpts/BS-Roformer-Resurrection.ckpt'
    download_file('https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection.ckpt')
    download_file('https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection-Config.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-BS-Roformer Revive (by unwa)':
    model_type = 'bs_roformer'
    config_path = 'ckpts/config.yaml'
    start_check_point = 'ckpts/bs_roformer_revive.ckpt'
    download_file('https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/bs_roformer_revive.ckpt')
    download_file('https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/config.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-BS-Roformer Revive 2 (by unwa)':
    model_type = 'bs_roformer'
    config_path = 'ckpts/config.yaml'
    start_check_point = 'ckpts/bs_roformer_revive2.ckpt'
    download_file('https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/bs_roformer_revive2.ckpt')
    download_file('https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/config.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-BS-Roformer Revive 3e (by unwa)':
    model_type = 'bs_roformer'
    config_path = 'ckpts/config.yaml'
    start_check_point = 'ckpts/bs_roformer_revive3e.ckpt'
    download_file('https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/bs_roformer_revive3e.ckpt')
    download_file('https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/config.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-Melband-Roformer BigBeta6X (by unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/big_beta6x.yaml'
    start_check_point = 'ckpts/big_beta6x.ckpt'
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6x.ckpt')
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6x.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-Melband-Roformer BigBeta6 (by unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/big_beta6.yaml'
    start_check_point = 'ckpts/big_beta6.ckpt'
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6.ckpt')
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-Melband-Roformer BigBeta5e (by unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/big_beta5e.yaml'
    start_check_point = 'ckpts/big_beta5e.ckpt'
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.ckpt')
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-Mel-Roformer BigBeta4 (by unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_melbandroformer_big_beta4.yaml'
    start_check_point = 'ckpts/melband_roformer_big_beta4.ckpt'
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/melband_roformer_big_beta4.ckpt')
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/raw/main/config_melbandroformer_big_beta4.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'INST-Mel-Roformer v1 (by unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_melbandroformer_inst.yaml'
    start_check_point = 'ckpts/melband_roformer_inst_v1.ckpt'
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v1.ckpt')
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'INST-Mel-Roformer v2 (by unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_melbandroformer_inst_v2.yaml'
    start_check_point = 'ckpts/melband_roformer_inst_v2.ckpt'
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v2.ckpt')
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst_v2.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'INST-Mel-Roformer v1e (by unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_melbandroformer_inst.yaml'
    start_check_point = 'ckpts/inst_v1e.ckpt'
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e.ckpt')
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'INST-BS-Roformer Resurrection (by unwa)':
    model_type = 'bs_roformer'
    config_path = 'ckpts/BS-Roformer-Resurrection-Inst-Config.yaml'
    start_check_point = 'ckpts/BS-Roformer-Resurrection-Inst.ckpt'
    download_file('https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection-Inst.ckpt')
    download_file('https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection-Inst-Config.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'INST-Mel-Roformer INSTV8B (by Gabox)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/inst_gabox.yaml'
    start_check_point = 'ckpts/Inst_FV8b.ckpt'
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/experimental/Inst_FV8b.ckpt')
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'INST-Mel-Roformer INSTV7 (by Gabox)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/inst_gabox.yaml'
    start_check_point = 'ckpts/Inst_GaboxV7.ckpt'
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxV7.ckpt')
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'INST-Mel-Roformer v1e+ (by unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_melbandroformer_inst.yaml'
    start_check_point = 'ckpts/inst_v1e_plus.ckpt'
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e_plus.ckpt')
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'INST-MelBand-Roformer Inst_Fv8 v2 (by Gabox)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/inst_gabox.yaml'
    start_check_point = 'ckpts/Inst_GaboxFv8.ckpt'
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxFv8.ckpt')
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'INST-MelBand-Roformer Inst_Fv4 (by Gabox)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/inst_gabox.yaml'
    start_check_point = 'ckpts/inst_Fv4.ckpt'
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_Fv4.ckpt')
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'INST-Mel-Roformer INSTFVX (by Gabox)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/inst_gabox.yaml'
    start_check_point = 'ckpts/Inst_GaboxFVX.ckpt'
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxFVX.ckpt')
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'INST-Mel-Roformer Neo_InstVFX (by neoculture)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_neo_inst.yaml'
    start_check_point = 'ckpts/Neo_InstVFX.ckpt'
    download_file('https://huggingface.co/natanworkspace/melband_roformer/resolve/main/Neo_InstVFX.ckpt')
    download_file('https://huggingface.co/natanworkspace/melband_roformer/resolve/main/config_neo_inst.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'INST-Mel-Roformer INSTFV7Z (by Gabox)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/inst_gabox.yaml'
    start_check_point = 'ckpts/Inst_GaboxFv7z.ckpt'
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxFv7z.ckpt')
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'INST-Mel-Roformer INSTV7N (by Gabox)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/inst_gabox.yaml'
    start_check_point = 'ckpts/INSTV7N.ckpt'
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV7N.ckpt')
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'INST-Mel-Roformer INSTV6N (by Gabox)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/inst_gabox.yaml'
    start_check_point = 'ckpts/INSTV6N.ckpt'
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV6N.ckpt')
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'INST-Mel-Roformer v1+ (by unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_melbandroformer_inst.yaml'
    start_check_point = 'ckpts/inst_v1_plus_test.ckpt'
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1_plus_test.ckpt')
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'INST-Mel-Roformer INSTV5 (by Gabox)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/inst_gabox.yaml'
    start_check_point = 'ckpts/INSTV5.ckpt'
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV5.ckpt')
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'INST-Mel-Roformer INSTV6 (by Gabox)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/inst_gabox.yaml'
    start_check_point = 'ckpts/INSTV6.ckpt'
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV6.ckpt')
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'INST-Mel-Roformer inst_gabox3 (by Gabox)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/inst_gabox.yaml'
    start_check_point = 'ckpts/inst_gabox3.ckpt'
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox3.ckpt')
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'INST-Mel-Roformer Metal Model Preview (by Mesk)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_inst_metal_roformer_mesk.yaml'
    start_check_point = 'ckpts/metal_roformer_inst_mesk_preview.ckpt'
    download_file('https://huggingface.co/meskvlla33/metal_roformer_preview/resolve/main/metal_roformer_inst_mesk_preview.ckpt')
    download_file('https://huggingface.co/meskvlla33/metal_roformer_preview/resolve/main/config_inst_metal_roformer_mesk.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'INST-VOC-Mel-Roformer deux (by Becruily)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_deux_becruily.yaml'
    start_check_point = 'ckpts/becruily_deux.ckpt'
    download_file('https://huggingface.co/becruily/mel-band-roformer-deux/resolve/main/becruily_deux.ckpt')
    download_file('https://huggingface.co/becruily/mel-band-roformer-deux/resolve/main/config_deux_becruily.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'INST-VOC-Mel-Roformer a.k.a. duality (by unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_melbandroformer_instvoc_duality.yaml'
    start_check_point = 'ckpts/melband_roformer_instvoc_duality_v1.ckpt'
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvoc_duality_v1.ckpt')
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/raw/main/config_melbandroformer_instvoc_duality.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'INST-VOC-Mel-Roformer a.k.a. duality v2 (by unwa)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_melbandroformer_instvoc_duality.yaml'
    start_check_point = 'ckpts/melband_roformer_instvox_duality_v2.ckpt'
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvox_duality_v2.ckpt')
    download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/raw/main/config_melbandroformer_instvoc_duality.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'INST-Mel-Roformer Rifforge (for metal; by mesk)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_rifforge_full_mesk.yaml'
    start_check_point = 'ckpts/rifforge_full_sdr_14.2436.ckpt'
    download_file('https://huggingface.co/meskvlla33/rifforge/resolve/main/rifforge_full_sdr_14.2436.ckpt')
    download_file('https://huggingface.co/meskvlla33/rifforge/resolve/main/config_rifforge_full_mesk.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'KARAOKE-BS-Roformer (by anvuew)':
    model_type = 'bs_roformer'
    config_path = 'ckpts/karaoke_bs_roformer_anvuew.yaml'
    start_check_point = 'ckpts/karaoke_bs_roformer_anvuew.ckpt'
    download_file('https://huggingface.co/anvuew/karaoke_bs_roformer/resolve/main/karaoke_bs_roformer_anvuew.ckpt')
    download_file('https://huggingface.co/anvuew/karaoke_bs_roformer/resolve/main/karaoke_bs_roformer_anvuew.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'KARAOKE-MelBand-Roformer (by becruily)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_karaoke_becruily.yaml'
    start_check_point = 'ckpts/mel_band_roformer_karaoke_becruily.ckpt'
    download_file('https://huggingface.co/becruily/mel-band-roformer-karaoke/resolve/main/mel_band_roformer_karaoke_becruily.ckpt')
    download_file('https://huggingface.co/becruily/mel-band-roformer-karaoke/resolve/main/config_karaoke_becruily.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'KARAOKE-MelBand-Roformer (by aufr33 & viperx)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_mel_band_roformer_karaoke.yaml'
    start_check_point = 'ckpts/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt'
    download_file('https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt')
    download_file('https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/config_mel_band_roformer_karaoke.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'OTHER-BS-Roformer_1053 (by viperx)':
    model_type = 'bs_roformer'
    config_path = 'ckpts/model_bs_roformer_ep_937_sdr_10.5309.yaml'
    start_check_point = 'ckpts/model_bs_roformer_ep_937_sdr_10.5309.ckpt'
    download_file('https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_937_sdr_10.5309.ckpt')
    download_file('https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_937_sdr_10.5309.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'CROWD-REMOVAL-MelBand-Roformer (by aufr33)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/model_mel_band_roformer_crowd.yaml'
    start_check_point = 'ckpts/mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt'
    download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.4/mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt')
    download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.4/model_mel_band_roformer_crowd.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-VitLarge23 (by ZFTurbo)':
    model_type = 'segm_models'
    config_path = 'ckpts/config_vocals_segm_models.yaml'
    start_check_point = 'ckpts/model_vocals_segm_models_sdr_9.77.ckpt'
    download_file('https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/refs/heads/main/configs/config_vocals_segm_models.yaml')
    download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_segm_models_sdr_9.77.ckpt')

elif model == 'CINEMATIC-BandIt_Plus (by kwatcharasupat)':
    model_type = 'bandit'
    config_path = 'ckpts/config_dnr_bandit_bsrnn_multi_mus64.yaml'
    start_check_point = 'ckpts/model_bandit_plus_dnr_sdr_11.47.chpt'
    download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/config_dnr_bandit_bsrnn_multi_mus64.yaml')
    download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/model_bandit_plus_dnr_sdr_11.47.chpt')

elif model == 'CINEMATIC-BandIt_v2 multi (by kwatcharasupat)':
    model_type = 'bandit_v2'
    config_path = 'ckpts/config_dnr_bandit_v2_mus64.yaml'
    start_check_point = 'ckpts/checkpoint-multi_state_dict.ckpt'
    download_file('https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/refs/heads/main/configs/config_dnr_bandit_v2_mus64.yaml')
    download_file('https://huggingface.co/jarredou/banditv2_state_dicts_only/resolve/main/checkpoint-multi_state_dict.ckpt')

elif model == 'DRUMSEP-MDX23C_DrumSep_5stem_new (by jarredou)':
    model_type = 'mdx23c'
    config_path = 'ckpts/config_mdx23c.yaml'
    start_check_point = 'ckpts/drumsep_5stems_mdx23c_jarredou.ckpt'
    download_file('https://github.com/jarredou/models/releases/download/DrumSep/drumsep_5stems_mdx23c_jarredou.ckpt')
    download_file('https://github.com/jarredou/models/releases/download/DrumSep/config_mdx23c.yaml')

elif model == 'DRUMSEP-MDX23C_DrumSep_6stem (by aufr33 & jarredou)':
    model_type = 'mdx23c'
    config_path = 'ckpts/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml'
    start_check_point = 'ckpts/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt'
    download_file('https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt')
    download_file('https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml')

elif model == '4STEMS-MelBand-Roformer_Large (by Amane)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_large.yaml'
    start_check_point = 'ckpts/mel_band_roformer_4stems_large_ver1.ckpt'
    download_file('https://huggingface.co/Aname-Tommy/melbandroformer4stems/resolve/main/config_large.yaml')
    download_file('https://huggingface.co/Aname-Tommy/melbandroformer4stems/resolve/main/mel_band_roformer_4stems_large_ver1.ckpt')

elif model == '6STEMS-BS-Roformer-SW':
    model_type = 'bs_roformer'
    config_path = 'ckpts/BS-Rofo-SW-Fixed.yaml'
    start_check_point = 'ckpts/BS-Rofo-SW-Fixed.ckpt'
    download_file('https://huggingface.co/jarredou/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.yaml')
    download_file('https://huggingface.co/jarredou/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.ckpt')

elif model == '4STEMS-SCNet_XL_MUSDB18 (by ZFTurbo)':
    model_type = 'scnet'
    config_path = 'ckpts/config_musdb18_scnet_xl.yaml'
    start_check_point = 'ckpts/model_scnet_ep_54_sdr_9.8051.ckpt'
    download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/config_musdb18_scnet_xl.yaml')
    download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/model_scnet_ep_54_sdr_9.8051.ckpt')

elif model == '4STEMS-SCNet_Large (by starrytong)':
    model_type = 'scnet'
    config_path = 'ckpts/config_musdb18_scnet_large_starrytong.yaml'
    start_check_point = 'ckpts/SCNet-large_starrytong_fixed.ckpt'
    download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/config_musdb18_scnet_large_starrytong.yaml')
    download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/SCNet-large_starrytong_fixed.ckpt')

elif model == '4STEMS-SCNet_MUSDB18 (by starrytong)':
    model_type = 'scnet'
    config_path = 'ckpts/config_musdb18_scnet.yaml'
    start_check_point = 'ckpts/scnet_checkpoint_musdb18.ckpt'
    download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/config_musdb18_scnet.yaml')
    download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/scnet_checkpoint_musdb18.ckpt')

elif model == '4STEMS-BS-Roformer_MUSDB18 (by ZFTurbo)':
    model_type = 'bs_roformer'
    config_path = 'ckpts/config_bs_roformer_384_8_2_485100.yaml'
    start_check_point = 'ckpts/model_bs_roformer_ep_17_sdr_9.6568.ckpt'
    download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/config_bs_roformer_384_8_2_485100.yaml')
    download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt')

elif model == 'DE-REVERB-MDX23C (by aufr33 & jarredou)':
    model_type = 'mdx23c'
    config_path = 'ckpts/config_dereverb_mdx23c.yaml'
    start_check_point = 'ckpts/dereverb_mdx23c_sdr_6.9096.ckpt'
    download_file('https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/dereverb_mdx23c_sdr_6.9096.ckpt')
    download_file('https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/config_dereverb_mdx23c.yaml')

elif model == 'DE-REVERB-MelBand-Roformer aggr./v2/19.1729 (by anvuew)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/dereverb_mel_band_roformer_anvuew.yaml'
    start_check_point = 'ckpts/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt'
    download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt')
    download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml')

elif model == 'DE-REVERB-BS-Roformer dereverb_room mono (by anvuew)':
    model_type = 'bs_roformer'
    config_path = 'ckpts/dereverb_room_anvuew.yaml'
    start_check_point = 'ckpts/dereverb_room_anvuew_sdr_13.7432.ckpt'
    download_file('https://huggingface.co/anvuew/dereverb_room/resolve/main/dereverb_room_anvuew_sdr_13.7432.ckpt')
    download_file('https://huggingface.co/anvuew/dereverb_room/resolve/main/dereverb_room_anvuew.yaml')

elif model == 'DE-REVERB-MelBand-Roformer mono 20.4029 (by anvuew)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/dereverb_mel_band_roformer_anvuew.yaml'
    start_check_point = 'ckpts/dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt'
    download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt')
    download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml')

elif model == 'DE-REVERB-Echo-MelBand-Roformer (by Sucial)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_dereverb-echo_mel_band_roformer.yaml'
    start_check_point = 'ckpts/dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt'
    download_file('https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt')
    download_file('https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb-echo_mel_band_roformer.yaml')

elif model == 'LEAD-VOCAL-DE-REVERB-MelBand-Roformer (by Gabox)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/karaokegabox_1750911344.yaml'
    start_check_point = 'ckpts/Lead_VocalDereverb.ckpt'
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/experimental/Lead_VocalDereverb.ckpt')
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/karaoke/karaokegabox_1750911344.yaml')

elif model == 'DENOISE-MelBand-Roformer-1 (by aufr33)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/model_mel_band_roformer_denoise.yaml'
    start_check_point = 'ckpts/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt'
    download_file('https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt')
    download_file('https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/model_mel_band_roformer_denoise.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'DENOISE-MelBand-Roformer-2 (by aufr33)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/model_mel_band_roformer_denoise.yaml'
    start_check_point = 'ckpts/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt'
    download_file('https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt')
    download_file('https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/model_mel_band_roformer_denoise.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'DEBLEED-MelBand-Roformer (by unwa/97chris)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_bleed_suppressor_v1.yaml'
    start_check_point = 'ckpts/bleed_suppressor_v1.ckpt'
    download_file('https://huggingface.co/jarredou/bleed_suppressor_melband_rofo_by_unwa_97chris/resolve/main/bleed_suppressor_v1.ckpt')
    download_file('https://huggingface.co/jarredou/bleed_suppressor_melband_rofo_by_unwa_97chris/resolve/main/config_bleed_suppressor_v1.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'DENOISE-DEBLEED-MelBand-Roformer (by Gabox)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/inst_gabox.yaml'
    start_check_point = 'ckpts/denoisedebleed.ckpt'
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/denoisedebleed.ckpt')
    download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'VOCALS-Male Female-BS-RoFormer Male Female Beta 7_2889 (by aufr33)':
    model_type = 'bs_roformer'
    config_path = 'ckpts/config_chorus_male_female_bs_roformer.yaml'
    start_check_point = 'ckpts/bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt'
    download_file('https://huggingface.co/RareSirMix/AIModelRehosting/resolve/main/bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt')
    download_file('https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/config_chorus_male_female_bs_roformer.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'PHANTOM-CENTER-HTDemucs (by wesleyr36)':
    model_type = 'htdemucs'
    config_path = 'ckpts/config_htdemucs_similarity.yaml'
    start_check_point = 'ckpts/model_htdemucs_ep_21_sdr_13.6970.ckpt'
    download_file('https://huggingface.co/jarredou/HTDemucs_Similarity_Extractor_by_wesleyr36/resolve/main/model_htdemucs_ep_21_sdr_13.6970.ckpt')
    download_file('https://huggingface.co/jarredou/HTDemucs_Similarity_Extractor_by_wesleyr36/resolve/main/config_htdemucs_similarity.yaml')
    conf_edit(config_path, chunk_size, overlap)

elif model == 'GUITAR-MelBand-Roformer (by becruily)':
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config_guitar_becruily.yaml'
    start_check_point = 'ckpts/becruily_guitar.ckpt'
    download_file('https://huggingface.co/becruily/mel-band-roformer-guitar/resolve/main/becruily_guitar.ckpt')
    download_file('https://huggingface.co/becruily/mel-band-roformer-guitar/resolve/main/config_guitar_becruily.yaml')
    conf_edit(config_path, chunk_size, overlap)


!python inference.py \
    --model_type {model_type} \
    --config_path '{config_path}' \
    --start_check_point '{start_check_point}' \
    --input_folder '{input_folder}' \
    --store_dir '{output_folder}' \
    {('--extract_instrumental' if extract_instrumental else '')} \
    {('--flac_file' if flac_file else '')} \
    {('--use_tta' if use_tta else '')} \
    {('--pcm_type ' + pcm_type if pcm_type else '')}
