from transformers import AutoModelForCausalLM, AutoTokenizer,AutoProcessor,Qwen2VLForConditionalGeneration
import torch
import json
import os
from qwen_vl_utils import process_vision_info
import numpy as np
import argparse
from transformers.models.qwen2_vl.modeling_qwen2_vl import (Qwen2VLAttention, _flash_attention_forward,
 is_flash_attn_greater_or_equal_2_10, QWEN2_VL_ATTENTION_CLASSES,apply_multimodal_rotary_pos_emb,repeat_kv)
from typing import Optional,Tuple
import time
from datasets import Dataset
from tqdm import tqdm
import yaml
import cv2
import re
class Qwen2VLFlashAttention2(Qwen2VLAttention):
    """
    Qwen2VL flash attention module, following Qwen2VL attention module. This module inherits from `Qwen2VLAttention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        self.q = None
        self.k = None
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2) #b,28,len,d
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)#b,4,len,d
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # print('---------刚进来----------')
        # print(query_states.shape,key_states.shape)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:  #kv cache
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            # print(f'kv_seq_len:{kv_seq_len}')
        # Because the input can be padded, the absolute sequence length depends on the max position id.
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # print('------repeat kv---------')
        # print(key_states.shape)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        # print('------before input---------')
        # print(query_states.shape,key_states.shape,value_states.shape)
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None
        self.q = query_states.detach().cpu().float().numpy()
        self.k = key_states.detach().cpu().float().numpy()
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        
        # present_qk = None
        # if self.att:
        #     present_qk = (query_states,key_states)
        
        return attn_output, attn_weights, past_key_value, #present_qk

# QWEN2_VL_ATTENTION_CLASSES["flash_attention_2"] = Qwen2VLFlashAttention2
    

def get_args():
    parser = argparse.ArgumentParser('qwen2-vl inference', add_help=False)
    parser.add_argument('--num_frames', default=64, type=int)
    parser.add_argument('--log', default=None, type=str)
    parser.add_argument('--sample_id', default=0, type=int)
    parser.add_argument('--json_path', default='/irip/houpuyue_2020/.cache/huggingface/datasets/lmms-lab___video-mme/videomme/0.0.0/ead1408f75b618502df9a1d8e0950166bf0a2a0b/video-mme-test.arrow', type=str)
    parser.add_argument('--base_cache_dir', default='/irip/houpuyue_2020/.cache/huggingface/videomme/', type=str)
    parser.add_argument('--eval_yaml', default='./videomme_w_subtitle_hpy.yaml', type=str)
    parser.add_argument('--prompt_path', default='./videomme_promt.json', type=str)
    return parser.parse_args()

def load_dataset_dict(json_dir,):
    dataset = Dataset.from_file(json_dir)
    df = dataset.to_pandas()
    data_dict = df.to_dict(orient='index')
    return data_dict


def videomme_doc_to_text_subtitle(doc, args, lmms_eval_specific_kwargs=None):
    
    cache_dir = args.base_cache_dir
    video_path = doc["videoID"] + ".mp4"
    video_path = os.path.join(cache_dir, "data", video_path)
    subtitle_path = os.path.join(cache_dir, "subtitle", doc["videoID"] + ".srt")
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(subtitle_path):  # Denote have subtitle
        subtitle = open(subtitle_path).readlines()
    else:
        subtitle = ""
    subtitles_prompt = "This video's subtitles are listed below: \n"
    if subtitle == "":
        subtitle = "No subtitles available"
    else:
        if "gemini_api_flag" in lmms_eval_specific_kwargs:  # specific for gemini_api
            if lmms_eval_specific_kwargs["gemini_api_flag"] == "full subtitle":
                textlist = []
                for ele in subtitle:
                    pattern = r'<font color="white" size=".72c">(.*?)</font>'
                    matches = re.findall(pattern, ele)
                    if matches:
                        textlist.append(matches[0])
                subtitle_text = "\n".join(textlist)
        else:
            if "frame_num" in lmms_eval_specific_kwargs:
                frame_num = lmms_eval_specific_kwargs["frame_num"]
                subtitle_by_frame, total_frame = extract_subtitles(video_path, subtitle_path)
                if frame_num == -1:
                    frame_num = total_frame
                uniform_sampled_frames = np.linspace(0, total_frame - 1, frame_num, dtype=int).tolist()

                subtitle_by_frame_idx = []
                for frame_idx in uniform_sampled_frames:
                    for idx, title in enumerate(subtitle_by_frame):
                        if frame_idx < title[1] and frame_idx >= title[0]:
                            subtitle_by_frame_idx.append(idx)
                subtitle_by_frame_idx = list(set(subtitle_by_frame_idx))

                textlist = []
                for idx in subtitle_by_frame_idx:
                    pattern = r'<font color="white" size=".72c">(.*?)</font>'
                    raw_text = re.findall(pattern, subtitle_by_frame[idx][2])
                    try:
                        textlist.append(raw_text[0])
                    except:
                        continue
                subtitle_text = "\n".join(textlist)
        subtitle = subtitle_text

    option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc["question"]
    option = "\n".join([f"{opt}" for i, opt in enumerate(doc["options"])])
    question = question + "\n" + option
    full_prompt = subtitles_prompt + subtitle + "\n" + option_prompt + "\n" + question + "\n" + "The best answer is:"
    return full_prompt

def extract_subtitles(video_path, subtitle_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    subtitles = load_subtitles(subtitle_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        start_frame = convert_time_to_frame(start_time, fps)
        end_frame = convert_time_to_frame(end_time, fps)
        subtitle_frames.append((start_frame, end_frame, text))

    return subtitle_frames, total_frame

def convert_time_to_frame(time_in_seconds, fps):
    return int(time_in_seconds * fps)

def load_subtitles(subtitle_path):
    subtitles = {}
    with open(subtitle_path, "r", encoding="utf-8") as file:
        content = file.read().split("\n\n")
        for section in content:
            if section.strip():
                lines = section.split("\n")
                if len(lines) >= 3:
                    time_range = lines[1].split(" --> ")
                    start_time = parse_subtitle_time(time_range[0])
                    end_time = parse_subtitle_time(time_range[1])
                    text = " ".join(line for line in lines[2:])
                    subtitles[(start_time, end_time)] = text
    return subtitles

def parse_subtitle_time(time_str):
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

def main(args):
    infer_data = load_dataset_dict(args.json_path)
    print(infer_data[0])
   
    # with open(args.eval_yaml, "r", encoding="utf-8") as file:
    #     eval_specific_kwargs = yaml.safe_load(file)
    # p=dict()
    # for i in range(len(infer_data)):
    #     p[i] = videomme_doc_to_text_subtitle(infer_data[i], args, eval_specific_kwargs)
    #     print(i)
    # with open('./videomme_promt.json','w', encoding="utf-8") as f:
    #     json.dump(p, f, ensure_ascii=False) 
    with open(args.prompt_path,'r') as f:
        prompt = json.load(f)
   
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto",max_memory={0:'5GiB'},attn_implementation="flash_attention_2"
    )
   
    # for m in model.model.layers:
    #     origin = m.self_attn
        
    #     m.self_attn = Qwen2VLFlashAttention2(config=origin.config , layer_idx= origin.layer_idx)
        # m.self_attn.load_state_dict(origin.state_dict())


    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct") #Qwen/Qwen2-VL-7B-Instruct
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    gen_kwargs = dict()
  
    gen_kwargs["max_new_tokens"] = 1

    gen_kwargs["temperature"] = 0

    gen_kwargs["top_p"] = None

    gen_kwargs["num_beams"] = 1

    correct = 0
    for i in tqdm(range(len(infer_data))):
        
        video_path = os.path.join(args.base_cache_dir, "data", infer_data[i]['videoID']+'.mp4')
        question = prompt[f'{i}']
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 1003520
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    

        image_inputs, video_inputs = process_vision_info(messages)
        total_frames = len(video_inputs[0])

        indices = np.linspace(0, total_frames - 1, args.num_frames, dtype=int)
        if total_frames - 1 not in indices:
            indices = np.append(indices, total_frames - 1)
        video_inputs[0] = video_inputs[0][indices] #(t,c,h,w)
    
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
    
        # print(inputs['input_ids'].shape)
        
        pad_token_id =tokenizer.pad_token_id
                
        cont = model.generate(
            **inputs,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=False,
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            temperature=gen_kwargs["temperature"],
            max_new_tokens=1,
            use_cache=True,
            
        )
    
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
    
        answers = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(answers)
        if answers == infer_data[i]['answer']:
            correct += 1
        print(correct)
    prin(float(correct/len(infer_data)))
if __name__ =="__main__":
    args = get_args()
   
    s = time.time()
    main(args)
    e = time.time()
   