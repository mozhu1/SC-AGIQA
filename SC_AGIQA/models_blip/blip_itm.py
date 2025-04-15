#from models_blip.med import BertConfig, BertModel
from med import BertConfig, BertModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F
try:
    from models_blip.blip import create_vit, init_tokenizer, load_checkpoint
except:
    from blip import create_vit, init_tokenizer, load_checkpoint
class BLIP_ITM(nn.Module):
    def __init__(self,                 
                 med_config = '/home/Newdisk/lq/BLIP-main/configs/med_config.json',  
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                      
                 embed_dim = 256,     
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)          

        text_width = self.text_encoder.config.hidden_size
        
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2) 
        
        
    def forward(self, image, caption, match_head='itm'):

        image_embeds = self.visual_encoder(image) # [B,577,768]
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
      
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to(image.device)   #text['input_ids'].shape torch.Size([1, 35])


                 
        if match_head=='itm':
            output = self.text_encoder(text.input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )
            itm_output = self.itm_head(output.last_hidden_state[:,0,:])      #last_hidden_state  torch.Size([1, 35, 768])
            return  F.softmax(itm_output, dim=-1)[:, 1:2]  #[B,2]匹配和不匹配 itm_output
            
        elif match_head=='itc':
            text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')                     
            image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)   
            text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)    
            
            sim = image_feat @ text_feat.t()
            return sim
        
#/home/Newdisk/lq/My-IQA/models_blip/model_base.pth         /home/Newdisk/lq/My-IQA/models_blip/model_base_retrieval_coco.pth
def blip_itm(pretrained='/home/Newdisk/lq/My-IQA/models_blip/model_base_retrieval_coco.pth',**kwargs):
    model = BLIP_ITM(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model         
            


            
import torch
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

# 假设 BLIP_ITM 类已经定义好了
def debug_blip_itm_with_real_image(model, image_path, caption_text, image_size=224):
    """
    使用真实图片调试 BLIP_ITM 模型。
    
    参数:
        model: 已经初始化好的 BLIP_ITM 模型实例。
        image_path: 图片路径。
        caption_text: 输入的文本描述。
        image_size: 图像调整后的尺寸，默认为 384。
    
    返回:
        输出模型的结果。
    """
    # 1. 加载并预处理图片
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    raw_image = Image.open(image_path).convert('RGB')
    image = image_transform(raw_image).unsqueeze(0)  # 添加 batch 维度

    # 2. 调用模型前向传播（直接传递原始文本）
    output = model(image.to(model.parameters().__next__().device),  # 确保设备一致
                   caption_text, 
                   match_head='itm')  # 使用 ITC 分支
    
    print("\nITC 输出 (相似度矩阵):")
    print(output)
    return output

# 示例：创建模型并调用调试函数
if __name__ == "__main__":
    # 初始化模型
    #/home/Newdisk/lq/BLIP-main/configs/pretrain.yaml  /home/Newdisk/lq/BLIP-main/configs/med_config.json
    med_config = '/home/Newdisk/lq/BLIP-main/configs/med_config.json'  # 确保路径正确
    model = blip_itm(med_config=med_config, image_size=224, vit='base')
    model = model.to('cuda:3' if torch.cuda.is_available() else 'cpu')  # 添加设备处理

    # 定义图片路径和文本
    image_path = '/home/Newdisk/lq/My-IQA/models_blip/cat.jpg'
    caption_text = "A ragdoll cat with long, fluffy white fur sits on a wooden - tiled floor. Its face features distinctive dark - colored patches around the eyes, ears, and nose, creating a striking contrast with the light fur. The cat has large, bright blue eyes that give it an attentive and curious expression. In the background, part of a white chair leg is visible, along with a twisted rope hanging nearby. The sunlight casts a shadow on the floor, adding warmth to the scene. "

    # 调试模型
    debug_blip_itm_with_real_image(model, image_path, caption_text)