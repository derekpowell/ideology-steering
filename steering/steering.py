import transformers
import torch
import torch.nn.functional as F
from baukit import Trace

class SteeringModel:
    def __init__(self, model, tokenizer):
        
        self.model = model
        self.tok = tokenizer

        self.tok.pad_token_id = tokenizer.eos_token_id
        self.model.generation_config.pad_token_id = self.tok.pad_token_id
        

    def generate(self, texts, **kwargs):
        
        if type(texts) != list:
            texts = [texts]
    
        model = self.model
        tokenizer = self.tok
        encoding = tokenizer(texts, padding=True, return_tensors='pt').to(self.model.device)

        with torch.no_grad():
            generated_ids = model.generate(**encoding, **kwargs) # 

            generated_texts = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            
        return(generated_texts)
    

    def logits(self, texts):
        # texts can be text or list/tensor of input_ids

        if type(texts)!=str and type(texts)!=list:
            encoding = texts.to(self.model.device)
        else:
            encoding = self.tok(texts, return_tensors='pt').to(self.model.device)

        # texts = self.preprompt + texts if type(texts)==str else [self.preprompt + t for t in texts]
    
        
        # encoding = self.tok(texts, padding=True, return_tensors='pt').to(self.model.device)

        with torch.no_grad():
            model_out = self.model(encoding["input_ids"])
            logits = model_out.logits
        
        return {"tokens": encoding, "logits": logits}
    
    
    def logprobs(self, texts):
        
        logits = self.logits(texts)
        
        return {"tokens": logits['tokens'], "logprobs": F.log_softmax(logits['logits'], -1)}
    
    
    def obs_logits(self, text):
    
        x = self.logits(text)
        logits = x['logits']
        
        obslogits = []

        if type(text) is list:
            for i in range(len(text)):
                tok_idx = x['tokens']['input_ids'][i].squeeze()
                mask = x['tokens']['attention_mask'][i] > 0
                
                obslogits.append(logits[0, :, tok_idx[1:]].squeeze().diag()[mask[1:]])

        else:
            tok_idx = x['tokens']['input_ids'].squeeze()
            logits = x['logits']
            obslogits = logits[0, :, tok_idx[1:]].squeeze().diag()

        return obslogits


    def obs_logprobs(self, text):
        logits = self.obs_logits(text)

        return [F.log_softmax(l, -1) for l in logits] if type(logits)==list else F.log_softmax(logits, -1)
    

    def completion_logprob(self, prefix, suffix, summed = True):

        full_text = prefix + suffix
        full = self.obs_logprobs(full_text)
        pre_len = len(self.tok(prefix)['input_ids'])
        res = full[(pre_len-1):] if not summed else full[(pre_len-1):].sum()

        return res

    
    def get_module(self, layer_id = 15):
        return self.model.model.layers[layer_id]
    
    
    def get_steering_vector(self, s1, s2, layer_id = 15):

        module = self.get_module(layer_id)

        with Trace(module) as ret:
            _ = self.logits(s1)
            act1 = ret.output[0]
            _ = self.logits(s2)
            act2 = ret.output[0]
            steering_vec = act1[:, -1, :] - act2[:, -1, :]

        return(steering_vec)