import torch
import argparse
from transformers import PreTrainedTokenizerFast
from kobart import get_kobart_tokenizer
from transformers import BartForConditionalGeneration

class KoBARTSummaryGenerator(Base):
    def __init__(self, hparams, **kwargs) -> None:
        super(KoBARTSummaryGenerator, self).__init__(hparams, **kwargs)
        # self.tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
        # self.model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
        self.tokenizer = get_kobart_tokenizer()
        ckpt = torch.load(self.hparams.model_path)    
        kobart_model= BartForConditionalGeneration(self.hparams)
        kobart_model.load_state_dict(ckpt['model_state_dict'])
        # kobart_model = BartForConditionalGeneration.from_pretrained('./kobart_summary')
        kobart_model.eval()
        self.model = kobart_model

    def print_summary(self, text):
        text = text.replace('\n', ' ')
        input_ids = self.tokenizer.encode(text)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.unsqueeze(0)
        output = self.model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return output