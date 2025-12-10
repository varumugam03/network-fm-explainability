import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from .mlp import MLP

class NetworkExplainer(nn.Module):
    """
    Multimodal Network Explainer.
    Combines a frozen MLP encoder with a LoRA-finetuned LLM.
    """
    def __init__(
        self, 
        mlp_checkpoint: str, 
        llm_model_id: str, 
        mlp_config: dict,
        device: str = "cuda"
    ):
        super().__init__()
        self.device = device
        
        # 1. Load and Freeze MLP
        print("Loading MLP...")
        self.mlp = MLP(**mlp_config)
        self.mlp.load_state_dict(torch.load(mlp_checkpoint, map_location=device))
        self.mlp.to(device)
        self.mlp.eval()
        for param in self.mlp.parameters():
            param.requires_grad = False
            
        # 2. Load LLM
        print(f"Loading LLM: {llm_model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_id, padding_side="left")
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load base model in bfloat16 for efficiency
        base_llm = AutoModelForCausalLM.from_pretrained(
            llm_model_id, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        
        # 3. Apply LoRA
        print("Applying LoRA...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"] # Common targets for attention
        )
        self.llm = get_peft_model(base_llm, peft_config)
        self.llm.print_trainable_parameters()
        self.llm.to(device)

    def forward(self, mlp_features, text_prompts, labels=None):
        """
        Forward pass for training.
        
        :param mlp_features: Tensor (B, Input_Dim)
        :param text_prompts: List[str] of plain english features
        :param labels: List[str] of ground truth explanations
        """
        batch_size = len(text_prompts)
        
        # 1. Get MLP Embedding
        # We only want the embedding, not the logits. 
        # The MLP refactor returns (logits, embedding) if requested.
        _, flow_embedding = self.mlp(mlp_features, return_embedding=True)
        
        # Reshape to (Batch, 1, Hidden) to act as a single token
        flow_embedding = flow_embedding.unsqueeze(1).to(self.llm.dtype) # Cast to bf16
        
        # 2. Tokenize Text Inputs
        # Structure: [User Prompt] -> [Response]
        # We need to manually construct embeddings to insert the flow token.
        
        inputs_embeds = []
        attention_masks = []
        target_ids = []

        # We construct the sequence for each item in batch
        for i in range(batch_size):
            # Format: "Flow info: [TEXT] \n Explanation:"
            # Note: You might want to wrap this in a chat template if using Instruct models
            user_text = f"Analyze this network flow:\n{text_prompts[i]}\n\nExplanation:"
            target_text = labels[i] + self.tokenizer.eos_token
            
            # Tokenize parts
            user_tokens = self.tokenizer(user_text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            target_tokens = self.tokenizer(target_text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            
            # Get Embeddings for text
            user_embeds = self.llm.get_input_embeddings()(user_tokens)
            target_embeds = self.llm.get_input_embeddings()(target_tokens)
            
            # Concatenate: [MLP_TOKEN] + [USER_TEXT] + [TARGET]
            # Current flow_embedding[i] is (1, H)
            combined_embeds = torch.cat([flow_embedding[i].unsqueeze(0), user_embeds, target_embeds], dim=1)
            
            inputs_embeds.append(combined_embeds.squeeze(0))
            
            # Create Attention Mask (1 for all)
            attention_masks.append(torch.ones(combined_embeds.shape[1], device=self.device))
            
            # Create Labels (Target IDs)
            # -100 for MLP token and User Text (don't train on prompts), Real IDs for Target
            ignore_len = 1 + user_tokens.shape[1] # 1 for MLP token
            label_ids = torch.cat([
                torch.full((ignore_len,), -100, device=self.device, dtype=torch.long),
                target_tokens.squeeze(0)
            ])
            target_ids.append(label_ids)

        # 3. Pad and Stack
        # Since lengths vary, we need to pad inputs_embeds, masks, and labels
        # Using a simple collation logic here
        max_len = max([x.size(0) for x in inputs_embeds])
        
        padded_embeds = torch.zeros(batch_size, max_len, self.llm.config.hidden_size, device=self.device, dtype=self.llm.dtype)
        padded_mask = torch.zeros(batch_size, max_len, device=self.device, dtype=torch.long)
        padded_labels = torch.full((batch_size, max_len), -100, device=self.device, dtype=torch.long)
        
        for i in range(batch_size):
            length = inputs_embeds[i].size(0)
            # Left padding is standard for generation, Right for training. 
            # Transformers handles right padding for training usually.
            padded_embeds[i, :length, :] = inputs_embeds[i]
            padded_mask[i, :length] = attention_masks[i]
            padded_labels[i, :length] = target_ids[i]

        # 4. LLM Forward
        outputs = self.llm(
            inputs_embeds=padded_embeds,
            attention_mask=padded_mask,
            labels=padded_labels
        )
        
        return outputs.loss