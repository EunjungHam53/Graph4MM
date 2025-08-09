
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    CLIPVisionModel,
    CLIPTextModel
)

from peft import (
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningInit,
    PromptTuningConfig,
    TaskType,
    get_peft_model,
)

from .graph import GCN


class TextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class AttentionDiffusion(nn.Module):
    def __init__(self, hidden_dim, num_heads, alpha):
        super(AttentionDiffusion, self).__init__()
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.alpha = alpha  

    def forward(self, node_features, graph, steps=3):
        B, N, D = node_features.shape
        device = node_features.device
        attention_output, attention_scores = self.self_attention(node_features, node_features, node_features)
        masked_attention_scores = attention_scores * graph.unsqueeze(1)  # (B, num_heads, N, N)
        row_sum = masked_attention_scores.sum(dim=-1, keepdim=True) + 1e-9  
        normalized_attention = masked_attention_scores / row_sum
        theta = [self.alpha * (1 - self.alpha) ** i for i in range(steps)]  # θ_i
        theta = torch.tensor(theta, device=device).view(steps, 1, 1)  # (steps, 1, 1)
        A = torch.zeros((B, N, N), device=device)  
        for i in range(steps):
            A += theta[i] * torch.matrix_power(normalized_attention.mean(dim=1), i)  
        updated_features = torch.bmm(A, node_features)  # (B, N, D)
        return updated_features

class TranslatorLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(TranslatorLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        # LayerNorm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        

    def forward(self, Q, z_v, t_v):
        Q_self = self.self_attention(Q, Q, Q)[0]  # (B, M, hidden_dim)
        Q = self.norm1(Q + Q_self)  
        
        Q_cross = self.cross_attention(Q, z_v, z_v)[0]  # (B, M, hidden_dim)
        Q = self.norm2(Q + Q_cross)  

        Q_t_concat = torch.cat([Q, t_v], dim=1)  # (B, M + L_t, hidden_dim)
        Q_t_updated = self.self_attention(Q_t_concat, Q_t_concat, Q_t_concat)[0]  

        Q_updated = Q_t_updated[:, :Q.size(1), :]  # (B, M, hidden_dim)
        t_v_updated = Q_t_updated[:, Q.size(1):, :]  # (B, L_t, hidden_dim)

        Q_ffn = self.ffn(Q_updated)
        Q_final = self.norm3(Q_updated + Q_ffn)  

        return Q_final, t_v_updated


class MultiLayerTranslator(nn.Module):

    def __init__(self, node_dim, text_dim, hidden_dim, num_query_tokens, num_heads, num_layers, alpha=0.1, diffusion_steps=4):
        """
        :param node_dim: node input dim
        :param text_dim: text input dim
        :param hidden_dim: hidden layer dim
        :param num_query_tokens: number of Query Tokens 
        :param num_heads: number of attention heads
        :param num_layers: Translator layers
        """
        super(MultiLayerTranslator, self).__init__()
        
        self.diffusion = AttentionDiffusion(hidden_dim, num_heads, alpha)
        self.diffusion_steps = diffusion_steps
        
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, hidden_dim))
        
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        self.layers = nn.ModuleList([
            TranslatorLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
    
    def forward(self, z_v, t_v, img_grpah, text_graph):
        batch_size = z_v.size(0)
        Q = self.query_tokens.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, M, hidden_dim)        
        z_v_proj = self.node_proj(z_v)  # (B, N, hidden_dim)
        t_v_proj = self.text_proj(t_v)  # (B, L_t, hidden_dim)
        z_v_diffused = self.diffusion(z_v_proj, img_grpah, steps=self.diffusion_steps)  # (B, N, hidden_dim)
        t_v_diffused = self.diffusion(t_v, text_graph, steps=self.diffusion_steps)  # (B, N, hidden_dim)

        for layer in self.layers:
            # Q, t_v_proj = layer(Q, z_v_diffused, t_v_proj)
            # Q, t_v_proj = layer(Q, z_v_proj, t_v_proj)       
            Q, t_v_proj = layer(Q, z_v_diffused, t_v_diffused)     
            # Q, t_v_proj = layer(Q, z_v_proj, t_v_diffused) 
                   
        return Q, t_v_proj




class SelfAttentionModel(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()

        self.args = args
        self.context = args.context
        self.decoder_only = args.decoder_only
        self.neighbor_mode = args.neighbor_mode
        self.position_type = args.position_type
        self.n_text_tokens = args.n_text_tokens
        self.n_visual_tokens = args.n_visual_tokens
        self.n_virtual_tokens = args.n_virtual_tokens
        self.tokenizer = tokenizer
        self.max_input_length = args.max_input_length

        if "t5" in args.model_name_or_path:
            peft_task_type = TaskType.SEQ_2_SEQ_LM
            config = AutoConfig.from_pretrained(args.model_name_or_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, config=config)
        elif "opt" in args.model_name_or_path:
            peft_task_type = TaskType.CAUSAL_LM
            config = AutoConfig.from_pretrained(args.model_name_or_path)
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config)
        else:
            raise ValueError(f"SelfAttentionModel does not support {args.model_name_or_path}.")

        if args.peft_type == "none":
            self.lm = model
        else:
            if args.peft_type == "lora":
                peft_config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    modules_to_save=["lm_head"],
                )
            elif args.peft_type == "prefix":
                peft_config = PrefixTuningConfig(
                    task_type=peft_task_type,
                    inference_mode=False,
                    prefix_projection=True,
                    num_virtual_tokens=self.n_virtual_tokens
                )
            elif args.peft_type == "prompt":
                peft_config = PromptTuningConfig(
                    task_type=peft_task_type,
                    prompt_tuning_init=PromptTuningInit.RANDOM,
                    num_virtual_tokens=self.n_virtual_tokens
                )
            else:
                raise ValueError(f"SelfAttentionModel does not support {args.peft_type}.")
            self.lm = get_peft_model(model, peft_config)

        self.input_embeddings = self.lm.get_input_embeddings()
        hidden_dim = self.input_embeddings.embedding_dim
        num_heads = 8
        num_layers = 12
        dropout = 0.1

        self.text_model = None
        if self.neighbor_mode == "prefix":
            # Text model processing text neighbors
            embedding_dim = self.input_embeddings.embedding_dim * args.n_text_tokens
            # self.text_model = CLIPTextModel.from_pretrained(args.text_model)
            self.text_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config)
            self.text_embeddings = nn.Linear(self.text_model.config.hidden_size, embedding_dim)
            if self.position_type == "none":
                self.text_position_embeddings = nn.Embedding(args.max_output_length + 1, embedding_dim) # + 1 for padding neighbors

            self.text_model.eval()
            for name, param in self.text_model.named_parameters():
                param.requires_grad = False
        

        self.visual_model = None
        if self.context in ("section_all", "all"):
            embedding_dim = self.input_embeddings.embedding_dim * args.n_text_tokens
            self.text_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config)
            self.text_embeddings = nn.Linear(self.text_model.config.hidden_size, embedding_dim)
            if self.position_type == "none":
                self.text_position_embeddings = nn.Embedding(args.max_output_length + 1, embedding_dim) # + 1 for padding neighbors

            self.text_model.eval()
            for name, param in self.text_model.named_parameters():
                param.requires_grad = False
                
            self.n_heads = 12
            embedding_dim = self.input_embeddings.embedding_dim
            self.visual_model = CLIPVisionModel.from_pretrained(args.visual_model)
            self.visual_embeddings = nn.Linear(self.text_model.config.hidden_size, embedding_dim)
            self.visual_qformer = MultiLayerTranslator(self.visual_model.config.hidden_size, self.text_model.config.hidden_size, hidden_dim=self.text_model.config.hidden_size, num_query_tokens= args.max_image_neighbors * args.n_visual_tokens, num_heads= self.n_heads, num_layers=1)
            
            
            if self.position_type == "none":
                self.visual_position_embeddings = nn.Embedding(args.max_output_length + 1, embedding_dim) # + 1 for padding neighbors
                self.text_hop_embedding = nn.Embedding(50 + 1, self.text_model.config.hidden_size)  # +1 for padding (-1 or 0 index)
                self.image_hop_embedding = nn.Embedding(args.max_image_neighbors + 5 , self.visual_model.config.hidden_size)  # +1 for padding
                
            self.visual_model.eval()
            for param in self.visual_model.parameters():
                param.requires_grad = False

        if self.position_type == "laplacian":
            if self.context in ("section_only", "section_all", "text_only") or self.neighbor_mode == "raw":
                raise ValueError(f"[Laplacian PE] neighbor mode: {self.neighbor_mode} and context: {self.context} are not supported.")
            k = 1 + args.max_text_neighbors + args.max_image_neighbors - 5
            embedding_dim = self.input_embeddings.embedding_dim * args.n_text_tokens
            self.lpe_embeddings = nn.Linear(k, embedding_dim)

        if self.neighbor_mode != 'raw' and self.position_type == "gnn":
            embedding_dim = self.input_embeddings.embedding_dim * args.n_text_tokens
            self.gnn = GCN(input_dim=embedding_dim, output_dim=embedding_dim, hidden_dim=self.text_model.config.hidden_size)

        # Freeze the base LM
        if self.args.freeze_lm:
            print("Freezing the LM.")
            self.lm.eval()
            for param in self.lm.parameters():
                param.requires_grad = False
        else:
            self.lm.train()

    def get_text_embs(self, input_ids, attention_mask, pos_ids):
        batch_size, neighbor_num, seq_len = input_ids.shape
        input_ids = input_ids.reshape(-1, seq_len)
        attention_mask = attention_mask.reshape(-1, seq_len)

        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        encoder_outputs = outputs.pooler_output
        text_embs = self.text_embeddings(encoder_outputs)

        if self.position_type == "none":
            pos_ids = pos_ids.reshape(-1)
            text_embs = text_embs + self.text_position_embeddings(pos_ids)

        text_embs = text_embs.reshape(text_embs.shape[0], self.n_text_tokens, -1)
        return text_embs.reshape(batch_size, neighbor_num, self.n_text_tokens, -1)


    def get_visual_embs(self, input_ids, attention_mask, pixel_values, text_hop, img_hop, text_neighbor, neighbor_attention, text_graph, img_graph):
        # === FIX: Handle section_all mode - early return với chỉ visual processing ===
        if self.context == "section_all":
            # Chỉ process visual của section hiện tại, không cần neighbors
            batch_size, visual_neighbor_num, pixel, width, height = pixel_values.shape
            pixel_values = pixel_values.reshape(-1, pixel, width, height)
            visual_outputs = self.visual_model(pixel_values)
            visual_encoder_outputs = visual_outputs.pooler_output
            visual_embs = visual_encoder_outputs.reshape(batch_size, visual_neighbor_num, -1)
            
            batch_size, seq_len = input_ids.shape  
            text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, 
                                        output_hidden_states=True).hidden_states[-1]
            text_outputs = text_outputs.mean(dim=1)  # Pool to get section representation
            text_outputs = text_outputs.unsqueeze(1).expand(-1, visual_neighbor_num, -1)  # Match visual dims
            
            # Process qua qformer với empty graphs
            empty_img_graph = None
            empty_text_graph = None
            Q_tokens, _ = self.visual_qformer(visual_embs, text_outputs, empty_img_graph, empty_text_graph)
            visual_embs = self.visual_embeddings(Q_tokens)
            
            return visual_embs.reshape(batch_size, visual_neighbor_num, self.n_visual_tokens, -1)
        
        # === ORIGINAL CODE cho "all" mode ===
        # Self Attention on input_ids
        batch_size, num_neighbors, seq_len = text_neighbor.shape
        text_neighbor = text_neighbor.view(batch_size * num_neighbors, seq_len)
        neighbor_attention = neighbor_attention.view(batch_size * num_neighbors, seq_len)
        text_outputs = self.text_model(input_ids=text_neighbor, attention_mask=neighbor_attention, output_hidden_states=True).hidden_states[-1] # (B*N, seq, D)
        text_outputs = text_outputs.mean(dim = 1)
        text_outputs = text_outputs.view(batch_size, num_neighbors, -1) 

        ## HE on input_ids
        # batch_size, seq_len = input_ids.shape
        # text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # text_encoder_outputs = text_outputs.hidden_states[-1][:,:self.max_input_length,:] # (B, seq_len, D)
        # text_hop_embs = self.text_hop_embedding(text_hop)  # (B, seq_len, D)
        # text_encoder_outputs[:,:self.max_input_length,:] = text_encoder_outputs[:,:self.max_input_length,:] + text_hop_embs  # Add hop embeddings
        
        # Process visual neighbors
        batch_size, visual_neighbor_num, pixel, width, height = pixel_values.shape
        pixel_values = pixel_values.reshape(-1, pixel, width, height)  # (B * visual_neighbor_num, pixel, H, W)
        visual_outputs = self.visual_model(pixel_values)
        visual_encoder_outputs = visual_outputs.pooler_output  # (B * visual_neighbor_num, D)
        visual_embs = visual_encoder_outputs.reshape(batch_size, visual_neighbor_num, -1)  # (B, visual_neighbor_num, D)
        # image_hop_embs = self.image_hop_embedding(img_hop)  # (B, visual_neighbor_num, D)
        # visual_embs += image_hop_embs  # Add hop embeddings
        
        # Q_tokens, _ = self.visual_qformer(visual_embs, text_encoder_outputs, img_graph, text_graph)
        Q_tokens, _ = self.visual_qformer(visual_embs, text_outputs, img_graph, text_graph)
        visual_embs = self.visual_embeddings(Q_tokens)  # (B, visual_neighbor_num, embedding_dim)
        # Return reshaped embeddings
        return visual_embs.reshape(batch_size, visual_neighbor_num, self.n_visual_tokens, -1)
        


    def train(self, mode=True):
        super(SelfAttentionModel, self).train(mode=mode)
        if self.args.freeze_lm:
            self.lm.eval()
        if self.text_model is not None:
            self.text_model.eval()
        if self.visual_model is not None:
            self.visual_model.eval()

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        images=None,
        image_positions=None,
        neighbor_input_ids=None,
        neighbor_attention_mask=None,
        neighbor_pos_ids=None,
        text_locations=None,
        neighbor_images=None,
        neighbor_images_pos_ids=None,
        image_locations=None,
        lpe=None,
        graph=None,
        text_hop = None,
        img_hop = None,
        text_graph = None,
        img_graph = None
    ):

        if self.neighbor_mode == "raw" and self.context in ("section_only", "text_only"):
            return self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        elif self.neighbor_mode == "raw" and self.context in ("section_all", "all"):
            input_embs = self.input_embeddings(input_ids)
            visual_embs = self.get_visual_embs(input_ids, attention_mask, images, text_hop, img_hop, neighbor_input_ids, neighbor_attention_mask, text_graph, img_graph)
            
            batch_size, seq_len, hidden_dim = input_embs.shape
            if self.context == "section_all":
                batch_idx = torch.arange(batch_size)[:, None]
                input_embs[batch_idx, image_positions] = visual_embs.reshape(batch_size, -1, hidden_dim)
                if self.decoder_only:
                    labels[batch_idx, image_positions] = -100
            else:
                for batch_idx in range(batch_size):
                    for image_idx in range(images.shape[1]):
                        image_position = image_positions[batch_idx][self.n_visual_tokens * image_idx: self.n_visual_tokens * (image_idx + 1)]
                        if image_position.sum() == -1 * self.n_visual_tokens:
                            continue
                        input_embs[batch_idx, image_position] = visual_embs[batch_idx, image_idx]
                        if self.decoder_only:
                            labels[batch_idx, image_position] = -100

            return self.lm(inputs_embeds=input_embs, attention_mask=attention_mask, labels=labels)
 
        else:
            raise ValueError(f"Neighbor mode: {self.neighbor_mode} and context: {self.context} are not supported.")

