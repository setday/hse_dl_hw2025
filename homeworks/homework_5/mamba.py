'''
    Origin: https://github.com/myscience/mamba/blob/c1d5e755c83542e0c1c1d9f1f11d1b4c80af43db/mamba/mamba.py
'''

import math

import torch
from torch import Tensor
import torch.nn as nn
from torch import Tensor
from einops import einsum
from einops import rearrange
from transformers import PreTrainedTokenizerBase

from dataclasses import dataclass

from typing import Dict, List, Tuple, Generator
from torch.nn.functional import silu
from torch.nn.functional import softplus
from torch.nn.functional import softmax
from torch.nn.functional import cross_entropy

from pscan import pscan


@dataclass
class MambaConfig:
    '''
    Configuration class for the Mamba model.
    
    Attributes:
        vocab_size (int): Size of the vocabulary.
        num_layers (int): Number of Mamba blocks in the model.
        d_input (int): Dimension of the input sequence.
        d_model (int): Dimension of the model state space.
        d_state (int, optional): Dimension of the state space in the SSM stage. Defaults to 16.
        d_discr (int | None, optional): Dimension of the discrete space in the SSM stage. Defaults to None.
        ker_size (int, optional): Kernel size for the convolutional layer. Defaults to 4.
        parallel (bool, optional): Whether to use parallel scan for the SSM stage. Defaults to False.
    '''
    
    vocab_size : int
    num_layers : int
    d_input : int
    d_model : int
    d_state : int = 16
    d_discr : int | None = None
    ker_size : int = 4
    parallel : bool = False


class LLMMamba(nn.Module):
    '''
    Class representing the Mamba model as introduced in Gu & Dao (2023)
    (see paper: https://arxiv.org/abs/2312.00752). It is a State Space
    Model with context-dependent capability that matches the performances
    of the strongest Transformer competitor (albeit only tested for small
    scales) while being much more compute efficient.
    '''
    
    def __init__(
        self,
        config : MambaConfig,
    ) -> None:
        super().__init__()

        self.config = config
        
        # Needed embedding layer for mapping input tokens to the network
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_input
        )
        
        # A Mamba model is composed of a series of MambaBlocks interleaved
        # with normalization layers (e.g. RMSNorm)
        self.layers = nn.ModuleList([
            nn.ModuleList(
                [
                    MambaBlock(config),
                    nn.RMSNorm(config.d_input)
                ]
            )
            for _ in range(config.num_layers)
        ])
        
        self.norm_f = nn.RMSNorm(self.config.d_input)
        
        # Prediction head to map the output of the Mamba model to the vocabulary
        self.head = nn.Linear(config.d_input, config.vocab_size, bias=False)
        
    def forward(self, tok : Tensor, cache :  Tuple[Tensor, Tensor] | None = None) -> Tuple[Tensor,  Tuple[Tensor, Tensor] | None]:
        '''
        Forward pass of the Mamba model.
        
        Args:
            tok (Tensor): Input sequence of word tokens, has expected
                shape: (batch_size, seq_len).
            cache (Tensor, optional): Cache tensor to store the hidden states
                of the model. Default is None.
            
        Returns:
            Tensor: Predicted logits. If cache was provided return tensor has
                shape: (batch_size, vocab_size), while if no cache was provided
                output shape is: (batch_size, seq_len, vocab_size).
        '''
        
        tok = torch.atleast_2d(tok).to(self.device)
        seq = self.embedding(tok)
        
        for mamba, norm in self.layers: # type: ignore
            # Apply the MambaBlock and normalize the
            # output plus the residual connection
            out, cache = mamba(norm(seq), cache)
            seq = out + seq
            
        seq = self.norm_f(seq)
        logits = self.head(seq)

        return logits, cache
    
    @torch.no_grad()
    def generate(
        self,
        prompt : str | List[str],
        tokenizer : PreTrainedTokenizerBase, 
        token_lim : int = 300,
        use_top_k : int = 50,
        temperature : float = 1.0,
    ) -> Generator[Dict[int, str], None, None]:
        # Set model in evaluation model for inference
        self.eval()
        
        if isinstance(prompt, str):
            prompt = [prompt]
        if len(prompt) == 1:
            prompt = [prompt[0], prompt[0]] # To avoid an issue, connected with random tensor reshaping
        
        # Encode the prompt using the tokenizer
        inp = tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
        ).input_ids
        
        batch_size, inp_len = inp.shape
        vocab_size = tokenizer.vocab_size # type: ignore
        
        d_model, ker_size = self.config.d_model, self.config.ker_size
        cache = (None, torch.zeros(batch_size, d_model, ker_size - 1, device=self.device))
        
        # Consume the prompt to get the hidden states
        for tok in rearrange(inp, 'b s -> s b 1'):
            logits, cache = self(tok, cache)
        
        # Start generating the output sequence until either the maximum
        # token limit is reach or the model generates the<|endoftext|> token
        num_tokes = 0
        out, pred = [inp], tok
        pidx = torch.arange(batch_size).to(self.device).view(-1, 1)

        yield {int(pid.squeeze()) : tokenizer.decode(raw.squeeze(), skip_special_tokens=True) for pid, raw in zip(pidx, inp)}

        while num_tokes < token_lim and len(pred):
            logits, cache = self(pred, cache)
            
            # Get the token with the highest probability by zeroing out
            # the probability of the lowest probability tokens
            prob = softmax(logits[:, -1] / temperature, dim=-1)
            idxs = prob.topk(k=vocab_size - use_top_k, largest=False, sorted=False).indices
            prob.scatter_(dim=-1, index=idxs, src=torch.zeros_like(prob))
            prob /= prob.sum(dim=-1, keepdim=True)
            
            # Sample the next token from the distribution modelled by the llm
            pred = torch.multinomial(prob, num_samples=1, replacement=True)
            
            # Append the token to the input sequence
            out.append(pred)
            
            num_tokes += 1
            
            # Drop from the batch every prediction that reached the <|endoftext|> token
            mask = pred.squeeze() != tokenizer.eos_token_id

            pred  = pred[mask]
            pidx  = pidx[mask]
            cache = (cache[0][mask], cache[1][mask])
            
            # Yield the decoded tokens
            yield {int(pid.squeeze()) : tokenizer.decode(raw.squeeze(), skip_special_tokens=True) for pid, raw in zip(pidx, pred)}
        
        self.train()
    
    def compute_loss(self, prev : Tensor, post : Tensor) -> Tensor:
        # Compute model predictions for the previous tokens
        pred, _ = self(prev)

        pred = rearrange(pred, 'b s v -> (b s) v')
        post = rearrange(post, 'b s -> (b s)')
        
        # Compute the loss using the cross entropy loss
        loss = cross_entropy(pred, post)
        
        return loss
    
    @property
    def device(self) -> torch.device:
        '''
        Get the device of the model.

        Returns:
            torch.device: The device of the model.
        '''
        return next(self.parameters()).device

class MambaBlock(nn.Module):
    '''
    Class representing the MambaBlock as introduced in Gu & Dao (2023).
    '''
    
    def __init__(
        self, 
        config : MambaConfig,
    ) -> None:
        '''Initialize the Mamba model.

        Args:
            config (MambaConfig): Configuration object for the Mamba model.
        '''
        super().__init__()

        if config.d_discr is None: config.d_discr = config.d_model // 16

        # Projection matrices from the input sequence space to the
        # model state space (of dimension d_model) and back.
        # NOTE: The in_proj matrix has a factor of 2 because it is
        #       used to split the input sequence into two branches
        self.in_proj  = nn.Linear(config.d_input, 2 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_input, bias=False)

        # Projection matrices for endowing the SSM stage with
        # context-dependent capability (i.e. input dependence)
        self.s_B = nn.Linear(config.d_model, config.d_state, bias=False)
        self.s_C = nn.Linear(config.d_model, config.d_state, bias=False)
        self.s_D = nn.Sequential(
            nn.Linear(config.d_model, config.d_discr, bias=False), # Fixing matrix rank to d_disc
            nn.Linear(config.d_discr, config.d_model, bias=False),
        )

        dt_init_std = config.d_discr**-0.5 * 1.0
        nn.init.uniform_(self.s_D[1].weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(config.d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.s_D[1].bias.copy_(inv_dt)
        
        self.conv = nn.Conv1d(
            in_channels=config.d_model,
            out_channels=config.d_model,
            kernel_size=config.ker_size,
            padding=config.ker_size - 1,
            groups=config.d_model,
            bias=True,
        )
        
        # Parameters for the SSM. Follows the S4 initialization
        self.A = nn.Parameter(torch.log(torch.arange(1, config.d_state + 1, dtype=torch.float).repeat(config.d_model, 1)))
        self.A._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(config.d_model, dtype=torch.float))
        self.D._no_weight_decay = True

        # Whether to use or not the parallel scan for the SSM
        self.parallel = config.parallel

    def forward(self, seq : Tensor, cache :  Tuple[Tensor, Tensor] | None = None) -> Tuple[Tensor,  Tuple[Tensor, Tensor] | None]:
        '''
        Forward pass of the MambaBlock.
        
        Args:
            seq (Tensor): Input sequence of shape (batch_size, seq_len, d_seq).
            
        Returns:
            Tensor: Output sequence of shape (batch_size, seq_len, d_seq).
        '''
        b, l, d = seq.shape
        
        (prev_hid, prev_inp) = (None, None)
        if cache is not None: (prev_hid, prev_inp) = cache
        
        # Project the input sequence from d_seq to d_model and into two
        # distinct branches, one for the SSM and the residual branch
        # (see Fig. 3 of the Mamba paper). The resulting shapes are:
        # a: (batch_size, seq_len, d_model), b: (batch_size, seq_len, d_model)
        a, b = self.in_proj(seq).chunk(2, dim=-1)
        
        # * The SSM branch
        # Apply the convolutional layer to the SSM branch
        # NOTE: We need to move the channel dimension to the second dimension
        #       for the convolution to work properly, hence the rearrange
        x = rearrange(a, 'b l d -> b d l')

        x = x if prev_inp is None else torch.cat((prev_inp, x), dim=-1)
        a = self.conv(x)[..., :l] # Crop the output to the original length
        a = rearrange(a, 'b d l -> b l d')
        
        # Apply the SSM
        a = silu(a)
        a, hid = self.ssm(a, prev_hid=prev_hid) 
        
        # * The residual branch
        b = silu(b)
        
        # Combine the two branches
        out = a * b
        out =  self.out_proj(out)
        
        # Update the cache for next call if provided
        if cache:
            # Drop the first element of the hidden input states and attach
            # the newly computed results from the convolutions
            cache = (hid.squeeze(), x[..., 1:]) # type: ignore
        
        return out, cache
    
    def ssm(self, seq : Tensor, prev_hid : Tensor | None) -> Tuple[Tensor, Tensor]:
        '''
        State Space Model (SSM) of the MambaBlock.
        
        Args:
            seq (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        '''
        
        # Compute the context-dependent projections
        A = -torch.exp(self.A) # shape: (d_model, d_state)
        D = +self.D            # shape: (d_model, )
        
        B = self.s_B(seq)               # shape: (batch_size, seq_len, d_state)
        C = self.s_C(seq)               # shape: (batch_size, seq_len, d_state)
        Δ = softplus(self.s_D(seq))     # shape: (batch_size, seq_len, d_model)
        
        # Discretize the A and B parameters using Δ
        A_bar = einsum(A, Δ, 'd s,   b l d -> b l d s')
        B_bar = einsum(B, Δ, 'b l s, b l d -> b l d s')
        
        X_bar = einsum(B_bar, seq, 'b l d s, b l d -> b l d s')
        
        # Compute the state space hidden states
        # NOTE: This can be done either sequentially (slow) or with
        # a parallel scan (fast)
        hid = self._hid_states(
            torch.exp(A_bar),
            X_bar,
            parallel=self.parallel,
            prev_hid=prev_hid,    
        )
        
        # Compute the output based on the hidden states
        out = einsum(hid, C, 'b l d s, b l s -> b l d')
    
        out = out + D * seq
        
        return out, hid
    
    def _hid_states(
        self,
        A : Tensor,
        X : Tensor,
        parallel : bool = False,
        prev_hid : Tensor | None = None,
    ) -> Tensor:
        '''
        Calculate the hidden states of the SSM.

        Args:
            A (Tensor): The tensor representing A_bar.
            X (Tensor): The tensor representing X.
            parallel (bool): Whether to use parallel scan or 
                sequential computation (slower).

        Returns:
            Tensor: The tensor representing the hidden states.
        '''
        b, l, d, s = A.shape
        
        A = rearrange(A, 'b l d s -> l b d s')
        X = rearrange(X, 'b l d s -> l b d s')
        
        if prev_hid is not None:
            # If we have a previous hidden state it means we are running the
            # efficient auto-regressive inference, so we expect both A and X
            # to have a trivial length of 1, we just drop it when returning
            return rearrange(A * prev_hid + X, 'l b d s -> b l d s')
        
        h = None if parallel else torch.zeros(b, d, s, device=self.device)
        
        return pscan(A, X) if parallel else torch.stack([
            h := A_t * h + X_t
            for A_t, X_t in zip(A, X)
        ], dim=1)

    @property
    def device(self) -> torch.device:
        '''
        Get the device of the model.

        Returns:
            torch.device: The device of the model.
        '''
        return next(self.parameters()).device
