# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Compression models or wrapper around existing models.
Also defines the main interface that a model must follow to be usable as an audio tokenizer.
"""

import logging
import math
import typing as tp
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import EncodecModel as HFEncodecModel

from .. import quantization as qt
from ..modules.lstm import StreamableLSTMforTimeCompression

logger = logging.getLogger()


class CompressionModel(ABC, nn.Module):
    """Base API for all compression model that aim at being used as audio tokenizers
    with a language model.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> qt.QuantizedResult:
        ...

    @abstractmethod
    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """See `EncodecModel.encode`."""
        ...

    @abstractmethod
    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        """See `EncodecModel.decode`."""
        ...

    @abstractmethod
    def decode_latent(self, codes: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        ...

    @property
    @abstractmethod
    def channels(self) -> int:
        ...

    @property
    @abstractmethod
    def frame_rate(self) -> float:
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        ...

    @property
    @abstractmethod
    def cardinality(self) -> int:
        ...

    @property
    @abstractmethod
    def num_codebooks(self) -> int:
        ...

    @property
    @abstractmethod
    def total_codebooks(self) -> int:
        ...

    @abstractmethod
    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        ...

    @staticmethod
    def get_pretrained(name: str, device: tp.Union[torch.device, str] = "cpu") -> "CompressionModel":
        """Instantiate a CompressionModel from a given pretrained model.

        Args:
            name (Path or str): name of the pretrained model. See after.
            device (torch.device or str): Device on which the model is loaded.

        Pretrained models:
            - dac_44khz (https://github.com/descriptinc/descript-audio-codec)
            - dac_24khz (same)
            - facebook/encodec_24khz (https://huggingface.co/facebook/encodec_24khz)
            - facebook/encodec_32khz (https://huggingface.co/facebook/encodec_32khz)
            - your own model on HugginFace. Export instructions to come...
        """

        from . import builders, loaders

        model: CompressionModel
        if name in ["dac_44khz", "dac_24khz"]:
            model_type = name.split("_")[1]
            logger.info("Getting pretrained compression model from DAC %s", model_type)
            model = DAC(model_type)
        elif name in ["debug_compression_model"]:
            logger.info("Getting pretrained compression model for debug")
            model = builders.get_debug_compression_model()
        elif Path(name).exists():
            # We assume here if the paths exist that it is in fact an AC checkpoint
            # that was exported using `audiocraft.utils.export` functions.
            model = loaders.load_compression_model(name, device=device)
        else:
            logger.info("Getting pretrained compression model from HF %s", name)
            hf_model = HFEncodecModel.from_pretrained(name)
            model = HFEncodecCompressionModel(hf_model).to(device)
        return model.to(device).eval()


class EncodecModel(CompressionModel):
    """Encodec model operating on the raw waveform.

    Args:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        quantizer (qt.BaseQuantizer): Quantizer network.
        frame_rate (int): Frame rate for the latent representation.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        causal (bool): Whether to use a causal version of the model.
        renormalize (bool): Whether to renormalize the audio before running the model.
    """

    # we need assignment to override the property in the abstract class,
    # I couldn't find a better way...
    frame_rate: float = 0
    sample_rate: int = 0
    channels: int = 0

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: qt.BaseQuantizer,
        frame_rate: int,
        sample_rate: int,
        channels: int,
        causal: bool = False,
        renormalize: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.channels = channels
        self.renormalize = renormalize
        self.causal = causal
        if self.causal:
            # we force disabling here to avoid handling linear overlap of segments
            # as supported in original EnCodec codebase.
            assert not self.renormalize, "Causal model does not support renormalize"

    @property
    def total_codebooks(self):
        """Total number of quantizer codebooks available."""
        return self.quantizer.total_codebooks

    @property
    def num_codebooks(self):
        """Active number of codebooks used by the quantizer."""
        return self.quantizer.num_codebooks

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        self.quantizer.set_num_codebooks(n)

    @property
    def cardinality(self):
        """Cardinality of each codebook."""
        return self.quantizer.bins

    def preprocess(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        scale: tp.Optional[torch.Tensor]
        if self.renormalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None
        return x, scale

    def postprocess(self, x: torch.Tensor, scale: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        if scale is not None:
            assert self.renormalize
            x = x * scale.view(-1, 1, 1)
        return x

    def forward(self, x: torch.Tensor) -> qt.QuantizedResult:
        assert x.dim() == 3
        length = x.shape[-1]
        x, scale = self.preprocess(x)

        emb = self.encoder(x)
        q_res = self.quantizer(emb, self.frame_rate)
        out = self.decoder(q_res.x)

        # remove extra padding added by the encoder and decoder
        assert out.shape[-1] >= length, (out.shape[-1], length)
        out = out[..., :length]

        q_res.x = self.postprocess(out, scale)

        return q_res

    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """Encode the given input tensor to quantized representation along with scale parameter.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T]

        Returns:
            codes, scale (tuple of torch.Tensor, torch.Tensor): Tuple composed of:
                codes a float tensor of shape [B, K, T] with K the number of codebooks used and T the timestep.
                scale a float tensor containing the scale for audio renormalizealization.
        """
        assert x.dim() == 3
        x, scale = self.preprocess(x)
        emb = self.encoder(x)
        codes = self.quantizer.encode(emb)
        return codes, scale

    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        """Decode the given codes to a reconstructed representation, using the scale to perform
        audio denormalization if needed.

        Args:
            codes (torch.Tensor): Int tensor of shape [B, K, T]
            scale (torch.Tensor, optional): Float tensor containing the scale value.

        Returns:
            out (torch.Tensor): Float tensor of shape [B, C, T], the reconstructed audio.
        """
        emb = self.decode_latent(codes)
        out = self.decoder(emb)
        out = self.postprocess(out, scale)
        # out contains extra padding added by the encoder and decoder
        return out

    def decode_latent(self, codes: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        return self.quantizer.decode(codes)


class EncodecfMRIModel(CompressionModel):
    """Encodec model operating on the raw waveform.

    Args:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        quantizer (qt.BaseQuantizer): Quantizer network.
        space_dim (int): Dimension of fMRI data.
        causal (bool): Whether to use a causal version of the model.
    """

    # we need assignment to override the property in the abstract class,
    # I couldn't find a better way...
    frame_rate: int = 0
    sample_rate: int = 0
    channels: int = 0

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: qt.BaseQuantizer,
        space_dim: int,
        frame_rate: int,
        causal: bool = False,
        without_tr: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer

        # embedding
        self.tr_emb_dim = encoder.tr_emb_dim
        self.tr_emb_pad_id = encoder.tr_emb_pad_id
        print ( "go to encodec.py and delet this ")
        print( "the tr_emb_pad_id is ", self.tr_emb_pad_id+ 1 , "the tr_emb_dim is ", self.tr_emb_dim)
        self.tr_embedding = nn.Embedding(self.tr_emb_pad_id + 1, self.tr_emb_dim, padding_idx=self.tr_emb_pad_id)
        self.space_dim = space_dim
        self.frame_rate = frame_rate
        self.causal = causal
        self.dimension = encoder.dimension
        self.add_nosie = encoder.add_noise
        if encoder.noise_lstm > 0:
            self.lstm_for_noise = StreamableLSTMforTimeCompression(
                self.dimension * 2,
                encoder.noise_lstm,
                bidirectional=encoder.bidiractional,
            )
        else:
            self.lstm_for_noise = None
        self.without_tr = without_tr

    # We need to override this property because the abstract class
    @property
    def total_codebooks(self):
        """Total number of quantizer codebooks available."""
        return self.quantizer.total_codebooks

    @property
    def num_codebooks(self):
        """Active number of codebooks used by the quantizer."""
        return self.quantizer.num_codebooks

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        self.quantizer.set_num_codebooks(n)

    @property
    def cardinality(self):
        """Cardinality of each codebook."""
        return self.quantizer.bins

    def decode_latent(self, codes: torch.Tensor, use_layer: tp.Optional[tp.List[int]]):
        """Decode from the discrete codes to continuous latent space."""
        return self.quantizer.decode(codes, use_layer)

    def forward(
        self,
        x: torch.Tensor,
        tr: torch.Tensor = None,
        noise_disable: bool = False,
    ) -> qt.QuantizedResult:
        print('the x dim is ', x.shape)
        assert x.dim() == 3
        length = x.shape[-1]
        print(' go to codec and delete this ')
        print("the without tr is ", self.without_tr)
        
        if tr is None : 
            print(' ohhhhhhhhh my god !! i will add this for a while but pay attention, you have to change it later ( go to encodec to do it ) : ')
            tr=torch.zeros(x.shape[0],x.shape[2])
            tr = tr.long()
            tr=tr.to(x.device)
        if self.without_tr:
            tr = None
        else:
            tr = self.tr_embedding(tr)
        print( f' the shape of x is {x.shape} and the shape of tr is {tr.shape}')
        
        emb = self.encoder(x, tr)
        print( ' self.add_noise= ', self.add_nosie)
        
        if self.add_nosie:
             #  oooooh there is sth here !! you have to explore more the divisionnssssssssssssssssssssssssssssssss
            mu, logvar = torch.split(emb, [self.dimension, self.dimension], dim=1)
            if self.lstm_for_noise is not None:
                stats = torch.cat([mu, logvar], dim=1)
                stats = self.lstm_for_noise(stats)
                mu, logvar = torch.split(stats, [self.dimension, self.dimension], dim=1)
                mu = mu.unsqueeze(-1)
                logvar = logvar.unsqueeze(-1)
            if noise_disable is True:
                mu = logvar = None
            q_res = self.quantizer(emb, self.frame_rate)
            out = self.decoder(None, tr, mu, logvar)  # VAE
        else:
            mu = logvar = None
            q_res = self.quantizer(emb, self.frame_rate)
            out = self.decoder(q_res.x, tr, mu, logvar)
                

        # remove extra padding added by the encoder and decoder
        assert out.shape[-1] >= length, (out.shape[-1], length)
        out = out[..., :length]

        q_res.x = out

        return q_res

    def encode(self, x: torch.Tensor, tr: torch.Tensor) -> torch.Tensor:
        """Encode the given input tensor to quantized representation along with scale parameter.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T]

        Returns:
            codes, scale (tuple of torch.Tensor, torch.Tensor): Tuple composed of:
                codes a float tensor of shape [B, K, T] with K the number of codebooks used and T the timestep.
                scale a float tensor containing the scale for audio renormalizealization.
        """
        assert x.dim() == 3
        if self.without_tr:
            tr = None
        else:
            tr = self.tr_embedding(tr)
        emb = self.encoder(x, tr)
        if self.add_nosie:
            mu, logvar = torch.split(emb, [self.dimension, self.dimension], dim=1)
            if self.lstm_for_noise is not None:
                stats = torch.cat([mu, logvar], dim=1)
                stats = self.lstm_for_noise(stats)
                mu, logvar = torch.split(stats, [self.dimension, self.dimension], dim=1)
                mu = mu.unsqueeze(-1)
                logvar = logvar.unsqueeze(-1)
            codes = None
        else:
            mu = logvar = None
            codes = self.quantizer.encode(emb)
        return codes, mu, logvar

    def decode(
        self,
        codes: torch.Tensor,
        tr: torch.Tensor,
        mu: tp.Optional[torch.Tensor],
        logvar: tp.Optional[torch.Tensor],
        noise_disable: bool = False,
        use_layer: tp.Optional[tp.List[int]] = None,
    ):
        """Decode the given codes to a reconstructed representation, using the scale to perform
        audio denormalization if needed.

        Args:
            codes (torch.Tensor): Int tensor of shape [B, K, T]
            scale (torch.Tensor, optional): Float tensor containing the scale value.

        Returns:
            out (torch.Tensor): Float tensor of shape [B, C, T], the reconstructed audio.
        """
        if self.without_tr:
            tr = None
        else:
            tr = self.tr_embedding(tr)
        if self.add_nosie:
            out = self.decoder(None, tr, mu, logvar)
        else:
            emb = self.decode_latent(codes, use_layer)
            if noise_disable is True:
                mu = logvar = None
            out = self.decoder(emb, tr, mu, logvar)
        return out


class DAC(CompressionModel):
    def __init__(self, model_type: str = "44khz"):
        super().__init__()
        try:
            import dac.utils
        except ImportError:
            raise RuntimeError(
                "Could not import dac, make sure it is installed, " "please run `pip install descript-audio-codec`"
            )
        self.model = dac.utils.load_model(model_type=model_type)
        self.n_quantizers = self.total_codebooks
        self.model.eval()

    def forward(self, x: torch.Tensor) -> qt.QuantizedResult:
        # We don't support training with this.
        raise NotImplementedError("Forward and training with DAC not supported.")

    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        codes = self.model.encode(x, self.n_quantizers)[1]
        return codes, None

    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        assert scale is None
        z_q = self.decode_latent(codes)
        return self.model.decode(z_q)

    def decode_latent(self, codes: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        return self.model.quantizer.from_codes(codes)[0]

    @property
    def channels(self) -> int:
        return 1

    @property
    def frame_rate(self) -> float:
        return self.model.sample_rate / self.model.hop_length

    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate

    @property
    def cardinality(self) -> int:
        return self.model.codebook_size

    @property
    def num_codebooks(self) -> int:
        return self.n_quantizers

    @property
    def total_codebooks(self) -> int:
        return self.model.n_codebooks

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        assert n >= 1
        assert n <= self.total_codebooks
        self.n_quantizers = n


class HFEncodecCompressionModel(CompressionModel):
    """Wrapper around HuggingFace Encodec."""

    def __init__(self, model: HFEncodecModel):
        super().__init__()
        self.model = model
        bws = self.model.config.target_bandwidths
        num_codebooks = [bw * 1000 / (self.frame_rate * math.log2(self.cardinality)) for bw in bws]
        deltas = [nc - int(nc) for nc in num_codebooks]
        # Checking we didn't do some bad maths and we indeed have integers!
        assert all(deltas) <= 1e-3, deltas
        self.possible_num_codebooks = [int(nc) for nc in num_codebooks]
        self.set_num_codebooks(max(self.possible_num_codebooks))

    def forward(self, x: torch.Tensor) -> qt.QuantizedResult:
        # We don't support training with this.
        raise NotImplementedError("Forward and training with HF EncodecModel not supported.")

    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        bandwidth_index = self.possible_num_codebooks.index(self.num_codebooks)
        bandwidth = self.model.config.target_bandwidths[bandwidth_index]
        res = self.model.encode(x, None, bandwidth)
        assert len(res[0]) == 1
        assert len(res[1]) == 1
        return res[0][0], res[1][0]

    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        if scale is None:
            scales = [None]  # type: ignore
        else:
            scales = scale  # type: ignore
        res = self.model.decode(codes[None], scales)
        return res[0]

    def decode_latent(self, codes: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        return self.model.quantizer.decode(codes.transpose(0, 1))

    @property
    def channels(self) -> int:
        return self.model.config.audio_channels

    @property
    def frame_rate(self) -> float:
        hop_length = int(np.prod(self.model.config.upsampling_ratios))
        return self.sample_rate / hop_length

    @property
    def sample_rate(self) -> int:
        return self.model.config.sampling_rate

    @property
    def cardinality(self) -> int:
        return self.model.config.codebook_size

    @property
    def num_codebooks(self) -> int:
        return self._num_codebooks

    @property
    def total_codebooks(self) -> int:
        return max(self.possible_num_codebooks)

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        if n not in self.possible_num_codebooks:
            raise ValueError(f"Allowed values for num codebooks: {self.possible_num_codebooks}")
        self._num_codebooks = n
