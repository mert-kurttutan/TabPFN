import torch
from enum import Enum
from torch.nn import LayerNorm

class Activation(Enum):
    """Enum for activation functions."""

    GELU = 1
    RELU = 2
class MLP(torch.nn.Module):

    linear1: torch.nn.Linear
    linear2: torch.nn.Linear
    activation: Activation

    def __init__(
        self,
        size: int,
        hidden_size: int,
        activation: Activation | str,
        *,
        device: torch.device | None,
        dtype: torch.dtype | None,
        initialize_output_to_zero: bool = False,
        recompute: bool = False,
    ):
        super().__init__()
        self.linear1 = torch.nn.Linear(
            size,
            hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.linear2 = torch.nn.Linear(
            hidden_size,
            size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        if isinstance(activation, str):
            activation = Activation[activation.upper()]
        self.activation = activation
        if initialize_output_to_zero:
            torch.nn.init.zeros_(self.linear2.weight)
        if recompute:
            self.forward = partial(checkpoint, self.forward, use_reentrant=False)  # type: ignore


    def forward(
        self,
        x0: torch.Tensor,
        add_input: bool = False,
        allow_inplace: bool = False,
    ) -> torch.Tensor:
        x = self.linear1(x0)
        if self.activation is Activation.GELU:
            x = torch.nn.functional.gelu(x)
        elif self.activation is Activation.RELU:
            x = torch.nn.functional.relu(x)
        else:
            raise NotImplementedError(
                f"Activation Function {self.activation} is not implemented.",
            )

        x = self.linear2(x)
        if add_input:
            x += x0
        return x

class PerFeatureEncoderLayer(Module):
    __constants__: ClassVar = ["batch_first"]

    def __init__(  # noqa: PLR0913
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int | None = None,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        pre_norm: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        recompute_attn: bool = False,
        second_mlp: bool = False,
        layer_norm_with_elementwise_affine: bool = False,
        zero_init: bool = False,
        save_peak_mem_factor: int | None = None,
        attention_between_features: bool = True,
        multiquery_item_attention: bool = False,
        multiquery_item_attention_for_test_set: bool = False,
        two_sets_of_queries: bool = False,
        attention_init_gain: float = 1.0,
        d_k: int | None = None,
        d_v: int | None = None,
        precomputed_kv: None | torch.Tensor | tuple[torch.Tensor, torch.Tensor] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        assert d_model % nhead == 0 or (d_k is not None and d_v is not None)
        if multiquery_item_attention_for_test_set and multiquery_item_attention:
            raise ValueError(
                "Cannot use both multiquery_item_attention_for_test_set"
                "and multiquery_item_attention",
            )
        if two_sets_of_queries and not multiquery_item_attention_for_test_set:
            raise ValueError(
                "two_sets_of_queries requires multiquery_item_attention_for_test_set",
            )

        if d_k is None:
            d_k = d_model // nhead

        if d_v is None:
            d_v = d_model // nhead

        self.self_attn_between_features: MultiHeadAttention | None = None
        if attention_between_features:
            self.self_attn_between_features = MultiHeadAttention(
                input_size=d_model,
                output_size=d_model,
                d_k=d_k,
                d_v=d_v,
                nhead=nhead,
                device=device,
                dtype=dtype,
                initialize_output_to_zero=zero_init,
                recompute=recompute_attn,
                init_gain=attention_init_gain,
            )

        if isinstance(precomputed_kv, tuple):
            precomputed_k, precomputed_v = precomputed_kv
            precomputed_kv = None
        else:
            precomputed_k = precomputed_v = None

        self.self_attn_between_items = MultiHeadAttention(
            input_size=d_model,
            output_size=d_model,
            d_k=d_k,
            d_v=d_v,
            nhead=nhead,
            device=device,
            dtype=dtype,
            share_kv_across_n_heads=nhead if multiquery_item_attention else 1,
            initialize_output_to_zero=zero_init,
            recompute=recompute_attn,
            precomputed_k=precomputed_k,
            precomputed_v=precomputed_v,
            precomputed_kv=precomputed_kv,
            init_gain=attention_init_gain,
            two_sets_of_queries=(
                multiquery_item_attention_for_test_set and two_sets_of_queries
            ),
        )

        if dim_feedforward is None:
            dim_feedforward = 2 * d_model

        self.mlp = MLP(
            size=d_model,
            hidden_size=dim_feedforward,
            activation=activation,
            device=device,
            dtype=dtype,
            initialize_output_to_zero=zero_init,
            recompute=recompute_attn,
        )

        self.layer_norms = nn.ModuleList(
            [
                LayerNorm(
                    d_model,  # type: ignore
                    layer_norm_eps,
                    elementwise_affine=layer_norm_with_elementwise_affine,
                    **factory_kwargs,
                )
                for _ in range(4 if second_mlp else 3)
            ],
        )

        self.second_mlp: MLP | None = None
        if second_mlp:
            self.second_mlp = MLP(
                size=d_model,
                hidden_size=dim_feedforward,
                activation=activation,
                device=device,
                dtype=dtype,
                initialize_output_to_zero=zero_init,
                recompute=recompute_attn,
            )

        self.pre_norm = pre_norm
        self.recompute_attn = recompute_attn
        self.save_peak_mem_factor = save_peak_mem_factor
        self.multiquery_item_attention_for_test_set = (
            multiquery_item_attention_for_test_set
        )
        self.two_sets_of_queries = two_sets_of_queries

    def __setstate__(self, state: dict[str, Any]) -> None:
        state.setdefault("save_peak_mem_factor", None)
        super().__setstate__(state)

    def forward(  # noqa: C901
        self,
        state: Tensor,
        single_eval_pos: int | None = None,
        *,
        cache_trainset_representation: bool = False,
        att_src: Tensor | None = None,
    ) -> Tensor:
        """Pass the input through the encoder layer.

        Args:
            state:
                The transformer state passed as input to the layer of shape
                (batch_size, num_items, num_feature_blocks, d_model).
            single_eval_pos:
                The position from which on everything is treated as test
                set.
            cache_trainset_representation:
                Whether to cache the trainset representation.
                If single_eval_pos is set (> 0 and not None), create a cache of the
                trainset KV. This may require a lot of memory. Otherwise, use
                cached KV representations for inference.
            att_src:
                The tensor to attend to from the final layer of the encoder.
                It has a shape of
                (batch_size, num_train_items, num_feature_blocks, d_model).
                This does not work with multiquery_item_attention_for_test_set and
                cache_trainset_representation at this point.

        Returns:
            The transformer state passed through the encoder layer.
        """
        assert (
            len(state.shape) == 4
        ), "src must be of shape (batch_size, num_items, num feature blocks, d_model)"
        if single_eval_pos is None:
            single_eval_pos = 0

        save_peak_mem_factor = self.save_peak_mem_factor
        if cache_trainset_representation and not single_eval_pos:
            assert self.self_attn_between_items.has_cached_kv
            save_peak_mem_factor = None

        if att_src is not None:
            assert (
                not self.multiquery_item_attention_for_test_set
            ), "Not implemented yet."
            assert not cache_trainset_representation, "Not implemented yet."
            assert not single_eval_pos, (
                "single_eval_pos should not be set, as the train representation"
                " is in att_src"
            )

        if self.self_attn_between_features is None:
            assert not cache_trainset_representation, "Not implemented yet."
            assert state.shape[2] == 1, (
                f"One group architecture expects one feature group, "
                f"but got {state.shape[2]} feature groups."
            )

        def attn_between_features(x: torch.Tensor) -> torch.Tensor:
            assert self.self_attn_between_features is not None
            return self.self_attn_between_features(
                x,
                save_peak_mem_factor=save_peak_mem_factor,
                add_input=True,
                allow_inplace=True,
            )

        def attn_between_items(x: torch.Tensor) -> torch.Tensor:
            # we need to transpose as self attention always treats
            # dim -2 as the sequence dimension
            if self.multiquery_item_attention_for_test_set:
                if single_eval_pos < x.shape[1]:
                    new_x_test = self.self_attn_between_items(
                        x[:, single_eval_pos:].transpose(1, 2),
                        x[:, :single_eval_pos].transpose(1, 2)
                        if single_eval_pos
                        else None,
                        save_peak_mem_factor=save_peak_mem_factor,
                        cache_kv=False,
                        add_input=True,
                        allow_inplace=True,
                        use_cached_kv=not single_eval_pos,
                        reuse_first_head_kv=True,
                        use_second_set_of_queries=self.two_sets_of_queries,
                    ).transpose(1, 2)
                else:
                    new_x_test = None

                if single_eval_pos:
                    new_x_train = self.self_attn_between_items(
                        x[:, :single_eval_pos].transpose(1, 2),
                        x[:, :single_eval_pos].transpose(1, 2),
                        save_peak_mem_factor=save_peak_mem_factor,
                        cache_kv=cache_trainset_representation,
                        only_cache_first_head_kv=True,
                        add_input=True,
                        allow_inplace=True,
                        use_cached_kv=False,
                    ).transpose(1, 2)
                else:
                    new_x_train = None

                return torch.cat(
                    [x_ for x_ in [new_x_train, new_x_test] if x_ is not None],
                    dim=1,
                )

            attention_src_x = None
            if att_src is not None:
                attention_src_x = att_src.transpose(1, 2)
            elif single_eval_pos:
                attention_src_x = x[:, :single_eval_pos].transpose(1, 2)

            return self.self_attn_between_items(
                x.transpose(1, 2),
                attention_src_x,
                save_peak_mem_factor=save_peak_mem_factor,
                cache_kv=cache_trainset_representation and single_eval_pos,
                add_input=True,
                allow_inplace=True,
                use_cached_kv=cache_trainset_representation and not single_eval_pos,
            ).transpose(1, 2)

        mlp_save_peak_mem_factor = (
            save_peak_mem_factor * 8 if save_peak_mem_factor is not None else None
        )

        sublayers = []
        if self.self_attn_between_features is not None:
            sublayers.append(attn_between_features)
        else:
            assert state.shape[2] == 1, (
                "If there is no attention between features, the number of feature"
                " blocks must be 1."
            )

        sublayers += [
            attn_between_items,
            partial(
                self.mlp.__call__,
                save_peak_mem_factor=mlp_save_peak_mem_factor
                if (
                    mlp_save_peak_mem_factor is not None
                    and state.numel() // state.shape[-1] // mlp_save_peak_mem_factor
                )
                > 32
                else None,
                add_input=True,
                allow_inplace=True,
            ),
        ]

        if self.second_mlp is not None:
            sublayers.insert(
                1,
                partial(
                    self.second_mlp.__call__,
                    save_peak_mem_factor=mlp_save_peak_mem_factor,
                    add_input=True,
                    allow_inplace=True,
                ),
            )

        for sublayer, layer_norm in zip(sublayers, self.layer_norms):
            if self.pre_norm:
                raise AssertionError(
                    "Pre-norm implementation is wrong, as the residual should never"
                    " be layer normed here.",
                )
                state = layer_norm(
                    state,
                    allow_inplace=True,
                    save_peak_mem_factor=save_peak_mem_factor,
                )

            state = sublayer(state)
            if not self.pre_norm:
                state = layer_norm(
                    state,
                    allow_inplace=True,
                    save_peak_mem_factor=save_peak_mem_factor,
                )

        return state

    def empty_trainset_representation_cache(self) -> None:
        """Empty the trainset representation cache."""
        self.self_attn_between_items.empty_kv_cache()

        # TODO: This could be None but this just ignored that fact here.
        assert self.self_attn_between_features is not None
        self.self_attn_between_features.empty_kv_cache()  # not necessary, just in case
