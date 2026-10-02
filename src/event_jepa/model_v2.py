from __future__ import annotations

import torch
import torch.nn.functional as F

from event_jepa.model import (
    EventJEPA,
    cosine_jepa_loss,
)


def permute_history_keep_current(
    context: torch.Tensor,
) -> torch.Tensor:
    """
    Construct a wrong-order context while keeping:
      1. exactly the same historical frames;
      2. exactly the same current frame;
      3. only changing temporal order.

    Tc=4:
      normal : [t-3, t-2, t-1, t]
      wrong  : [t-2, t-1, t-3, t]

    Input:
      context [B,T,N,D]
    """
    if context.ndim != 4:
        raise ValueError(
            "context must have shape [B,T,N,D]"
        )

    T = context.shape[1]

    if T < 3:
        raise ValueError(
            "order objective requires at least "
            "3 context frames"
        )

    #
    # Cyclically rotate history while leaving
    # the latest/current frame untouched.
    #
    history = context[:, :-1]

    wrong_history = torch.roll(
        history,
        shifts=-1,
        dims=1,
    )

    return torch.cat(
        [
            wrong_history,
            context[:, -1:],
        ],
        dim=1,
    )


def per_sample_cosine_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    prediction/target:
        [B,K,N,D]

    returns:
        [B]
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"shape mismatch: "
            f"{prediction.shape} vs {target.shape}"
        )

    distance = (
        1.0
        - F.cosine_similarity(
            prediction.float(),
            target.detach().float(),
            dim=-1,
        )
    )

    return distance.mean(
        dim=(1, 2)
    )


def residual_jepa_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
) -> torch.Tensor:
    """
    Explicitly predict latent CHANGE rather than
    allowing the objective to be dominated by
    future/current persistence.

    prediction : [B,K,N,D]
    target     : [B,K,N,D]
    current    : [B,N,D]
    """

    if current.ndim != 3:
        raise ValueError(
            "current must have shape [B,N,D]"
        )

    current = current[:, None]

    pred_delta = (
        prediction.float()
        - current.float()
    )

    target_delta = (
        target.detach().float()
        - current.detach().float()
    )

    return (
        1.0
        - F.cosine_similarity(
            pred_delta,
            target_delta,
            dim=-1,
        )
    ).mean()


def history_order_ranking_loss(
    correct_prediction: torch.Tensor,
    wrong_prediction: torch.Tensor,
    target: torch.Tensor,
    margin: float,
):
    """
    Require correct chronological history to predict
    the future better than the same frames in a
    wrong temporal order.

        d(correct) + margin < d(wrong)

    loss = max(0, margin + d_correct - d_wrong)

    Returns:
        loss
        mean order gap = d_wrong - d_correct

    Positive order_gap means correct history is better.
    """

    if margin < 0:
        raise ValueError(
            "order margin must be nonnegative"
        )

    d_correct = per_sample_cosine_distance(
        correct_prediction,
        target,
    )

    d_wrong = per_sample_cosine_distance(
        wrong_prediction,
        target,
    )

    loss = F.relu(
        margin
        + d_correct
        - d_wrong
    ).mean()

    order_gap = (
        d_wrong
        - d_correct
    ).mean()

    return loss, order_gap


class OrderAwareResidualEventJEPA(EventJEPA):
    """
    V2 Event-JEPA.

    Architecture:
        EXACTLY the same online encoder,
        target encoder and predictor as V1.

    Objective:
        L =
            L_future
          + lambda_delta * L_residual
          + lambda_order * L_order

    No additional learnable parameters are introduced.
    """

    def __init__(
        self,
        *args,
        residual_weight: float = 0.5,
        order_weight: float = 0.5,
        order_margin: float = 0.005,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        if residual_weight < 0:
            raise ValueError(
                "residual_weight must be >= 0"
            )

        if order_weight < 0:
            raise ValueError(
                "order_weight must be >= 0"
            )

        if order_margin < 0:
            raise ValueError(
                "order_margin must be >= 0"
            )

        self.residual_weight = float(
            residual_weight
        )

        self.order_weight = float(
            order_weight
        )

        self.order_margin = float(
            order_margin
        )

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        delta_t: torch.Tensor,
    ):
        #
        # -------------------------------------------------
        # 1. Normal future prediction — identical to V1.
        # -------------------------------------------------
        #
        context_latent = (
            self.online_encoder(
                context
            )
        )

        prediction = self.predictor(
            context_latent,
            delta_t,
        )

        B, K, N, D = target.shape

        #
        # -------------------------------------------------
        # 2. EMA future target — identical to V1.
        # -------------------------------------------------
        #
        with torch.no_grad():

            encoded_target = (
                self.target_encoder(
                    target.reshape(
                        B * K,
                        1,
                        N,
                        D,
                    )
                )
            )

            encoded_target = (
                encoded_target.reshape(
                    B,
                    K,
                    N,
                    D,
                )
            )

        future_loss = cosine_jepa_loss(
            prediction,
            encoded_target,
        )

        #
        # -------------------------------------------------
        # 3. Residual dynamics objective.
        # -------------------------------------------------
        #
        residual_loss = (
            future_loss.new_zeros(())
        )

        if self.residual_weight > 0:

            #
            # Encode current t using the SAME EMA target
            # encoder / same one-frame latent space as
            # future targets.
            #
            with torch.no_grad():

                encoded_current = (
                    self.target_encoder(
                        context[:, -1:]
                    )
                )

            residual_loss = (
                residual_jepa_loss(
                    prediction,
                    encoded_target,
                    encoded_current,
                )
            )

        #
        # -------------------------------------------------
        # 4. Temporal order objective.
        # -------------------------------------------------
        #
        order_loss = (
            future_loss.new_zeros(())
        )

        order_gap = (
            future_loss.new_zeros(())
        )

        if self.order_weight > 0:

            wrong_context = (
                permute_history_keep_current(
                    context
                )
            )

            wrong_memory = (
                self.online_encoder(
                    wrong_context
                )
            )

            wrong_prediction = (
                self.predictor(
                    wrong_memory,
                    delta_t,
                )
            )

            (
                order_loss,
                order_gap,
            ) = history_order_ranking_loss(
                prediction,
                wrong_prediction,
                encoded_target,
                margin=self.order_margin,
            )

        #
        # -------------------------------------------------
        # 5. Total V2 loss.
        # -------------------------------------------------
        #
        loss = (
            future_loss
            + self.residual_weight
            * residual_loss
            + self.order_weight
            * order_loss
        )

        normalized_prediction = F.normalize(
            prediction.detach().float(),
            dim=-1,
        )

        normalized_target = F.normalize(
            encoded_target.detach().float(),
            dim=-1,
        )

        mean_cosine = (
            normalized_prediction
            * normalized_target
        ).sum(
            dim=-1
        ).mean()

        return {
            "loss":
                loss,

            "future_loss":
                future_loss,

            "residual_loss":
                residual_loss,

            "order_loss":
                order_loss,

            "order_gap":
                order_gap,

            "prediction":
                prediction,

            "target":
                encoded_target,

            "representation_std":
                encoded_target.float()
                .std(
                    dim=(0, 1, 2)
                )
                .mean(),

            "mean_cosine":
                mean_cosine,
        }
