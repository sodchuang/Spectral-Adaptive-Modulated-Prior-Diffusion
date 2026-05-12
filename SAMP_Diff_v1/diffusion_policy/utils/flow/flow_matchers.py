"""Flow matcher implementations.

Adapted from A2A Flow Matching (roboverse_learn/il/utils/flow/flow_matchers.py),
re-packaged for the SAMP_Diff_v1 pipeline that uses Freqpolicy's MAE Transformer
as the velocity network backbone instead of SimpleFlowNet.

Dependency: torchcfm  (pip install torchcfm)
"""
import numpy as np
import torch
import torchcfm.conditional_flow_matching as cfm

from diffusion_policy.utils.flow.base_flow_matcher import BaseFlowMatcher


class TorchFlowMatcher(BaseFlowMatcher):
    """Generic wrapper around a torchcfm flow-matching object.

    The wrapped ``self.fm`` object must expose::

        sample_location_and_conditional_flow(x0, x1)
            -> (t, x_t, u_t)

    where
      - t  : sampled time  (B,)   ∈ [0, 1]
      - x_t: interpolated point
      - u_t: target velocity (x_1 - x_0 for straight CFM)
    """

    def __init__(self, fm, num_sampling_steps: int = 6):
        self.fm = fm
        self.num_sampling_steps = num_sampling_steps

    def compute_loss(self, model, target: torch.Tensor, start=None, **kwargs):
        """Compute flow-matching training loss.

        Args:
            model   : Callable ``(x_t, t, **kwargs) -> v_t``.
            target  : Ground-truth end-point x_1, shape ``(B, ...)``.
            start   : Optional source distribution x_0.  When *None* a
                      standard-normal sample is used (vanilla CFM); when
                      provided this implements the A2A warm-start strategy.
            **kwargs: Forwarded to ``model`` (e.g. ``global_cond``).

        Returns:
            ``(loss, {'loss': float})``
        """
        x0 = torch.randn_like(target) if start is None else start
        t, x_t, u_t = self.fm.sample_location_and_conditional_flow(x0, target)
        v_t = model(x_t, t, **kwargs)
        loss = torch.mean((v_t - u_t) ** 2)
        return loss, {'loss': loss.item()}

    def sample(
        self,
        model,
        shape,
        device: torch.device,
        num_steps: int = None,
        return_traces: bool = False,
        start=None,
        **kwargs,
    ):
        """Euler-method ODE integration from x_0 to x_1.

        Args:
            model       : Callable ``(x_t, t, **kwargs) -> v_t``.
            shape       : Output shape ``(B, ...)``.
            device      : Target device.
            num_steps   : Number of Euler steps; defaults to
                          ``self.num_sampling_steps``.
            return_traces: If True, also return (traj_history, vel_history).
            start       : Optional warm-start tensor x_0.  When *None* a
                          fresh standard-normal sample is drawn.
            **kwargs    : Forwarded to ``model`` at each step.

        Returns:
            ``x`` (final sample), or ``(x, (traj_history, vel_history))``
            when ``return_traces=True``.
        """
        if num_steps is None:
            num_steps = self.num_sampling_steps
        x = torch.randn(shape, device=device) if start is None else start
        dt = 1.0 / num_steps

        if return_traces:
            traj_history = [x.detach().clone().cpu()]
            vel_history = [np.zeros_like(x.cpu().numpy())]

        for step in range(num_steps):
            t = torch.ones(x.shape[0], device=device) * (step / num_steps)
            v_t = model(x, t, **kwargs)
            x = x + v_t * dt

            if return_traces:
                traj_history.append(x.detach().clone().cpu())
                vel_history.append(v_t.detach().clone().cpu().numpy())

        if return_traces:
            return x, (traj_history, vel_history)
        return x


# ---------------------------------------------------------------------------
# Convenience subclasses — thin wrappers that instantiate the underlying
# torchcfm object and forward all remaining kwargs to it.
# ---------------------------------------------------------------------------

class ConditionalFlowMatcher(TorchFlowMatcher):
    """Standard conditional flow matcher (straight paths)."""

    def __init__(self, num_sampling_steps: int = 6, **kwargs):
        super().__init__(cfm.ConditionalFlowMatcher(**kwargs), num_sampling_steps)


class TargetConditionalFlowMatcher(TorchFlowMatcher):
    """Target-conditional flow matcher."""

    def __init__(self, num_sampling_steps: int = 6, **kwargs):
        super().__init__(cfm.TargetConditionalFlowMatcher(**kwargs), num_sampling_steps)


class SchrodingerBridgeConditionalFlowMatcher(TorchFlowMatcher):
    """Schrödinger-bridge conditional flow matcher."""

    def __init__(self, num_sampling_steps: int = 6, **kwargs):
        super().__init__(
            cfm.SchrodingerBridgeConditionalFlowMatcher(**kwargs),
            num_sampling_steps,
        )


class ExactOptimalTransportConditionalFlowMatcher(TorchFlowMatcher):
    """Exact OT conditional flow matcher."""

    def __init__(self, num_sampling_steps: int = 6, **kwargs):
        super().__init__(
            cfm.ExactOptimalTransportConditionalFlowMatcher(**kwargs),
            num_sampling_steps,
        )
